"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""

import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional

import numpy as np
import torch
import torchtune.generation as ttg
from transformers import AutoTokenizer

from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard
from exo.inference.tokenizers import _resolve_tokenizer
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
  ShardInferenceState
)

from exo.inference.torch.models.general_mha import ShardedGeneralModel

# from torchtune generate recipe
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml#L40
TEMP = 0.6
TOP_K = 35

class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Pytorch based inferece engine for sharded models
  """
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.sharded_model = None
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    self.oom_cnt = 0

    # cache settings
    self.use_cache = bool(os.getenv("TORCH_USE_CACHE", "True").lower() == "true")
    self.cache_setup = False

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    # rng setup for sampling
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(1234)

  def setup_cache(self, batch_size: int=1, total_response_length: int=1024):
    # setup cache
    # this is needed for a primary node that gets the initial encoding
    if not self.sharded_model.model.caches_are_enabled() and self.use_cache:
      if DEBUG >= 1:
        print(f"🔧 Setting up cache with batch_size={batch_size}, total_response_length={total_response_length}")
      with self.device:
        self.sharded_model.model.setup_caches(
          batch_size,
          self.model_config["torch_dtype"],
          decoder_max_seq_len=total_response_length
        )
      
      self.cache_setup = True
      if DEBUG >= 1:
        print(f"✅ Cache setup complete, cache_enabled={self.sharded_model.model.caches_are_enabled()}")


  def clear_model(self):
    """
    Clear out model and shard
    A way to avoid OOM issues
    
    All prompts are stored in VRAM
    while inference engine is up and using the same
    model class instance, this will clear it for each prompt.

    OOM issue might occur in longer chats/contexts depending on your machine.
    """
    if self.sharded_model.model.caches_are_enabled():
      self.sharded_model.model.reset_caches()
    
    del self.sharded_model
    self.sharded_model = None
    
    if self.device == torch.device("cuda"):
      torch.cuda.empty_cache()
    
    self.shard = None
    self.state = None

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      print("encode called")
      print(f"shard: {shard}")
      print(f"prompt: {prompt}")

    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      """
      Encode the tensors from prompt along with the
      initial input_pos and mask
      """
      tokens = self.tokenizer.encode(
        prompt,
        return_tensors="pt"
      )

      # move to proper device, default is CPU
      if tokens.device != self.device:
        tokens = tokens.to(device=self.device)
      
      if DEBUG >= 4:
        print("encoded_wrapper called")
        print(f"tokens: {tokens}")

      # if going past max, just take from max onward
      if len(tokens) > self.sharded_model.max_generated_tokens:
        max_gen_tokens = self.sharded_model.max_generated_tokens
        tokens = tokens[-max_gen_tokens:]

      self.state.tokens = tokens

      bsz, tklng = tokens.size()
      total_response_length = tklng + self.sharded_model.max_generated_tokens
      
      # For speculative decoding, we need extra cache space for draft tokens AND multiple iterations
      # Each speculative step can add gamma tokens, and we may have multiple iterations
      # Conservative estimate: 10 speculative iterations × gamma tokens per iteration
      max_speculative_iterations = 20  # Conservative estimate
      gamma = 10  # Conservative gamma estimate
      speculative_buffer = max_speculative_iterations * gamma
      total_response_length += speculative_buffer

      if DEBUG >= 1:
        print(f"🔧 Cache setup: tokens={tklng}, max_gen={self.sharded_model.max_generated_tokens}, buffer={speculative_buffer}, total={total_response_length}")
      self.setup_cache(bsz, total_response_length)
      
      # setup max sequence length
      if not self.sharded_model.model.caches_are_enabled():
        max_seq_len = total_response_length
      else:
        max_seq_len = self.sharded_model.model.decoder_max_cache_seq_len

      # set pad_id
      if hasattr(self.tokenizer, "pad_id"):
        pad_id = self.tokenizer.pad_id
      elif hasattr(self.tokenizer, "pad_token_id"):
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        if self.tokenizer.pad_token_id is not None:
          pad_id = self.tokenizer.pad_token_id
        else:
          pad_id = 0
      else:
        pad_id = 0
      
      padding_masks = tokens != pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.sharded_model.max_generated_tokens),
          value=True,
        )

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.state.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      return tokens

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(encode_wrapper),
    )

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      print("decode called")
      print(f"shard: {shard}")
      print(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"temp: {temp}")
      print(f"top_k: {top_k}")
      print(self.device)

    logits = torch.tensor(x).to(self.device)

    def sample_wrapper():
      q = torch.empty((logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings), device=logits.device).exponential_(1, generator=self.rng)

      tokens = ttg.sample(logits.clone(), temperature=temp, top_k=top_k, q=q.to(self.device))
      
      if DEBUG >= 4:
        print(f"tokens: {tokens}")

      return tokens.numpy(force=True)

    return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:

    await self.ensure_shard(shard)

    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")

    if inference_state is not None and inference_state.get("tokens") is not None:
      self.state.from_dict(inference_state)

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    input_tensor = None
    if input_data.ndim == 3:
      hidden_state = torch.tensor(input_data).to(
        device=self.device,
        dtype=self.model_config["torch_dtype"]
      )
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(
        device=self.device
      )

    if self.use_cache and not self.cache_setup:
      if input_tensor is not None:
        bsz, tklng = input_tensor.size()
        # Add speculative buffer for draft tokens and multiple iterations
        max_speculative_iterations = 20
        gamma = 10
        speculative_buffer = max_speculative_iterations * gamma
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens + speculative_buffer
        )
      else:
        bsz, tklng = self.state.tokens.size()
        # Add speculative buffer for draft tokens and multiple iterations
        max_speculative_iterations = 20
        gamma = 10
        speculative_buffer = max_speculative_iterations * gamma
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens + speculative_buffer
        )

    def infer_wrapper():
      if DEBUG >= 4:
        print(f"infer_wrapper called [{self.oom_cnt} OOM]")
        print(f"self.state: {self.state}")
        print(f"self.state.tokens: {self.state.tokens if self.state else 'State is None'}")
        print(f"hidden_state: {hidden_state}")

      model_cache = self.sharded_model.model.caches_are_enabled()

      # Handle case where state is None (e.g., in speculative decoding draft engine)
      if self.state is None:
        if DEBUG >= 2:
          print("PyTorch engine: State is None - creating new state for draft engine")
        self.state = ShardInferenceState()

      if self.state.tokens is not None:
        if input_data.ndim == 2 and input_tensor.size(-1) == 1:
          # Adding new tokens to existing sequence - update all state tensors
          old_seq_len = self.state.tokens.size(-1)
          self.state.tokens = torch.cat([
            self.state.tokens.to(self.device),
            input_tensor.clone()
          ], dim=-1).to(self.device)
        elif input_tensor.size(-1) > self.state.tokens.size(-1):
          # CRITICAL FIX: Extended sequence (e.g. speculative verification) - use full input
          if DEBUG >= 1:
            print(f"🔧 Extended sequence detected: {self.state.tokens.size(-1)} -> {input_tensor.size(-1)}")
          old_seq_len = self.state.tokens.size(-1)
          self.state.tokens = input_tensor.clone().to(self.device)
          new_seq_len = self.state.tokens.size(-1)
        else:
          # Same length - just use current state
          old_seq_len = self.state.tokens.size(-1)
          new_seq_len = old_seq_len
          
        # SMART LOGIC: Handle sequence length changes intelligently
          if self.state.input_pos is not None and self.state.mask is not None:
            if DEBUG >= 2:
              print(f"🔧 Smart state update: {old_seq_len} → {new_seq_len} tokens")
            
            if old_seq_len == new_seq_len:
              # Same length → just reset cache (preserve warmup)
              if model_cache and self.sharded_model.model.caches_are_enabled():
                if DEBUG >= 2:
                  print(f"   🚀 Same length: Fast cache reset only")
                self.sharded_model.model.reset_caches()
              # Keep existing input_pos and mask - they're still valid
            else:
              # Length changed → process incrementally to preserve context
              if DEBUG >= 2:
                print(f"   🔄 Length changed: Using incremental processing to preserve cache context")
              
              # Calculate how many tokens were added
              tokens_added = new_seq_len - old_seq_len
              
              if tokens_added <= 3:
                # Small addition (≤3 tokens) - try incremental processing to preserve cache
                if DEBUG >= 2:
                  print(f"   🔧 Small addition ({tokens_added} tokens) - preserving cache context")
                
                # Update curr_pos to point to where new tokens should be processed
                self.state.curr_pos = old_seq_len
                
                # Keep existing input_pos and mask - the generate() method will handle extension
              else:
                # Large addition (>3 tokens) - clear state for fresh setup
                if DEBUG >= 2:
                  print(f"   🔄 Large addition ({tokens_added} tokens) - clearing state for fresh setup")
                
                # Clear state to force fresh dimension setup for large jumps
                self.state.input_pos = None
                self.state.mask = None
                self.state.curr_pos = 0
                
                # Reset cache for large sequence changes
                if model_cache and self.sharded_model.model.caches_are_enabled():
                  self.sharded_model.model.reset_caches()
              if DEBUG >= 2:
                print(f"   🔧 Preserving cache context, curr_pos set to {self.state.curr_pos}")
            
            if DEBUG >= 2:
              state_status = "CLEARED" if self.state.input_pos is None else "PRESERVED"
              print(f"   ✅ State {state_status}, cache reset, ready for processing")
      else:
        self.state.tokens = input_tensor.clone()

      try:
        in_tokens = self.state.tokens.clone().to(
          device=self.device
        )

        # Handle case where infer_tensor is called without prior encode() (e.g., in speculative decoding)
        # OR when a longer sequence is passed after initial setup (speculative verification)
        current_seq_len = in_tokens.size(1)
        
        if DEBUG >= 1:
          print(f"🔍 DEBUG: infer_tensor called with sequence length: {current_seq_len}")
          if self.state.input_pos is not None:
            print(f"🔍 DEBUG: Current input_pos size: {self.state.input_pos.size()}")
          else:
            print(f"🔍 DEBUG: input_pos is None")
          if self.state.mask is not None:
            print(f"🔍 DEBUG: Current mask size: {self.state.mask.size()}")
          else:
            print(f"🔍 DEBUG: mask is None")
          if hasattr(self.sharded_model.model, 'caches_are_enabled') and self.sharded_model.model.caches_are_enabled():
            print(f"🔍 DEBUG: Cache is enabled")
          print(f"🔍 DEBUG: Cache setup flag: {self.cache_setup}")
        
        # Special case: Cache is set up but state is not initialized (common in speculative decoding)
        # CRITICAL FIX: Initialize state to be compatible with existing cache (preserve context)
        if (self.cache_setup and model_cache and 
            (self.state.input_pos is None or self.state.mask is None)):
          if DEBUG >= 1:
            print(f"🔄 PyTorch engine: Cache setup but no state - creating compatible state for speculative decoding")
          
          # Get cache dimensions to ensure compatibility
          cache_seq_len = getattr(self.sharded_model.model, 'decoder_max_cache_seq_len', current_seq_len)
          
          if DEBUG >= 1:
            print(f"🔧 Cache size: {cache_seq_len}, Input size: {current_seq_len}")
          
          # Create state that's compatible with cache dimensions
          # This preserves the large cache while making input compatible
        
        if self.state.input_pos is None or self.state.mask is None:
          # No prior state - create fresh state
          pass
        elif self.state.input_pos.size(-1) != current_seq_len:
          # Sequence length mismatch - adapt for speculative verification
          prev_seq_len = self.state.input_pos.size(-1)
          
          if DEBUG >= 1:
            print(f"🔄 PyTorch engine: Sequence length mismatch ({prev_seq_len} → {current_seq_len}) - adapting for speculative verification")
          
          # SMART CACHE HANDLING: Only reset if absolutely necessary
          if model_cache and self.sharded_model.model.caches_are_enabled():
            cache_seq_len = getattr(self.sharded_model.model, 'decoder_max_cache_seq_len', current_seq_len)
            
            if current_seq_len <= cache_seq_len and abs(current_seq_len - prev_seq_len) <= 5:
              # Small extension that fits in cache - preserve context
              if DEBUG >= 1:
                print(f"   ✅ Cache preservation: {current_seq_len} <= {cache_seq_len}, small extension ({abs(current_seq_len - prev_seq_len)} tokens)")
              
              # Update state without full reset
              self.state.input_pos = None   # Will regenerate
              self.state.mask = None        # Will regenerate  
              # Keep curr_pos to maintain context flow
            else:
              # Large extension or cache overflow - need reset
              if DEBUG >= 1:
                print(f"   🔄 Cache reset needed: seq_len={current_seq_len}, cache_len={cache_seq_len}, extension={abs(current_seq_len - prev_seq_len)}")
              
              # Full reset for major changes
              self.state.input_pos = None
              self.state.mask = None
              self.state.curr_pos = 0
              self.sharded_model.model.reset_caches()
              print(f"   🔄 Cache reset for sequence length change")
          else:
            # Non-cached model - just update state
            self.state.input_pos = None
            self.state.mask = None
            self.state.curr_pos = 0
        
        if self.state.input_pos is None or self.state.mask is None:
          if DEBUG >= 2:
            print("PyTorch engine: No encode() detected - creating fallback initialization for speculative decoding")
            print(f"Current tokens shape: {in_tokens.shape}")
          
          # Create fallback initialization for speculative decoding
          # This happens when the draft engine is called without proper encode() setup
          bsz, seq_len = in_tokens.shape
          
          # Setup cache if needed
          if self.use_cache and not self.cache_setup:
            # Add generous buffer for speculative decoding
            max_speculative_iterations = 20
            gamma = 10
            speculative_buffer = max_speculative_iterations * gamma
            self.setup_cache(bsz, seq_len + self.sharded_model.max_generated_tokens + speculative_buffer)
          
          # Create basic input_pos and mask tensors
          total_response_length = seq_len + self.sharded_model.max_generated_tokens
          
          # Setup max sequence length
          if not self.sharded_model.model.caches_are_enabled():
            max_seq_len = total_response_length
          else:
            max_seq_len = self.sharded_model.model.decoder_max_cache_seq_len
                    
          # Create input_pos for current sequence - ALWAYS dynamically generate
          # This is crucial for torchtune models to avoid tensor shape mismatches
          self.state.input_pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
          
          # Create causal mask matching the cache dimensions for compatibility
          if model_cache and self.cache_setup:
            # For cached models, create mask that matches cache size but only attend to relevant positions
            cache_seq_len = getattr(self.sharded_model.model, 'decoder_max_cache_seq_len', seq_len)
            
            # Create mask sized for cache but only allow attention within actual sequence
            self.state.mask = torch.tril(torch.ones(
              1, cache_seq_len, cache_seq_len,
              dtype=torch.bool,
              device=self.device,
            ))
            
            # Mask out positions beyond current sequence
            if cache_seq_len > seq_len:
              self.state.mask[:, seq_len:, :] = False
              self.state.mask[:, :, seq_len:] = False
              
            if DEBUG >= 1:
              print(f"🔧 Created cache-compatible mask: {self.state.mask.shape} (seq_len={seq_len}, cache_len={cache_seq_len})")
          elif model_cache:
            # For cached models without explicit cache setup, use current sequence
            self.state.mask = torch.tril(torch.ones(
              1, seq_len, seq_len,
              dtype=torch.bool,
              device=self.device,
            ))
          else:
            # For non-cached models, create mask for full response length
            self.state.mask = torch.tril(torch.ones(
              1, total_response_length, total_response_length,
              dtype=torch.bool,
              device=self.device,
            ))
          
          # Initialize current position
          self.state.curr_pos = 0
          
          if DEBUG >= 2:
            print(f"   ✅ Created fallback state: input_pos={self.state.input_pos.shape}, mask={self.state.mask.shape}")
        
        # CRITICAL FIX: Always dynamically generate input_pos based on current sequence length
        # This prevents tensor shape mismatches in torchtune models during speculative decoding
        current_seq_len = in_tokens.size(1)
        
        # For cached models with speculative decoding, we need to handle position correctly
        if model_cache and self.cache_setup:
          # Use incremental positions for cached generation
          # This assumes cache contains context and we're adding new tokens
          cache_seq_len = getattr(self.sharded_model.model, 'decoder_max_cache_seq_len', current_seq_len)
          
          if current_seq_len <= cache_seq_len:
            # Normal case: sequence fits in cache
            in_input_pos = torch.arange(0, current_seq_len, device=self.device).unsqueeze(0)
          else:
            # Edge case: sequence longer than cache - use last positions
            start_pos = max(0, current_seq_len - cache_seq_len)
            in_input_pos = torch.arange(start_pos, current_seq_len, device=self.device).unsqueeze(0)
            
          if DEBUG >= 1:
            print(f"🔧 Cached model input_pos: {in_input_pos.shape} (cache_len={cache_seq_len}, seq_len={current_seq_len})")
        else:
          # Non-cached model: use standard positioning
          in_input_pos = torch.arange(0, current_seq_len, device=self.device).unsqueeze(0)
        
        # Ensure mask is compatible with cache and current sequence
        if model_cache and self.cache_setup:
          cache_seq_len = getattr(self.sharded_model.model, 'decoder_max_cache_seq_len', current_seq_len)
          
          if self.state.mask.size(-1) != cache_seq_len:
            # Create cache-compatible mask
            in_mask = torch.tril(torch.ones(
              1, cache_seq_len, cache_seq_len,
              dtype=torch.bool,
              device=self.device,
            ))
            
            # Mask out positions beyond current sequence
            if cache_seq_len > current_seq_len:
              in_mask[:, current_seq_len:, :] = False
              in_mask[:, :, current_seq_len:] = False
              
            if DEBUG >= 1:
              print(f"🔧 Generated cache-compatible mask: {in_mask.shape} (seq_len={current_seq_len}, cache_len={cache_seq_len})")
          else:
            # Use existing mask but update attention boundaries
            in_mask = self.state.mask.clone().to(device=self.device)
            if cache_seq_len > current_seq_len:
              in_mask[:, current_seq_len:, :] = False
              in_mask[:, :, current_seq_len:] = False
        else:
          # Non-cached model: use simple sequence-length mask
          if self.state.mask.size(-1) != current_seq_len:
            in_mask = torch.tril(torch.ones(
              1, current_seq_len, current_seq_len,
              dtype=torch.bool,
              device=self.device,
            ))
            if DEBUG >= 1:
              print(f"🔧 Regenerated mask for seq_len {current_seq_len}: {in_mask.shape}")
          else:
            in_mask = self.state.mask.clone().to(device=self.device)
        
        if DEBUG >= 1:
          print(f"🔧 Dynamic input_pos: {in_input_pos.shape} for seq_len {current_seq_len}")

        # CRITICAL FIX: For cached models, properly track position for context preservation
        if model_cache and self.cache_setup:
          # For speculative decoding, track position based on actual progress
          # If this is the first call for this sequence length, start from beginning
          # Otherwise, use incremental position for proper caching
          if hasattr(self, '_last_seq_len') and self._last_seq_len == current_seq_len:
            # Same sequence length - use incremental position
            actual_curr_pos = current_seq_len - 1
          else:
            # New or extended sequence - start fresh but preserve cache
            actual_curr_pos = 0
            self._last_seq_len = current_seq_len
            
          if DEBUG >= 1:
            print(f"🔧 Using curr_pos={actual_curr_pos} for cached model (seq_len={current_seq_len}, last_len={getattr(self, '_last_seq_len', 'None')})")
        else:
          actual_curr_pos = self.state.curr_pos

        if hidden_state is not None:
          model_hs, model_logits = self.sharded_model.generate(
            tokens=in_tokens,
            hidden_state=hidden_state,
            input_pos=in_input_pos,
            mask=in_mask,
            curr_pos=actual_curr_pos
          )
        else:
          if not model_cache:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=in_tokens,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=actual_curr_pos
            )
          else:
            # CRITICAL FIX: Use in_tokens (extended sequence) not input_tensor for cached models
            # This ensures speculative decoding target verification receives the full extended sequence
            model_hs, model_logits = self.sharded_model.generate(
              tokens=in_tokens,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=actual_curr_pos
            )
      except torch.cuda.OutOfMemoryError:
        print(f"OOM on cuda, clearing model and stopping")
        self.oom_cnt += 1
        self.clear_model()
        return
      except Exception as err:
        print(f"infer_tensor err\n{err}")
        raise

      if model_hs is not None:
        # numpy current no support for bf16
        if model_hs.dtype == torch.bfloat16:
          model_hs = model_hs.float()

        if DEBUG >= 4:
          print("sending hidden states")
          print(f"model_hs: {model_hs.size()}")
          print(f"state.tokens: {self.state.tokens}")
          print(f"state.input_pos: {self.state.input_pos.size()}")
          print(f"state.mask: {self.state.mask.size()}")
        
        return (
          model_hs.numpy(force=True),
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      # numpy current no support for bf16
      if model_logits.dtype == torch.bfloat16:
        model_logits = model_logits.float()

      # CRITICAL FIX FOR SPECULATIVE DECODING:
      # For speculative decoding verification, we need to return logits for all new positions,
      # not just the last position. Check if this is an extended sequence for verification.
      if hasattr(self, '_original_seq_len') and current_seq_len > self._original_seq_len:
        # This is speculative verification - return logits for new positions only
        num_new_positions = current_seq_len - self._original_seq_len
        if DEBUG >= 1:
          print(f"🔍 SPECULATIVE VERIFICATION: Returning {num_new_positions} new positions (seq: {self._original_seq_len} -> {current_seq_len})")
        
        if model_logits.shape[1] >= num_new_positions:
          # Return logits for new positions
          new_position_logits = model_logits[:, -num_new_positions:]
          if DEBUG >= 1:
            print(f"🔍 Returning logits shape: {new_position_logits.shape} for speculative verification")
          return (
            new_position_logits.numpy(force=True),
            self.state.to_dict(),
          )
        else:
          if DEBUG >= 1:
            print(f"🚨 WARNING: Not enough logits positions ({model_logits.shape[1]}) for {num_new_positions} new tokens")
      
      # Standard case: return only last position
      return (
        model_logits[:, -1].numpy(force=True),
        self.state.to_dict(),
      )

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    # reset model after last layer to fix OOM
    if self.shard == shard:
      return

    self.shard = shard

    # Using CPU to store inference state
    self.state = ShardInferenceState()

    # download model safetensors and shard

    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(self.model_path)

    def start_model():
      if DEBUG >= 4:
        print("start_model called")

      self.sharded_model = ShardedGeneralModel(
        config=self.model_config,
        shard=shard,
        device=self.device,
        dtype=self.model_config["torch_dtype"],
        use_cache=self.use_cache
      )

      load_model_weights_torchtune(
        cache_dir=self.model_path,
        shard=self.shard,
        model=self.sharded_model,
        num_heads=self.model_config["num_heads"],
        num_kv_heads=self.model_config["num_kv_heads"],
        dim=self.model_config["embed_dim"],
        head_dim=self.model_config["head_dim"]
      )
    
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(start_model),
    )

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
