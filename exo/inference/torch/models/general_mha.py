"""
GeneralMHA class
Return transformer model with MHA
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchtune.modules as ttm

from torchtune.modules import RMSNorm
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings
from torchtune.modules import RotaryPositionalEmbeddings
from exo.inference.shard import Shard
from exo.inference.torch.models.llm_utils import (
  layer_mlp,
  ShardTransformerDecoder
)

from exo.helpers import DEBUG

def GeneralMHA(
    config: dict,
    shard: Shard
):
  use_tied = False
  attn_bias = config.get("attn_bias", False)
  output_bias = config.get("attn_bias", False)

  if "llama" in shard.model_id or "Llama" in shard.model_id:
    # rope scaling config
    rope = Llama3ScaledRoPE(
      dim=config["head_dim"],
      max_seq_len=config["max_seq_len"],
      base=config["rope_base"],
      scale_factor=config["rope_scaling_factor"],
    )

    # tied needed for 3.2 llama models
    if "3.2" in shard.model_id:
      use_tied = True
  elif "qwen" in shard.model_id or "Qwen" in shard.model_id:
    # rope scaling config
    rope = Qwen2RotaryPositionalEmbeddings(
      dim=config["head_dim"],
      max_seq_len=config["max_seq_len"],
      base=config["rope_base"]
    )
    attn_bias = True
    output_bias = False

    # tied needed for 0.5B qwen models
    if "0.5B" in shard.model_id or "0.5b" in shard.model_id:
      use_tied = True
  else:
    rope = RotaryPositionalEmbeddings(
      dim=config["head_dim"],
      max_seq_len=config["max_seq_len"],
      base=config["rope_base"]
    )
  
  if DEBUG >= 4:
    print(f"model_id: {shard.model_id}")
    print(f"rope: {rope}")
    print(f"attn_bias: {attn_bias}")
    print(f"output_bias: {output_bias}")
    print(f"use_tied: {use_tied}")

  # hack to align sharded weights with layers
  # fill unused layer positions with None
  layers = [None for _ in range(shard.n_layers)]

  # build layers
  for i in range(shard.start_layer, shard.end_layer + 1):
    self_attn = ttm.MultiHeadAttention(
      embed_dim=config["embed_dim"],
      num_heads=config["num_heads"],
      num_kv_heads=config["num_kv_heads"],
      head_dim=config["head_dim"],
      q_proj=nn.Linear(
        config["embed_dim"],
        config["num_heads"]*config["head_dim"],
        bias=attn_bias,
      ),
      k_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"]*config["head_dim"],
        bias=attn_bias,
      ),
      v_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"]*config["head_dim"],
        bias=attn_bias,
      ),
      output_proj=nn.Linear(
        config["embed_dim"],
        config["embed_dim"],
        bias=output_bias,
      ),
      max_seq_len=config["max_seq_len"],
      attn_dropout=config["attn_dropout"],
      pos_embeddings=rope,
    )

    mlp = layer_mlp(
      dim=config["embed_dim"],
      hidden_dim=config["intermediate_dim"],
    )

    layer = ttm.TransformerSelfAttentionLayer(
      attn=self_attn,
      mlp=mlp,
      sa_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
      mlp_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    )

    layers[i] = layer

  layers = nn.ModuleList(layers)

  tok_embeddings = nn.Embedding(config["vocab_size"], config["embed_dim"])
  if use_tied:
    output_proj = ttm.TiedLinear(tok_embeddings)
  else:
    output_proj = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

  norm = RMSNorm(config["embed_dim"], eps=config["norm_eps"])

  return ShardTransformerDecoder(
    tok_embeddings=tok_embeddings,
    shard=shard,
    layers=layers,
    max_seq_len=config["max_seq_len"],
    num_heads=config["num_heads"],
    head_dim=config["head_dim"],
    norm=norm,
    output=output_proj,
    num_layers=config["num_layers"],
  )

class ShardedGeneralModel(nn.Module):
  def __init__(
    self,
    config: dict,
    shard: Shard,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    use_cache: Optional[bool] = False,
    max_generated_tokens: int = 1024,
  ):
    super(ShardedGeneralModel, self).__init__()

    self.shard = shard
    self.config = config
    self.dtype = dtype
    self.device = device if device is not None else torch.device("cpu")
    self.max_seq_len = self.config["max_seq_len"]
    self.use_cache = use_cache
    
    self.model = GeneralMHA(
      config,
      self.shard
    ).to(
      dtype=self.dtype,
      device=self.device
    )

    if DEBUG >= 4:
      print("ShardedGeneralModel called")
      print(f"self.model {self.model}")

    # keep track of current position in generation
    self.max_generated_tokens = max_generated_tokens

  def generate(
    self,
    tokens: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    hidden_state: Optional[torch.Tensor] = None,
    curr_pos: Optional[int] = 0
  ) -> Tuple[
    Optional[torch.Tensor],
    torch.Tensor,
  ]:
    """
    Generate logits and/or hidden_states from llama model

    Args
      tokens (torch.Tensor) - tokens from prompt tokenization and generation
      hidden_state (torch.Tensor, optional) - hidden state from last activated hidden layer, if any
    """
    if DEBUG >= 4:
      print("generate called")
      print(f"tokens: {tokens}")
      if mask is not None:
        print(f"mask: {mask.size()}")
        print(f"input_pos: {input_pos.size()}") 
      print(f"hidden_state: {hidden_state}")
      print(f"curr_pos: {curr_pos}")
      print(f"cached? {self.model.caches_are_enabled()}")

    model_hs = None
    model_logits = None

    self.model.output_hidden_states = [self.shard.end_layer]

    if curr_pos > 0:
      if self.model.caches_are_enabled():
        # Fix: Handle case where curr_pos might be at or beyond input_pos length
        if curr_pos < input_pos.size(1):
          input_pos = input_pos[:, curr_pos].contiguous()
          mask = mask[:, curr_pos, None, :].contiguous()
        else:
          # If curr_pos is beyond current input_pos, use the last position
          input_pos = input_pos[:, -1].contiguous()
          mask = mask[:, -1, None, :].contiguous()
      else:
        input_pos = input_pos[:, :curr_pos + 1]
        mask = mask[:, :curr_pos + 1, :curr_pos + 1]
    else:
      _, tklng = tokens.size()

      if self.model.caches_are_enabled():
        mask = mask[:, :tklng]
      else:
        mask = mask[:, :tklng, :tklng]

      input_pos = input_pos[:, :tklng]
      # Only squeeze if it's safe to do so (batch dimension should remain)
      if input_pos.dim() > 1 and input_pos.shape[0] == 1 and input_pos.shape[1] > 1:
        input_pos = input_pos.squeeze(0)

    if DEBUG >= 4:
      print("model_input")
      if tokens is not None:
        print(f"tokens: {tokens}")
      if hidden_state is not None:
        print(f"hidden_state: {hidden_state}")
      print(f"mask: {mask}")
      print(f"input_pos: {input_pos}")
      
    # CRITICAL FIX: Ensure input_pos matches the actual sequence length
    # In speculative decoding, we may pass a full sequence but input_pos might not match
    if tokens is not None:
      actual_seq_len = tokens.size(1)
      if input_pos is not None:
        # If input_pos doesn't match the actual sequence length, recreate it
        if input_pos.dim() == 1 and input_pos.numel() != actual_seq_len:
          if DEBUG >= 2:
            print(f"⚠️  input_pos length ({input_pos.numel()}) doesn't match sequence length ({actual_seq_len})")
            print(f"   Recreating input_pos to match sequence...")
          input_pos = torch.arange(actual_seq_len, device=tokens.device, dtype=input_pos.dtype)
        elif input_pos.dim() == 2 and input_pos.size(1) != actual_seq_len:
          if DEBUG >= 2:
            print(f"⚠️  input_pos shape {input_pos.shape} doesn't match sequence length ({actual_seq_len})")
            print(f"   Recreating input_pos to match sequence...")
          input_pos = torch.arange(actual_seq_len, device=tokens.device, dtype=input_pos.dtype).unsqueeze(0)
      else:
        # If input_pos is None, create it based on actual sequence length
        if DEBUG >= 2:
          print(f"🔧 Creating input_pos for sequence length {actual_seq_len}")
        input_pos = torch.arange(actual_seq_len, device=tokens.device)

    if DEBUG >= 4:
      print("model_input (after fix)")
      if tokens is not None:
        print(f"tokens: {tokens}")
      if hidden_state is not None:
        print(f"hidden_state: {hidden_state}")
      print(f"mask: {mask}")
      print(f"input_pos: {input_pos}")

    # Debug cache state for troubleshooting
    if DEBUG >= 3 and self.model.caches_are_enabled() and input_pos is not None:
      for i, layer in enumerate(self.model.layers[:1]):  # Just check first layer
        if hasattr(layer.attn, 'kv_cache') and layer.attn.kv_cache is not None:
          cache = layer.attn.kv_cache
          if hasattr(cache, 'cache_pos') and len(cache.cache_pos) > 0:
            current_pos = cache.cache_pos[0].item()
            max_cache_size = cache.k_cache.shape[2]
            seq_len = tokens.size(1) if tokens is not None else 1
            print(f"   🔧 Cache debug: layer {i}, pos={current_pos}, seq_len={seq_len}, max={max_cache_size}")
            
            if current_pos + seq_len > max_cache_size:
              print(f"⚠️  WOULD OVERFLOW: {current_pos} + {seq_len} = {current_pos + seq_len} > {max_cache_size}")
            break

    with torch.no_grad():
      model_output = self.model(
        tokens=tokens,
        mask=mask,
        input_pos=input_pos,
        hidden_state=hidden_state,
        dtype=self.dtype
      )

    if self.shard.is_last_layer():
      model_logits = model_output
    else:
      model_hs = model_output

    if DEBUG >= 4:
      print(f"model_hs\n{model_hs}\nmodel_logits\n{model_logits}")

    return model_hs, model_logits
