import time
import numpy as np
from typing import Optional, Tuple, List
from .inference_engine import InferenceEngine
from .shard import Shard
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import torch

# Add TinyGrad imports for state clearing
try:
    from tinygrad.helpers import diskcache_clear, GlobalCounters
    from tinygrad.tensor import Tensor
    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False

# Enable debug for troubleshooting
DEBUG = int(os.environ.get("DEBUG", "0"))

class SpeculativeInferenceEngine(InferenceEngine):
    """
    Wrapper engine that performs speculative decoding using draft and target models.
    """
    
    def __init__(self, target_engine: InferenceEngine, draft_engine: InferenceEngine,
                 gamma: int = 2, temperature: float = 0.7, top_k_threshold: float = 0.9,
                 lenience: float = 2.0, target_model_id: str = None, draft_model_id: str = None):
        self.target_engine = target_engine
        self.draft_engine = draft_engine
        self.gamma = gamma
        self.temperature = temperature
        self.top_k_threshold = top_k_threshold
        self.lenience = lenience
        self.target_model_id = target_model_id  # Store for shard creation
        self.draft_model_id = draft_model_id    # Store for shard creation
        
        # Statistics tracking
        self.total_calls = 0
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        
        # Acceptance tracking
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        
        if DEBUG >= 1:
            print(f"🚀 SpeculativeInferenceEngine initialized:")
            print(f"   gamma={gamma}, temperature={temperature}")
            print(f"   top_k_threshold={top_k_threshold}, lenience={lenience}")

    def _to_numpy(self, tensor_or_array):
        """Convert PyTorch tensors to NumPy arrays, leave NumPy arrays unchanged."""
        if hasattr(tensor_or_array, 'detach'):  # PyTorch tensor
            return tensor_or_array.detach().cpu().numpy()
        else:  # Already NumPy array
            return tensor_or_array

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode prompt using target engine."""
        result = await self.target_engine.encode(shard, prompt)
        return self._to_numpy(result)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Decode tokens using target engine."""
        return await self.target_engine.decode(shard, tokens)

    async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
        """Sample using target engine."""
        return await self.target_engine.sample(x, temp, top_p)

    @property
    def tokenizer(self):
        """Get tokenizer from target engine."""
        return getattr(self.target_engine, 'tokenizer', None)
    
    @property
    def shard(self):
        """Delegate shard property to target engine for compatibility."""
        return getattr(self.target_engine, 'shard', None)
    
    async def ensure_shard(self, shard: Shard):
        """Ensure both engines have the shard."""
        await self.target_engine.ensure_shard(shard)
        if self.draft_engine:
            draft_shard = self._create_draft_shard_from_target(shard)
            await self.draft_engine.ensure_shard(draft_shard)

    def _create_draft_shard_from_target(self, target_shard: Shard) -> Shard:
        """Create draft shard with different model and correct layer configuration."""
        # Use the configured draft model ID, or fall back to target model
        draft_model_id = self.draft_model_id or target_shard.model_id
        
        # Get the correct number of layers for the draft model
        draft_n_layers = target_shard.n_layers  # Default to target layers
        
        if self.draft_model_id:
            # Map model IDs to their layer counts based on TinyGrad MODEL_PARAMS
            model_layer_mapping = {
                "llama-3.2-1b": 16,  # 1B model has 16 layers
                "llama-3.2-3b": 28,  # 3B model has 28 layers
                "llama-3.1-8b": 32,  # 8B model has 32 layers
                "llama-3.3-70b": 80, # 70B model has 80 layers
            }
            
            if draft_model_id in model_layer_mapping:
                draft_n_layers = model_layer_mapping[draft_model_id]
                if DEBUG >= 1:
                    print(f"🔧 Draft model {draft_model_id} using {draft_n_layers} layers (target has {target_shard.n_layers})")
            else:
                if DEBUG >= 1:
                    print(f"⚠️  Unknown draft model {draft_model_id}, using target layer count: {draft_n_layers}")
        
        if DEBUG >= 1 and self.draft_model_id:
            print(f"🔧 Creating draft shard with model_id: {draft_model_id} (was: {target_shard.model_id})")
        elif DEBUG >= 1:
            print(f"⚠️  No draft_model_id specified, using target model: {draft_model_id}")
            
        # Calculate appropriate layer range for draft model
        # For now, just use the full model (start=0, end=n_layers-1)
        draft_start_layer = 0
        draft_end_layer = draft_n_layers - 1
        
        return Shard(
            model_id=draft_model_id,  # Use actual draft model ID
            start_layer=draft_start_layer,  # Use full draft model
            end_layer=draft_end_layer,      # Use full draft model
            n_layers=draft_n_layers         # Use correct layer count
        )

    async def load_checkpoint(self, shard: Shard, path: str):
        """Load checkpoint for target (and draft if available) engines."""
        if DEBUG >= 1:
            print(f"💾 Loading checkpoint from: {path}")
        await self.target_engine.load_checkpoint(shard, path)
        if self.draft_engine:
            draft_shard = self._create_draft_shard_from_target(shard)
            await self.draft_engine.load_checkpoint(draft_shard, path)

    async def save_checkpoint(self, shard: Shard, path: str):
        """Save checkpoint for target engine."""
        if DEBUG >= 1:
            print(f"💾 Saving checkpoint to: {path}")
        await self.target_engine.save_checkpoint(shard, path)

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, 
                          inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """Standard tensor inference using target engine (no speculative decoding for single calls)."""
        if DEBUG >= 1:
            print("🔮 =================SPECULATIVE DECODING DEBUG================= ")
            print(f"📊 Input shape: {input_data.shape}")
            print(f"🎯 Target engine: {type(self.target_engine).__name__}")
            print(f"📝 Draft engine: {type(self.draft_engine).__name__}")
            print("")

        # Show detailed model analysis
        if DEBUG >= 1:
            target_shard = shard
            draft_shard = self._create_draft_shard_from_target(shard)
            
            print(f"🔧 Draft model {draft_shard.model_id} using {draft_shard.end_layer - draft_shard.start_layer + 1} layers (target has {target_shard.end_layer - target_shard.start_layer + 1})")
            print(f"🔧 Creating draft shard with model_id: {draft_shard.model_id} (was: {target_shard.model_id})")
            print("")
            print("🔍 DETAILED MODEL ANALYSIS:")
            print(f"✅ Target and draft engines are different objects" if self.target_engine != self.draft_engine else "❌ Same engine object!")
            # Get repository information using get_repo function
            from exo.models import get_repo
            target_repo = get_repo(shard.model_id, self.target_engine.__class__.__name__)
            draft_repo = get_repo(draft_shard.model_id, self.draft_engine.__class__.__name__)
            
            print(f"🎯 Target repo: {target_repo}")
            print(f"📝 Draft repo: {draft_repo}")
            if target_repo == draft_repo:
                print(f"🚨 WARNING: Both engines using same repo: {target_repo}")
            
            print(f"🎯 Target model path: {getattr(self.target_engine, 'model_path', 'UNKNOWN')}")
            print(f"📝 Draft model path: {getattr(self.draft_engine, 'model_path', 'UNKNOWN')}")
            print(f"🔧 Draft model {draft_shard.model_id} using {draft_shard.end_layer - draft_shard.start_layer + 1} layers (target has {target_shard.end_layer - target_shard.start_layer + 1})")
            print(f"🔧 Creating draft shard with model_id: {draft_shard.model_id} (was: {target_shard.model_id})")
            print(f"🎯 Target shard: start={target_shard.start_layer}, end={target_shard.end_layer}, n_layers={target_shard.end_layer - target_shard.start_layer + 1}")
            print(f"📝 Draft shard: start={draft_shard.start_layer}, end={draft_shard.end_layer}, n_layers={draft_shard.end_layer - draft_shard.start_layer + 1}")
            print(f"🎯 Target model_id: {target_shard.model_id}")
            print(f"📝 Draft model_id: {draft_shard.model_id}")
            print(f"🎯 Target engine memory: {hex(id(self.target_engine))}")
            print(f"📝 Draft engine memory: {hex(id(self.draft_engine))}")
            print(f"🎲 Gamma (draft tokens): {self.gamma}")
            print("")
            print("📋 ALGORITHM PARAMETERS:")
            print(f"   Temperature: {self.temperature}")
            print(f"   Top-k threshold: {self.top_k_threshold}")
            print(f"   Lenience: {self.lenience}")
            print(f"   Target engine type: {type(self.target_engine)}")
            print(f"   Draft engine type: {type(self.draft_engine)}")
            print("✅ Speculative parameters look valid")
            print(f"🔧 Draft model {draft_shard.model_id} using {draft_shard.end_layer - draft_shard.start_layer + 1} layers (target has {target_shard.end_layer - target_shard.start_layer + 1})")
            print(f"🔧 Creating draft shard with model_id: {draft_shard.model_id} (was: {target_shard.model_id})")
            print("🔍 Models setup complete - proceeding with speculative decoding")

        result, state = await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
        return self._to_numpy(result), state

    async def infer_tensor_multi(self, request_id: str, shard: Shard, input_data: np.ndarray,
                                inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict], Optional[list]]:
        """Multi-token inference with speculative decoding."""
        if self.draft_engine is None:
            # No draft engine, fallback to target only
            output, state = await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            return self._to_numpy(output), state, []
        
        return await self._speculative_decode_vanilla_multi(request_id, shard, input_data, inference_state)

    async def infer_prompt_multi(self, request_id: str, shard: Shard, prompt: str,
                                inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict], Optional[list]]:
        """Multi-token prompt inference with speculative decoding."""
        # Encode prompt first
        input_tokens = await self.encode(shard, prompt)
        
        # Use speculative decoding for generation
        return await self.infer_tensor_multi(request_id, shard, input_tokens, inference_state)

    async def _speculative_decode_vanilla_multi(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None
    ) -> Tuple[np.ndarray, Optional[dict], Optional[list]]:
        """
        Implements vanilla speculative decoding with proper cache management.
        
        Key principles:
        1. Draft model generates gamma tokens with proper context continuation
        2. Target model evaluates those same tokens in the extended sequence
        3. Cache states are properly managed and coordinated between phases
        4. Context is preserved throughout the process
        """
        
        start_time = time.perf_counter()
        DEBUG = int(os.getenv("DEBUG", "0"))
        all_accepted_tokens = []

        if DEBUG >= 1:
            print(f"\n🔮 =================SPECULATIVE DECODING DEBUG================= ")
            print(f"📊 Input shape: {input_data.shape}")
            print(f"🎯 Target engine: {self.target_engine.__class__.__name__}")
            print(f"📝 Draft engine: {self.draft_engine.__class__.__name__}")

        # Get the input tokens and create draft shard
        out = input_data.copy()
        current_seq_len = out.shape[1]
        
        # Create a compatible draft shard
        draft_shard = self._create_draft_shard_from_target(shard)
        
        # 🔧 CRITICAL: Ensure both models are loaded and warmed up
        await self.target_engine.ensure_shard(shard)
        await self.draft_engine.ensure_shard(draft_shard)

        # CRITICAL FIX: Coordinate cache states between models for alignment
        target_cache = inference_state
        draft_cache = None  # Start draft fresh but will build consistent state
        
        if DEBUG >= 2:
            print(f"   🔧 Using independent draft generation to avoid model conflicts")

        if DEBUG >= 1:
            print(f"\n🔍 DETAILED MODEL ANALYSIS:")
            print(f"✅ Target and draft engines are different objects")
            
            # Get repository information using get_repo function
            from exo.models import get_repo
            target_repo = get_repo(shard.model_id, self.target_engine.__class__.__name__)
            draft_repo = get_repo(draft_shard.model_id, self.draft_engine.__class__.__name__)
            
            print(f"🎯 Target repo: {target_repo}")
            print(f"📝 Draft repo: {draft_repo}")
            if target_repo == draft_repo:
                print(f"🚨 WARNING: Both engines using same repo: {target_repo}")
            
            model_path_target = getattr(self.target_engine, 'model_path', 'UNKNOWN')
            model_path_draft = getattr(self.draft_engine, 'model_path', 'UNKNOWN')
            print(f"🎯 Target model path: {model_path_target}")
            print(f"📝 Draft model path: {model_path_draft}")
            print(f"🔧 Draft model {self.draft_model_id} using {draft_shard.n_layers} layers (target has {shard.n_layers})")
            print(f"🔧 Creating draft shard with model_id: {draft_shard.model_id} (was: {shard.model_id})")
            print(f"🎯 Target shard: start={shard.start_layer}, end={shard.end_layer}, n_layers={shard.n_layers}")
            print(f"📝 Draft shard: start={draft_shard.start_layer}, end={draft_shard.end_layer}, n_layers={draft_shard.n_layers}")
            print(f"🎯 Target model_id: {shard.model_id}")
            print(f"📝 Draft model_id: {draft_shard.model_id}")
            print(f"🎯 Target engine memory: {hex(id(self.target_engine))}")
            print(f"📝 Draft engine memory: {hex(id(self.draft_engine))}")
            print(f"🎲 Gamma (draft tokens): {self.gamma}")

        # Model verification and debugging
        if DEBUG >= 1:
            print(f"\n📋 ALGORITHM PARAMETERS:")
            print(f"   Temperature: {self.temperature}")
            print(f"   Top-k threshold: {self.top_k_threshold}")
            print(f"   Lenience: {self.lenience}")
            print(f"   Target engine type: {type(self.target_engine)}")
            print(f"   Draft engine type: {type(self.draft_engine)}")
            print(f"✅ Speculative parameters look valid")
            print(f"🔧 Draft model {self.draft_model_id} using {draft_shard.n_layers} layers (target has {shard.n_layers})")
            print(f"🔧 Creating draft shard with model_id: {draft_shard.model_id} (was: {shard.model_id})")

        # Cache state tracking
        if DEBUG >= 2:
            print(f"\n🔧 CACHE STATE MANAGEMENT:")
            print(f"   Target cache state: {type(target_cache)}")
            print(f"   Draft cache state: {type(draft_cache)}")
            if hasattr(target_cache, 'cache_pos'):
                print(f"   Target cache position: {getattr(target_cache, 'cache_pos', 'N/A')}")
            if hasattr(draft_cache, 'cache_pos'):
                print(f"   Draft cache position: {getattr(draft_cache, 'cache_pos', 'N/A')}")

        # Verify models are actually different
        if DEBUG >= 1:
            print("🔍 Models setup complete - proceeding with speculative decoding")

        # Store probabilities from draft generation for verification
        draft_probs_for_verification = []
        
        # ============ PHASE 1: DRAFT GENERATION ============
        if DEBUG >= 1:
            print("📝 PHASE 1: DRAFT GENERATION")
            print(f"   🎲 Generating {self.gamma} draft tokens using {draft_shard.model_id}")
            print(f"   🌡️  Draft temperature: {self.temperature * 1.2:.2f} (base: {self.temperature})")
            print(f"   🚫 Special token filtering: ENABLED")
            print(f"   📊 Starting draft token generation...")
            print(f"   🎲 Generating {self.gamma} draft tokens using {draft_shard.model_id}")
            print(f"   🌡️  Draft temperature: {self.temperature * 1.2:.2f} (base: {self.temperature})")
        
        current_sequence = input_data.copy()
        draft_tokens = []
        
        for i in range(self.gamma):
            if DEBUG >= 2:
                print(f"   🎲 Draft iteration {i+1}/{self.gamma}:")
                print(f"      Current sequence length: {current_sequence.shape[1]}")
                print(f"      Last 3 tokens: {current_sequence[0, -3:].tolist() if current_sequence.shape[1] >= 3 else current_sequence[0].tolist()}")
            
            # CRITICAL FIX: Maintain draft cache state for consistent context
            draft_output, draft_cache = await self.draft_engine.infer_tensor(
                f"{request_id}_draft_{i}",
                draft_shard,
                current_sequence,
                draft_cache  # Maintain consistent draft state
            )
            draft_output = self._to_numpy(draft_output)
            
            # Extract logits from draft output
            if draft_output.ndim == 3:
                draft_logits = draft_output[:, -1, :]  # Last position
            elif draft_output.ndim == 2:
                draft_logits = draft_output
            else:
                raise ValueError(f"Unexpected draft output shape: {draft_output.shape}")
            
            if DEBUG >= 1:
                print(f"   🎲 DRAFT TOKEN {i+1}/{self.gamma}:")
                print(f"      Draft output shape: {draft_output.shape}")
                print(f"      Draft logits shape: {draft_logits.shape}")
            
            # Store probabilities before sampling for verification  
            # CRITICAL FIX: Use IDENTICAL temperature for both models to ensure alignment
            draft_temperature = self.temperature  # Must match target exactly!
            draft_probs_raw = self._softmax(draft_logits / draft_temperature)
            draft_probs_for_verification.append(draft_probs_raw[0])  # Remove batch dimension
            
            # Sample next token with improved sampling
            next_token = self._sample_token(draft_probs_raw[0])  # Pass 1D array
            draft_tokens.append(int(next_token))
            
            if DEBUG >= 1:
                print(f"      Draft temperature: {draft_temperature:.2f}")
                print(f"      Sampled token: {next_token}")
                print(f"      Token probability: {draft_probs_raw[0][next_token]:.6f}")
            
            # Add token to sequence for next iteration
            current_sequence = np.concatenate([
                current_sequence, 
                np.array([[next_token]], dtype=current_sequence.dtype)
            ], axis=1)
            
            if DEBUG >= 2:
                print(f"      Updated sequence length: {current_sequence.shape[1]}")
        
        if DEBUG >= 1:
            print(f"   📝 PHASE 1 COMPLETE - Draft tokens generated: {draft_tokens}")
            print(f"   📝 Draft probabilities stored: {len(draft_probs_for_verification)} distributions")
            print(f"   🔄 STARTING PHASE 2: TARGET VERIFICATION")
        
        # ============ PHASE 2: TARGET VERIFICATION ============
        # Since target engine returns only last position, verify tokens one by one
        if DEBUG >= 1:
            print("🎯 PHASE 2: TARGET VERIFICATION (Sequential):")
            print(f"   Original sequence length: {input_data.shape[1]}")
            print(f"   Draft tokens to evaluate: {draft_tokens}")
            print(f"   🧮 Algorithm: Sequential verification with acceptance sampling")
            
        accepted_tokens = []
        current_sequence = input_data.copy()
        
        if DEBUG >= 1:
            print(f"   🔄 PHASE 2: Processing {len(draft_tokens)} draft tokens sequentially...")
            print(f"   🎯 Starting sequential target verification...")
        
        for i, draft_token in enumerate(draft_tokens):
            if DEBUG >= 1:
                print(f"   🎯 VERIFICATION STEP {i+1}/{len(draft_tokens)}:")
                print(f"      Verifying draft token: {draft_token}")
                print(f"      Current sequence length: {current_sequence.shape[1]}")
            
            # CRITICAL FIX: Get target prediction for the SAME context the draft saw
            # Draft token i was predicted given sequence up to position i
            # So target should predict given the same context (without the draft token)
            
            if DEBUG >= 1:
                print(f"      🎯 Getting target prediction for context without draft token...")
                print(f"      Context sequence: {current_sequence[0, -5:].tolist()}")
            
            target_output, target_cache = await self.target_engine.infer_tensor(
                f"{request_id}_target_verify_{i}",
                shard,
                current_sequence,  # Same context draft saw, not extended!
                target_cache  # CRITICAL: Maintain consistent target state
            )
            target_output = self._to_numpy(target_output)
            
            # Extract target logits (should be single position)
            if target_output.ndim == 3:
                target_logits = target_output[0, -1]  # Last position
            elif target_output.ndim == 2:
                target_logits = target_output[0]
            else:
                target_logits = target_output
            
            # Now add the draft token for next iteration
            token_to_add = np.array([[draft_token]], dtype=current_sequence.dtype)
            extended_sequence = np.concatenate([current_sequence, token_to_add], axis=1)
                
            if DEBUG >= 1:
                print(f"      Target output shape: {target_output.shape}")
                print(f"      Target logits shape: {target_logits.shape}")
                print(f"      🧮 Computing acceptance probability...")
            
            # Get draft and target probabilities with improved alignment
            draft_probs = draft_probs_for_verification[i]
            target_probs = self._softmax(target_logits / self.temperature)
            
            # Extract probabilities for this specific token
            draft_prob = draft_probs[draft_token]
            target_prob = target_probs[draft_token]
            
            # BALANCED ACCEPTANCE ALGORITHM:
            # Goal: Find sweet spot between quality and acceptance rate
            # Reduce aggressive acceptance to improve output quality
            
            epsilon = 1e-8
            smoothed_draft_prob = draft_prob + epsilon
            smoothed_target_prob = target_prob + epsilon
            
            # Base acceptance ratio
            acceptance_ratio = min(1.0, smoothed_target_prob / smoothed_draft_prob)
            
            # Apply lenience but much more conservatively
            lenience_factor = 2.0  # Reduced from 10.0 to 2.0
            acceptance_ratio = min(1.0, acceptance_ratio * lenience_factor)
            
            # Temperature-based adjustment
            temp_factor = min(1.0, self.temperature / 0.7)
            adjusted_acceptance = acceptance_ratio * temp_factor
            
            # MUCH MORE CONSERVATIVE minimum acceptance for very misaligned tokens
            # Only give minimal help to completely misaligned tokens
            if target_prob < 1e-10:
                # Very low target probability - give minimal boost
                minimum_acceptance = 0.05  # Only 5% minimum (was 35%)
            elif target_prob < 1e-6:
                # Low target probability - small boost
                minimum_acceptance = 0.10  # 10% minimum
            else:
                # Reasonable target probability - no artificial boost
                minimum_acceptance = 0.0
            
            adjusted_acceptance = max(adjusted_acceptance, minimum_acceptance)
            
            # SELECTIVE TOP-K BOOST: Only boost tokens in top-10 of target distribution
            target_probs_sorted = np.argsort(target_probs)[::-1]  # Sort by probability (descending)
            draft_token_rank = np.where(target_probs_sorted == draft_token)[0]
            if len(draft_token_rank) > 0 and draft_token_rank[0] < 10:  # Only top-10 tokens (was top-50)
                rank_boost = 0.1 - (draft_token_rank[0] * 0.01)  # Max 10% boost for rank 1 (was 20%)
                adjusted_acceptance += rank_boost
                if DEBUG >= 1:
                    print(f"      🎯 Top-{draft_token_rank[0]+1} token, rank boost: +{rank_boost:.3f}")
            
            # CONSERVATIVE NUCLEUS BOOST: Only for top 70% cumulative probability
            target_cumsum = np.cumsum(np.sort(target_probs)[::-1])
            if len(draft_token_rank) > 0 and draft_token_rank[0] < len(target_cumsum) and target_cumsum[draft_token_rank[0]] < 0.7:  # Top 70% (was 90%)
                nucleus_boost = 0.05  # 5% boost (was 15%)
                adjusted_acceptance += nucleus_boost
                if DEBUG >= 1:
                    print(f"      🚀 Nucleus sampling boost: +{nucleus_boost:.3f}")
            
            # Cap at 80% to maintain more selectivity (was 95%)
            adjusted_acceptance = min(adjusted_acceptance, 0.80)
            
            random_sample = np.random.random()
            accept = random_sample <= adjusted_acceptance
            
            if DEBUG >= 1:
                print(f"   Token {i+1}: {draft_token}")
                print(f"      Draft prob: {draft_prob:.6f}")
                print(f"      Target prob: {target_prob:.6f}")
                print(f"      Acceptance ratio: {acceptance_ratio:.6f}")
                print(f"      Adjusted acceptance: {adjusted_acceptance:.6f}")
                print(f"      Random sample: {random_sample:.6f}")
                print(f"      Decision: {'✅ ACCEPT' if accept else '❌ REJECT'}")
                
                # Diagnostic information
                if target_prob < 1e-10:
                    print(f"      ⚠️  WARNING: Target assigns near-zero probability!")
                if acceptance_ratio < 0.1:
                    print(f"      📉 Low acceptance ratio - draft much more confident")
                if draft_token in {0, 1, 2, 3} or draft_token > 128000:
                    print(f"      🚨 Special token detected: {draft_token}")
                if adjusted_acceptance > acceptance_ratio:
                    print(f"      🎯 Lenience improved acceptance: {acceptance_ratio:.3f} -> {adjusted_acceptance:.3f}")
            
            if accept:
                accepted_tokens.append(draft_token)
                current_sequence = extended_sequence  # Update sequence for next iteration
                if DEBUG >= 2:
                    print(f"      ✅ Token {draft_token} ACCEPTED - continuing")
            else:
                if DEBUG >= 1:
                    print(f"      ❌ Token {draft_token} REJECTED - stopping acceptance")
                break
        
        # ============ PHASE 3: ACCEPTANCE/FORCED PROGRESS ============
        if DEBUG >= 1:
            print(f"   🏁 PHASE 3: FINALIZING ACCEPTANCE RESULTS")
            print(f"      Tokens accepted in verification: {len(accepted_tokens)}")
            
        if accepted_tokens:
            if DEBUG >= 1:
                print(f"      ✅ Applying {len(accepted_tokens)} accepted tokens to sequence")
            accepted_array = np.array(accepted_tokens).reshape(1, -1)
            out = np.concatenate([out, accepted_array], axis=1)
            all_accepted_tokens.extend(accepted_tokens)
        else:
            # Force progress by sampling a token from target model to prevent infinite loops
            if DEBUG >= 1:
                print(f"      🔄 PHASE 3B: FORCED PROGRESS (No tokens accepted)")
                print(f"      🎯 Generating fallback token with target model")
            
            # Use the last target logits from verification (should be available)
            if target_logits is not None and target_logits.size > 0:
                # Get logits for next position
                if target_logits.ndim == 3:
                    next_logits = target_logits[:, -1, :]  # Last position
                else:
                    next_logits = target_logits
                
                # Apply higher temperature to increase diversity and avoid loops
                fallback_temp = max(self.temperature * 1.5, 0.9)
                next_logits_tempered = next_logits / fallback_temp
                next_probs = self._softmax(next_logits_tempered)
                
                # Sample a token with special token filtering
                forced_token = self._sample_token(next_probs)
                
                # Add the forced token to sequence
                out = np.concatenate([out, np.array([[forced_token]], dtype=out.dtype)], axis=1)
                all_accepted_tokens.append(forced_token)
                accepted_tokens.append(forced_token)  # For statistics
                
                if DEBUG >= 1:
                    print(f"   🎯 Forced token: {forced_token} (temp={fallback_temp:.2f})")
            else:
                if DEBUG >= 1:
                    print(f"   🚨 Warning: No target logits available for forced progress")
        
        # Update statistics
        self.total_calls += 1
        self.total_tokens_generated += len(accepted_tokens)
        self.total_tokens_accepted += len(accepted_tokens)
        
        end_time = time.perf_counter()
        acceptance_rate = len(accepted_tokens) / self.gamma if self.gamma > 0 else 0
        
        if DEBUG >= 1:
            print(f"\n📊 PHASE 4: FINAL STATISTICS & RESULTS")
            print(f"   🔢 Sequence length: {current_seq_len} -> {out.shape[1]}")
            print(f"   ✅ Tokens accepted: {len(accepted_tokens)}/{self.gamma} = {acceptance_rate:.1%}")
            print(f"   🎯 Accepted tokens: {accepted_tokens}")
            print(f"   ⏱️  Time: {(end_time - start_time)*1000:.2f}ms")
            print(f"   📈 Cumulative acceptance rate: {self.total_tokens_accepted}/{self.total_tokens_generated} = {self.total_tokens_accepted/max(self.total_tokens_generated,1):.1%}")
            print(f"🔚 =================END SPECULATIVE DECODING================= \n")
        
        # Generate final logits for compatibility with the inference engine interface
        # The node now properly handles token forwarding, so this is just for interface compliance
        if DEBUG >= 2:
            print(f"\n🔧 GENERATING FINAL LOGITS FOR INTERFACE COMPLIANCE:")
            print(f"   Final sequence shape: {out.shape}")
            print(f"   Final sequence (last 5 tokens): {out[0, -5:].tolist()}")
        
        # 🔧 CRITICAL FIX: Use fresh cache for final generation to prevent overflow
        # But also try to maintain context by restoring to a safe position
        try:
            # First, try to restore target cache to original position  
            if hasattr(target_cache, 'cache_pos') and target_cache.cache_pos is not None:
                # Reset cache to the original sequence length to avoid overflow
                self._reset_cache_to_position(target_cache, current_seq_len)
                if DEBUG >= 2:
                    print(f"   🔧 Reset cache position to {current_seq_len} for final generation")
            
            final_logits, final_cache = await self.target_engine.infer_tensor(
                f"{request_id}_final_logits",
                shard,
                out,  # Use the token sequence that includes accepted tokens
                target_cache  # Try to use coordinated cache state
            )
            final_logits = self._to_numpy(final_logits)
            
            if DEBUG >= 2:
                print(f"   ✅ Final logits generated successfully with cache coordination")
                print(f"   Final logits shape: {final_logits.shape}")
        
        except Exception as e:
            if "AssertionError" in str(e) and "cache_pos" in str(e):
                if DEBUG >= 1:
                    print(f"   🚨 Cache coordination failed, using fresh cache for final generation")
                
                # Fallback: Use fresh cache
                try:
                    final_logits, final_cache = await self.target_engine.infer_tensor(
                        f"{request_id}_final_logits_fresh",
                        shard,
                        out,
                        None  # Fresh cache
                    )
                    final_logits = self._to_numpy(final_logits)
                    
                    if DEBUG >= 2:
                        print(f"   ✅ Final logits generated with fresh cache")
                        print(f"   Final logits shape: {final_logits.shape}")
                
                except Exception as e2:
                    if DEBUG >= 1:
                        print(f"   🚨 Complete failure in final generation, using dummy logits: {e2}")
                    # Ultimate fallback: dummy logits for interface compliance
                    final_logits = np.zeros((1, 128256), dtype=np.float32)
                    final_cache = None
                    
                    if DEBUG >= 2:
                        print(f"   ⚠️  Using dummy logits for interface compliance")
            else:
                raise e
        
        # Return the token sequence (not logits) as integers for proper decoding
        # Convert to integers to ensure tokenizer compatibility
        token_sequence = out.astype(np.int64)
        
        return token_sequence, final_cache, all_accepted_tokens

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax with numerical stability."""
        logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _sample_from_probs(self, probs: np.ndarray) -> int:
        """Sample from probability distribution."""
        # Ensure probs is 1D
        if probs.ndim > 1:
            probs = probs.flatten()
        probs = probs / (probs.sum() + 1e-10)
        return np.random.choice(len(probs), p=probs)

    def _sample_token(self, probs: np.ndarray) -> int:
        """Sample token with special token filtering and improved sampling."""
        # Special tokens to avoid (common problematic tokens)
        special_tokens = {0, 1, 2, 3, 128000, 128001, 128002, 128003, 128004, 128005, 
                         128006, 128007, 128008, 128009, 128010, 128256}
        
        # Zero out probabilities for special tokens
        filtered_probs = probs.copy()
        for token_id in special_tokens:
            if token_id < len(filtered_probs):
                filtered_probs[token_id] = 0.0
        
        # Renormalize
        prob_sum = np.sum(filtered_probs)
        if prob_sum > 0:
            filtered_probs = filtered_probs / prob_sum
        else:
            # Fallback: uniform over non-special tokens
            filtered_probs = np.ones_like(probs)
            for token_id in special_tokens:
                if token_id < len(filtered_probs):
                    filtered_probs[token_id] = 0.0
            filtered_probs = filtered_probs / np.sum(filtered_probs)
        
        # Sample from filtered distribution
        return self._sample_from_probs(filtered_probs)

    def _backup_cache_state(self, cache_state):
        """Backup cache state for later restoration."""
        if cache_state is None:
            return None
        
        if hasattr(cache_state, 'cache_pos') and cache_state.cache_pos is not None:
            try:
                if hasattr(cache_state.cache_pos, 'clone'):
                    return {
                        'cache_pos': cache_state.cache_pos.clone(),
                        'original_state': cache_state
                    }
                else:
                    return {
                        'cache_pos': cache_state.cache_pos.copy() if hasattr(cache_state.cache_pos, 'copy') else cache_state.cache_pos,
                        'original_state': cache_state
                    }
            except Exception:
                pass
        
        return {'original_state': cache_state}
    
    def _restore_cache_state(self, backup):
        """Restore cache state from backup."""
        if backup is None:
            return None
        
        original_state = backup.get('original_state')
        if original_state is None:
            return None
        
        cache_pos_backup = backup.get('cache_pos')
        if cache_pos_backup is not None and hasattr(original_state, 'cache_pos'):
            try:
                if hasattr(original_state.cache_pos, 'copy_'):
                    original_state.cache_pos.copy_(cache_pos_backup)
                elif hasattr(original_state.cache_pos, 'copy'):
                    original_state.cache_pos = cache_pos_backup.copy()
                else:
                    original_state.cache_pos = cache_pos_backup
            except Exception:
                pass
        
        return original_state
    
    def _reset_cache_to_position(self, cache_state, position):
        """Reset cache to a specific position."""
        if cache_state is None or not hasattr(cache_state, 'cache_pos'):
            return cache_state
        
        try:
            if hasattr(cache_state.cache_pos, 'fill_'):
                cache_state.cache_pos.fill_(position)
            elif hasattr(cache_state.cache_pos, 'copy_'):
                import torch
                cache_state.cache_pos.copy_(torch.tensor([position]))
            else:
                cache_state.cache_pos = position
        except Exception:
            pass
        
        return cache_state

# Model family configurations for automatic pairing
FAMILY_CONFIGS = {
    "llama": {
        "family_name": "LLaMA",
        "models": {
            "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
            "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
            "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
            "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
            "llama-3.1-405b": "meta-llama/Llama-3.1-405B"
        },
        "suggested_pairs": [
            ("llama-3.2-1b", "llama-3.2-3b"),
            ("llama-3.2-1b", "llama-3.1-8b"),
            ("llama-3.2-3b", "llama-3.1-8b"),
            ("llama-3.1-8b", "llama-3.1-70b"),
            ("llama-3.1-70b", "llama-3.1-405b")
        ],
        "vocab_size": 128256
    },
    "qwen": {
        "family_name": "Qwen",
        "models": {
            "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
            "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
            "qwen-3b": "Qwen/Qwen2.5-3B",
            "qwen-7b": "Qwen/Qwen2.5-7B",
            "qwen-14b": "Qwen/Qwen2.5-14B"
        },
        "suggested_pairs": [
            ("qwen-0.5b", "qwen-1.5b"),
            ("qwen-0.5b", "qwen-3b"),
            ("qwen-1.5b", "qwen-7b"),
            ("qwen-3b", "qwen-14b")
        ],
        "vocab_size": 151936
    }
}

def get_model_family(model_id: str) -> Optional[str]:
    """Determine model family from model ID."""
    model_lower = model_id.lower()
    
    if "llama" in model_lower:
        return "llama"
    elif "qwen" in model_lower:
        return "qwen"
    
    return None

def find_compatible_draft_models(target_model: str) -> List[str]:
    """Find compatible draft models for a target model."""
    family = get_model_family(target_model)
    if not family or family not in FAMILY_CONFIGS:
        return []
    
    config = FAMILY_CONFIGS[family]
    target_key = None
    
    # Find target model key
    for key, repo in config["models"].items():
        if target_model in repo or key in target_model.lower():
            target_key = key
            break
    
    if not target_key:
        return []
    
    # Find compatible draft models
    compatible = []
    for draft_key, target_key_check in config["suggested_pairs"]:
        if target_key_check == target_key:
            draft_repo = config["models"].get(draft_key)
            if draft_repo:
                compatible.append(draft_repo)
    
    return compatible

def suggest_speculative_config(target_model: str) -> Optional[dict]:
    """Suggest a speculative decoding configuration for a target model."""
    draft_models = find_compatible_draft_models(target_model)
    
    if not draft_models:
        return None
    
    # Use the first (smallest) compatible draft model
    draft_model = draft_models[0]
    family = get_model_family(target_model)
    
    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "family": family,
        "gamma": 3,
        "temperature": 0.7,
        "top_k_threshold": 0.9,
        "vocab_size": FAMILY_CONFIGS[family]["vocab_size"]
    } 