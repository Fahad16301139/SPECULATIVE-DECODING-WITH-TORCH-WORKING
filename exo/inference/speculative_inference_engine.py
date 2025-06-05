import time
import numpy as np
from typing import Optional, Tuple, List
from .inference_engine import InferenceEngine
from .shard import Shard

# Enable debug for troubleshooting
DEBUG = 1

class SpeculativeInferenceEngine(InferenceEngine):
    """
    Wrapper engine that performs speculative decoding using draft and target models.
    """
    
    def __init__(self, target_engine: InferenceEngine, draft_engine: InferenceEngine,
                 gamma: int = 6, temperature: float = 0.8, top_k_threshold: float = 0.0,
                 lenience: float = 1.1, target_model_id: str = None, draft_model_id: str = None):
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
        
        if DEBUG >= 1:
            print(f"🚀 SpeculativeInferenceEngine initialized:")
            print(f"   gamma={gamma}, temperature={temperature}")
            print(f"   top_k_threshold={top_k_threshold}, lenience={lenience}")

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode prompt using target engine."""
        return await self.target_engine.encode(shard, prompt)

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
        return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)

    async def infer_tensor_multi(self, request_id: str, shard: Shard, input_data: np.ndarray,
                                inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict], Optional[list]]:
        """Multi-token inference with speculative decoding."""
        if self.draft_engine is None:
            # No draft engine, fallback to target only
            output, state = await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            return output, state, []
        
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
        Vanilla speculative decoding implementation.
        """
        start_time = time.perf_counter()
        
        # Add extensive debugging for TinyGrad model verification
        if DEBUG >= 1:
            print(f"\n🔮 =================SPECULATIVE DECODING DEBUG================= ")
            print(f"📊 Input shape: {input_data.shape}")
            print(f"🎯 Target engine: {type(self.target_engine).__name__}")
            print(f"📝 Draft engine: {type(self.draft_engine).__name__}")
            
            # DETAILED MODEL LOADING VERIFICATION
            print(f"\n🔍 DETAILED MODEL ANALYSIS:")
            
            # Check if engines are the same object (major red flag)
            if self.target_engine is self.draft_engine:
                print(f"🚨 CRITICAL ERROR: Target and draft engines are THE SAME OBJECT!")
                print(f"   This means no speculative decoding is happening at all!")
            else:
                print(f"✅ Target and draft engines are different objects")
            
            # Model repository verification
            target_repo = getattr(self.target_engine, 'repo', 'NO_REPO_ATTR')
            draft_repo = getattr(self.draft_engine, 'repo', 'NO_REPO_ATTR')
            print(f"🎯 Target repo: {target_repo}")
            print(f"📝 Draft repo: {draft_repo}")
            
            if target_repo == draft_repo:
                print(f"🚨 WARNING: Both engines using same repo: {target_repo}")
            
            # Check model paths/files being loaded
            target_model_path = getattr(self.target_engine, 'model_path', 'NO_MODEL_PATH')
            draft_model_path = getattr(self.draft_engine, 'model_path', 'NO_MODEL_PATH')
            print(f"🎯 Target model path: {target_model_path}")
            print(f"📝 Draft model path: {draft_model_path}")
            
            # Check shard configurations
            target_shard_info = f"start={shard.start_layer}, end={shard.end_layer}, n_layers={shard.n_layers}"
            draft_shard = self._create_draft_shard_from_target(shard)
            draft_shard_info = f"start={draft_shard.start_layer}, end={draft_shard.end_layer}, n_layers={draft_shard.n_layers}"
            print(f"🎯 Target shard: {target_shard_info}")
            print(f"📝 Draft shard: {draft_shard_info}")
            print(f"🎯 Target model_id: {shard.model_id}")
            print(f"📝 Draft model_id: {draft_shard.model_id}")
            
            # Check actual loaded model details if available
            if hasattr(self.target_engine, 'model') and hasattr(self.draft_engine, 'model'):
                if self.target_engine.model is self.draft_engine.model:
                    print(f"🚨 CRITICAL: Target and draft are using THE SAME MODEL INSTANCE!")
                else:
                    print(f"✅ Target and draft are using different model instances")
                    
            # Memory addresses for verification
            print(f"🎯 Target engine memory: {hex(id(self.target_engine))}")
            print(f"📝 Draft engine memory: {hex(id(self.draft_engine))}")
            
            print(f"🎲 Gamma (draft tokens): {self.gamma}")
            
            # Validate algorithm parameters
            print(f"\n📋 ALGORITHM PARAMETERS:")
            print(f"   Temperature: {self.temperature}")
            print(f"   Top-k threshold: {self.top_k_threshold}")
            print(f"   Lenience: {self.lenience}")
            print(f"   Target engine type: {type(self.target_engine)}")
            print(f"   Draft engine type: {type(self.draft_engine)}")
            
            # Check if we're actually doing speculative decoding or just regular inference
            if self.draft_engine is None:
                print(f"🚨 NO DRAFT ENGINE - This is regular inference, not speculative!")
            elif self.gamma <= 0:
                print(f"🚨 GAMMA <= 0 - No draft tokens will be generated!")
            else:
                print(f"✅ Speculative parameters look valid")
        
        # Initialize working variables - input should always be token IDs now
        if DEBUG >= 2:
            print(f"🔍 INPUT VALIDATION: shape={input_data.shape}, dtype={input_data.dtype}")
            print(f"   Range: [{input_data.min():.3f}, {input_data.max():.3f}]")
            if input_data.ndim == 3 and input_data.shape[-1] > 1000:
                print(f"🚨 WARNING: Input has shape suggesting logits, but node should pass token IDs!")
        
        # Handle token ID input (should be the only case now)
        if input_data.ndim == 1:
            out = input_data.reshape(1, -1).copy()
        else:
            out = input_data.copy()
            
        # CRITICAL: Check for empty input early to prevent crashes
        if out.size == 0 or (out.ndim >= 1 and out.shape[-1] == 0):
            if DEBUG >= 1:
                print(f"🚨 EMPTY INPUT DETECTED: {out.shape}")
                print(f"   Cannot perform speculative decoding on empty sequence!")
                print(f"   Falling back to target engine only...")
            
            # Fallback to target engine for empty input
            result, state = await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            return result, state, []
            
        cache = inference_state
        small_cache = None
        all_accepted_tokens = []
        
        # Safely get sequence length
        if out.ndim == 1:
            current_seq_len = out.shape[0]
            out = out.reshape(1, -1)
        else:
            current_seq_len = out.shape[1]
        
        # Create draft shard
        draft_shard = self._create_draft_shard_from_target(shard)
        
        # COMPREHENSIVE DEBUGGING: Model verification
        if DEBUG >= 1:
            print(f"\n🔍 MODEL VERIFICATION PHASE:")
            print(f"   Input 'out' shape: {out.shape}, dtype: {out.dtype}")
            if out.size > 0:
                print(f"   Input 'out' range: [{out.min():.3f}, {out.max():.3f}]")
                print(f"   Input 'out' sample values: {out.flatten()[:5].tolist()}")
            else:
                print(f"   🚨 CRITICAL: Input 'out' is EMPTY! This will cause failures.")
                print(f"   Input 'out' sample values: []")
            
            # Test both engines on same input to verify they're different
            # Ensure we handle 1D input_data properly
            if out.ndim == 1:
                test_input = out[:min(5, out.shape[0])].reshape(1, -1)
            else:
                test_input = out[:, :min(5, out.shape[1])]  # First 5 tokens
            
            # Skip model verification if test input would be empty
            if test_input.size == 0:
                print(f"   ⚠️  Skipping model verification - test input would be empty")
                print(f"   Test input shape: {test_input.shape}")
            else:
                print(f"   Testing both models on input shape: {test_input.shape}")
                print(f"   Test input tokens: {test_input.flatten()[:5].astype(np.int64).tolist()}")
                print(f"   Test input dtype: {test_input.dtype}")
                
                # Ensure test input is integer tokens
                if test_input.dtype != np.int64:
                    print(f"   🔧 Converting test input from {test_input.dtype} to int64")
                    test_input = test_input.astype(np.int64)
            
            # CRITICAL: Check if models are actually being loaded from different sources
            print(f"\n   🔍 PRE-INFERENCE MODEL STATE:")
            
            # Check what actual model files/weights are loaded
            target_loaded_info = "UNKNOWN"
            draft_loaded_info = "UNKNOWN"
            
            if hasattr(self.target_engine, 'repo'):
                target_loaded_info = f"repo={self.target_engine.repo}"
            if hasattr(self.draft_engine, 'repo'):
                draft_loaded_info = f"repo={self.draft_engine.repo}"
                
            print(f"   🎯 Target loaded from: {target_loaded_info}")
            print(f"   📝 Draft loaded from: {draft_loaded_info}")
            
            # Check if they're loading the same model ID
            actual_target_model = getattr(shard, 'model_id', 'UNKNOWN')
            actual_draft_model = getattr(draft_shard, 'model_id', 'UNKNOWN')
            print(f"   🎯 Target model ID: {actual_target_model}")
            print(f"   📝 Draft model ID: {actual_draft_model}")
            
            if actual_target_model == actual_draft_model:
                print(f"   🚨 SMOKING GUN: Both engines loading same model ID: {actual_target_model}")
                print(f"   🚨 This explains why they're identical!")
                
            # Check if there's any model override happening
            print(f"\n   🔍 ENGINE CONFIGURATION CHECK:")
            target_config = {}
            draft_config = {}
            
            for attr in ['model_id', 'model_path', 'config', 'repo']:
                if hasattr(self.target_engine, attr):
                    target_config[attr] = getattr(self.target_engine, attr)
                if hasattr(self.draft_engine, attr):
                    draft_config[attr] = getattr(self.draft_engine, attr)
            
            print(f"   🎯 Target config: {target_config}")
            print(f"   📝 Draft config: {draft_config}")
            
            # Only run model inference if we have valid test input
            if test_input.size > 0:
                target_test, _ = await self.target_engine.infer_tensor(f"{request_id}_verify_target", shard, test_input)
                draft_test, _ = await self.draft_engine.infer_tensor(f"{request_id}_verify_draft", draft_shard, test_input)
            else:
                print(f"   ⚠️  Skipping model inference verification due to empty test input")
                # Continue with rest of the function without model comparison
            
            print(f"\n   🎯 Target output shape: {target_test.shape}")
            print(f"   📝 Draft output shape: {draft_test.shape}")
            print(f"   🎯 Target logits range: [{target_test.min():.3f}, {target_test.max():.3f}]")
            print(f"   📝 Draft logits range: [{draft_test.min():.3f}, {draft_test.max():.3f}]")
            
            # More detailed comparison
            if target_test.shape == draft_test.shape:
                diff = np.abs(target_test - draft_test).mean()
                max_diff = np.abs(target_test - draft_test).max()
                
                print(f"   📊 Mean absolute difference: {diff:.10f}")
                print(f"   📊 Max absolute difference: {max_diff:.10f}")
                
                # Sample specific logits for comparison
                sample_indices = [0, 100, 1000, 10000] if target_test.shape[-1] > 10000 else [0, 1, 2, 3]
                print(f"\n   🔍 SAMPLE LOGIT COMPARISON:")
                for idx in sample_indices:
                    if idx < target_test.shape[-1]:
                        target_val = target_test[0, -1, idx]
                        draft_val = draft_test[0, -1, idx]
                        print(f"      Token {idx}: Target={target_val:.6f}, Draft={draft_val:.6f}, Diff={abs(target_val-draft_val):.10f}")
                
                if diff < 1e-15:
                    print(f"   🚨 IDENTICAL: Models are EXACTLY the same (machine precision)")
                elif diff < 1e-6:
                    print(f"   ⚠️  WARNING: Models appear to be identical! (diff < 1e-6)")
                elif diff < 1e-3:
                    print(f"   ⚠️  WARNING: Models very similar! (diff < 1e-3)")
                else:
                    print(f"   ✅ Models appear different (diff = {diff:.6f})")
            
            # Vocabulary analysis
            if target_test.shape[-1] != draft_test.shape[-1]:
                print(f"   🔍 VOCAB SIZE MISMATCH:")
                print(f"      Target vocab: {target_test.shape[-1]}")
                print(f"      Draft vocab: {draft_test.shape[-1]}")
                print(f"      🚨 CRITICAL: Different vocab sizes will break speculative decoding!")
                print(f"      🚨 Models are NOT compatible for speculative decoding!")
            else:
                print(f"   ✅ Both models have same vocab size: {target_test.shape[-1]}")
                
                # Additional tokenizer compatibility checks
                print(f"\n   🔍 TOKENIZER COMPATIBILITY CHECK:")
                target_tokenizer = getattr(self.target_engine, 'tokenizer', None)
                draft_tokenizer = getattr(self.draft_engine, 'tokenizer', None)
                
                if target_tokenizer is None or draft_tokenizer is None:
                    print(f"   ⚠️  Cannot access tokenizers for comparison")
                    print(f"      Target tokenizer: {type(target_tokenizer) if target_tokenizer else 'None'}")
                    print(f"      Draft tokenizer: {type(draft_tokenizer) if draft_tokenizer else 'None'}")
                else:
                    # Test tokenization of a simple string
                    test_string = "Hello world"
                    try:
                        target_tokens = target_tokenizer.encode(test_string) if hasattr(target_tokenizer, 'encode') else None
                        draft_tokens = draft_tokenizer.encode(test_string) if hasattr(draft_tokenizer, 'encode') else None
                        
                        if target_tokens is not None and draft_tokens is not None:
                            if target_tokens == draft_tokens:
                                print(f"   ✅ Tokenizers produce identical output for test string")
                                print(f"      Test: '{test_string}' -> {target_tokens}")
                            else:
                                print(f"   🚨 TOKENIZER MISMATCH:")
                                print(f"      Target: '{test_string}' -> {target_tokens}")
                                print(f"      Draft:  '{test_string}' -> {draft_tokens}")
                                print(f"      🚨 Different tokenization will break speculative decoding!")
                        else:
                            print(f"   ⚠️  Could not test tokenizer compatibility (encode method not available)")
                    except Exception as e:
                        print(f"   ⚠️  Error testing tokenizers: {e}")
                
                # Check vocabulary token mappings for some common tokens
                print(f"\n   🔍 VOCAB TOKEN MAPPING CHECK:")
                common_tokens = [0, 1, 2, 100, 1000]  # BOS, EOS, UNK, and some common tokens
                for token_id in common_tokens:
                    if token_id < target_test.shape[-1]:
                        target_logit = target_test[0, -1, token_id].item()
                        draft_logit = draft_test[0, -1, token_id].item()
                        print(f"      Token {token_id}: Target={target_logit:.6f}, Draft={draft_logit:.6f}")
                        
                        # If models are truly different, we should see different logits
                        # If they're the same (problematic), logits will be identical
        
        if DEBUG >= 1:
            print(f"\n📝 DRAFT GENERATION PHASE:")
        
        # Generate gamma tokens with draft model
        draft_tokens = []
        draft_input = out.copy()
        
        for i in range(self.gamma):
            if DEBUG >= 2:
                print(f"   🎲 Generating draft token {i+1}/{self.gamma}...")
                print(f"      Current sequence length: {draft_input.shape[1]}")
            
            draft_output, small_cache = await self.draft_engine.infer_tensor(
                f"{request_id}_draft_{i}",
                draft_shard,
                draft_input,
                small_cache
            )
            
            # Sample from draft
            draft_logits = draft_output[:, -1:, :]  # Last token logits
            if DEBUG >= 3:
                print(f"      Draft logits shape: {draft_logits.shape}")
                print(f"      Draft logits range: [{draft_logits.min():.3f}, {draft_logits.max():.3f}]")
            
            # Apply temperature to logits before softmax
            draft_logits_tempered = draft_logits / self.temperature
            draft_probs = self._softmax(draft_logits_tempered)
            draft_token = self._sample_from_probs(draft_probs[0, 0, :])
            
            draft_tokens.append(draft_token)
            
            if DEBUG >= 2:
                print(f"      ✅ Sampled draft token: {draft_token}")
                print(f"      Draft probability: {draft_probs[0, 0, draft_token]:.6f}")
            
            # Prepare next input
            new_token = np.array([[draft_token]], dtype=np.int64)
            draft_input = np.concatenate([draft_input, new_token], axis=1)
        
        if DEBUG >= 1:
            print(f"   📝 Draft tokens generated: {draft_tokens}")
        
        # Add draft tokens to sequence
        draft_sequence = np.array(draft_tokens).reshape(1, -1)
        extended_input = np.concatenate([out, draft_sequence], axis=1)
        
        if DEBUG >= 1:
            print(f"\n🎯 TARGET VERIFICATION PHASE:")
            print(f"   Original sequence length: {out.shape[1]}")
            print(f"   Extended sequence length: {extended_input.shape[1]}")
        
        # Get target model's opinion on the extended sequence
        target_output, cache = await self.target_engine.infer_tensor(
            f"{request_id}_target",
            shard,
            extended_input,
            cache
        )
        
        if DEBUG >= 1:
            print(f"   Target output shape: {target_output.shape}")
        
        # DETAILED ACCEPTANCE ANALYSIS
        if DEBUG >= 1:
            print(f"\n⚖️  ACCEPTANCE ANALYSIS PHASE:")
            print(f"   🔍 ALGORITHM VALIDATION:")
            print(f"      Original sequence length: {current_seq_len}")
            print(f"      Draft tokens to verify: {len(draft_tokens)}")
            print(f"      Target output covers positions: {current_seq_len-1} to {current_seq_len-1 + len(draft_tokens)}")
            print(f"      Target logits shape: {target_output.shape}")
            
            # Validate we're following vanilla algorithm structure
            if len(draft_tokens) != self.gamma:
                print(f"   ⚠️  Draft token count mismatch: expected {self.gamma}, got {len(draft_tokens)}")
            
            if target_output.shape[1] != current_seq_len + len(draft_tokens):
                print(f"   ⚠️  Target output length mismatch: expected {current_seq_len + len(draft_tokens)}, got {target_output.shape[1]}")
        
        accepted_tokens = []
        target_logits = target_output[:, current_seq_len-1:, :]  # Logits for new positions
        
        if DEBUG >= 1:
            print(f"   🎯 Target logits extracted shape: {target_logits.shape}")
            print(f"   📊 Starting vanilla acceptance loop for {self.gamma} tokens...")
        
        for i in range(self.gamma):
            target_pos_logits = target_logits[:, i, :]  # Target logits at position i
            # Apply temperature to target logits before softmax
            target_pos_logits_tempered = target_pos_logits / self.temperature
            target_probs = self._softmax(target_pos_logits_tempered)
            
            # Get draft and target probabilities for the drafted token
            drafted_token = draft_tokens[i]
            target_prob = target_probs[0, drafted_token]
            
            # Draft probability (re-compute for exact comparison)
            if i == 0:
                # For first token, use original sequence
                draft_context = out
            else:
                # For subsequent tokens, use sequence + accepted tokens so far
                accepted_so_far = np.array(draft_tokens[:i]).reshape(1, -1)
                draft_context = np.concatenate([out, accepted_so_far], axis=1)
                
            draft_output_verify, _ = await self.draft_engine.infer_tensor(
                f"{request_id}_draft_verify_{i}",
                draft_shard,
                draft_context,
                None  # Fresh cache for verification
            )
            draft_verify_logits = draft_output_verify[:, -1, :]
            # Apply temperature to draft verification logits before softmax
            draft_verify_logits_tempered = draft_verify_logits / self.temperature
            draft_verify_probs = self._softmax(draft_verify_logits_tempered)
            draft_prob = draft_verify_probs[0, drafted_token]
            
            # Acceptance probability calculation
            acceptance_ratio = target_prob / (draft_prob + 1e-10)
            acceptance_prob = min(1.0, acceptance_ratio)
            
            # Sample uniform random for acceptance decision
            uniform_sample = np.random.random()
            accept = uniform_sample < acceptance_prob
            
            if DEBUG >= 1:
                print(f"   Token {i+1}: {drafted_token}")
                print(f"      Draft prob: {draft_prob:.6f}")
                print(f"      Target prob: {target_prob:.6f}")
                print(f"      Acceptance ratio: {acceptance_ratio:.6f}")
                print(f"      Acceptance prob: {acceptance_prob:.6f}")
                print(f"      Random sample: {uniform_sample:.6f}")
                print(f"      Decision: {'✅ ACCEPT' if accept else '❌ REJECT'}")
                
                # VANILLA ALGORITHM VALIDATION
                print(f"      🔍 VANILLA VERIFICATION:")
                print(f"         Following r > (p/q) rejection rule: {uniform_sample} > {acceptance_ratio:.6f} = {uniform_sample > acceptance_ratio}")
                print(f"         Correct rejection condition: {not accept and uniform_sample >= acceptance_prob}")
                
                # Check if probabilities are suspiciously identical
                prob_diff = abs(target_prob - draft_prob)
                if prob_diff < 1e-10:
                    print(f"      🚨 IDENTICAL PROBS: Target and draft probs are identical! (diff={prob_diff:.2e})")
                    print(f"      🚨 This confirms models are the same!")
                
                # DIAGNOSTIC: Check for zero probabilities
                if target_prob < 1e-10:
                    print(f"      ⚠️  WARNING: Target assigns near-zero probability!")
                if draft_prob < 1e-10:
                    print(f"      ⚠️  WARNING: Draft assigns near-zero probability!")
                if acceptance_ratio > 10:
                    print(f"      📈 High acceptance ratio - target much more confident")
                elif acceptance_ratio < 0.1:
                    print(f"      📉 Low acceptance ratio - draft much more confident")
                    
                # Check if we're just accepting everything (sign of identical models)
                if acceptance_prob >= 0.999:
                    print(f"      🚨 ALWAYS ACCEPTING: Acceptance prob ≈ 1.0 indicates identical models!")
            
            if accept:
                accepted_tokens.append(drafted_token)
                if DEBUG >= 2:
                    print(f"      ✅ Token {drafted_token} ACCEPTED")
            else:
                if DEBUG >= 1:
                    print(f"      ❌ Token {drafted_token} REJECTED - stopping acceptance")
                break
        
        # Update sequence with accepted tokens
        if accepted_tokens:
            accepted_array = np.array(accepted_tokens).reshape(1, -1)
            out = np.concatenate([out, accepted_array], axis=1)
            all_accepted_tokens.extend(accepted_tokens)
        
        # Update statistics
        self.total_calls += 1
        self.total_tokens_generated += len(accepted_tokens)
        self.total_tokens_accepted += len(accepted_tokens)
        
        end_time = time.perf_counter()
        acceptance_rate = len(accepted_tokens) / self.gamma if self.gamma > 0 else 0
        
        if DEBUG >= 1:
            print(f"\n📊 FINAL RESULTS:")
            print(f"   Sequence length: {current_seq_len} -> {out.shape[1]}")
            print(f"   Tokens accepted: {len(accepted_tokens)}/{self.gamma} = {acceptance_rate:.1%}")
            print(f"   Accepted tokens: {accepted_tokens}")
            print(f"   Time: {(end_time - start_time)*1000:.2f}ms")
            print(f"   Cumulative acceptance rate: {self.total_tokens_accepted}/{self.total_tokens_generated} = {self.total_tokens_accepted/max(self.total_tokens_generated,1):.1%}")
            print(f"=================END SPECULATIVE DECODING================= \n")
        
        # Generate final logits for compatibility with the inference engine interface
        # The node now properly handles token forwarding, so this is just for interface compliance
        if DEBUG >= 2:
            print(f"\n🔧 GENERATING FINAL LOGITS FOR INTERFACE COMPLIANCE:")
            print(f"   Final sequence shape: {out.shape}")
            print(f"   Final sequence (last 5 tokens): {out[0, -5:].tolist()}")
        
        # Generate logits for the final sequence using target model
        final_logits, final_cache = await self.target_engine.infer_tensor(
            f"{request_id}_final_logits",
            shard,
            out,  # Use the token sequence that includes accepted tokens
            cache
        )
        
        if DEBUG >= 2:
            print(f"   Final logits shape: {final_logits.shape}")
        
        return final_logits, final_cache, all_accepted_tokens

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax with numerical stability."""
        logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _sample_from_probs(self, probs: np.ndarray) -> int:
        """Sample from probability distribution."""
        probs = probs / (probs.sum() + 1e-10)
        return np.random.choice(len(probs), p=probs)

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
        "gamma": 6,
        "temperature": 0.8,
        "top_k_threshold": 0.0,
        "vocab_size": FAMILY_CONFIGS[family]["vocab_size"]
    } 