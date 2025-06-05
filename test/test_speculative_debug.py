#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive debug output for speculative decoding.
This script shows all phases of the speculative decoding algorithm with detailed logging.
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import exo modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.inference.shard import Shard
from exo.helpers import DEBUG

# Mock tokenizer for testing
class MockTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
    
    def encode(self, text):
        # Simple hash-based encoding for consistent results
        return [hash(text) % self.vocab_size]
    
    def decode(self, tokens):
        return f"decoded_{tokens[0]}" if len(tokens) > 0 else ""

# Enhanced dummy engine with tokenizer
class EnhancedDummyEngine(DummyInferenceEngine):
    def __init__(self, model_id="dummy", delay=0.1):
        super().__init__()
        self.model_id = model_id
        self.delay = delay
        self.tokenizer = MockTokenizer()
        self.call_count = 0
    
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state=None):
        """Enhanced dummy inference with realistic timing and shapes."""
        self.call_count += 1
        
        if DEBUG >= 3:
            print(f"     EnhancedDummyEngine[{self.model_id}] call #{self.call_count}")
            print(f"     Request: {request_id}")
            print(f"     Input shape: {input_data.shape}")
        
        # Simulate inference delay
        await asyncio.sleep(self.delay)
        
        # Create realistic logits output
        batch_size, seq_len = input_data.shape
        vocab_size = 50000
        
        # Generate realistic logits with some patterns
        logits = np.random.randn(batch_size, seq_len, vocab_size) * 0.1
        
        # Add some bias to make certain tokens more likely
        top_tokens = [1, 2, 3, 42, 100, 500, 1000]  # Some "preferred" tokens
        for token in top_tokens:
            if token < vocab_size:
                logits[:, :, token] += np.random.randn() * 2.0
        
        if DEBUG >= 4:
            print(f"     Generated logits shape: {logits.shape}")
            print(f"     Logits stats: mean={logits.mean():.3f}, std={logits.std():.3f}")
        
        return logits, {"step": self.call_count}
    
    async def ensure_shard(self, shard: Shard):
        """Mock shard loading."""
        if DEBUG >= 2:
            print(f"   EnhancedDummyEngine[{self.model_id}] loading shard: {shard.model_id}")
        await asyncio.sleep(0.05)  # Simulate loading time

async def test_speculative_phases(debug_level=1):
    """
    Test all phases of speculative decoding with comprehensive debug output.
    """
    print(f"\n{'='*60}")
    print(f"TESTING SPECULATIVE DECODING PHASES (DEBUG={debug_level})")
    print(f"{'='*60}")
    
    # Create engines with different "models" for realistic testing
    target_engine = EnhancedDummyEngine("llama-3.1-8b", delay=0.2)  # Slower target
    draft_engine = EnhancedDummyEngine("llama-3.2-3b", delay=0.1)   # Faster draft
    
    # Create speculative engine
    spec_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        target_model_id="llama-3.1-8b",
        draft_model_id="llama-3.2-3b",
        gamma=3,  # Small gamma for easier debugging
        temperature=1.0,
        top_k_threshold=0.9,
        lenience=1.0,
        enable_speculative=True
    )
    
    # Create test shard - MAKE SURE IT'S A FINAL LAYER for speculative decoding
    # For is_last_layer() to return True, we need end_layer == n_layers - 1
    shard = Shard(
        model_id="llama-3.1-8b",
        start_layer=24,  # Start near the end
        end_layer=31,    # End at layer 31 (last layer for 32 total layers)
        n_layers=32     # Total 32 layers (0-31)
    )
    
    # Verify this is a final layer
    print(f"   Is final layer: {shard.is_last_layer()}")  # Should be True
    
    # Mock input tokens
    input_tokens = np.array([[1, 2, 3, 4, 5]])  # Shape: (1, 5)
    
    print(f"\n🧪 TEST SETUP:")
    print(f"   Target engine: {target_engine.model_id} (delay: {target_engine.delay}s)")
    print(f"   Draft engine: {draft_engine.model_id} (delay: {draft_engine.delay}s)")
    print(f"   Gamma: {spec_engine.gamma}")
    print(f"   Input tokens: {input_tokens.flatten()}")
    print(f"   Shard: {shard}")
    
    try:
        print(f"\n🚀 Starting speculative inference...")
        result, state = await spec_engine.infer_tensor(
            request_id="test_debug",
            shard=shard,
            input_data=input_tokens,
            inference_state=None
        )
        
        print(f"\n✅ FINAL RESULTS:")
        print(f"   Output shape: {result.shape}")
        print(f"   Output tokens: {result.flatten()}")
        print(f"   State keys: {list(state.keys()) if state else 'None'}")
        
        # Show statistics
        stats = spec_engine.get_speculation_stats()
        print(f"\n📊 SPECULATION STATISTICS:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Show engine call counts
        print(f"\n🔢 ENGINE CALL COUNTS:")
        print(f"   Target engine calls: {target_engine.call_count}")
        print(f"   Draft engine calls: {draft_engine.call_count}")
        
    except Exception as e:
        print(f"\n❌ ERROR during speculative inference: {e}")
        import traceback
        traceback.print_exc()

async def test_compatibility_checking():
    """Test model compatibility checking phase."""
    print(f"\n{'='*60}")
    print(f"TESTING MODEL COMPATIBILITY CHECKING")
    print(f"{'='*60}")
    
    # Test compatible models (same family)
    print(f"\n🧪 Test 1: Compatible models (LLaMA family)")
    target_engine = EnhancedDummyEngine("llama-3.1-8b")
    draft_engine = EnhancedDummyEngine("llama-3.2-3b")
    
    spec_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        target_model_id="llama-3.1-8b",
        draft_model_id="llama-3.2-3b"
    )
    
    target_shard = Shard("llama-3.1-8b", 24, 31, 32)  # Final layer shard
    draft_shard = Shard("llama-3.2-3b", 18, 23, 24)   # Final layer shard for draft model
    
    # Verify these are final layers
    print(f"   Target is final layer: {target_shard.is_last_layer()}")
    print(f"   Draft is final layer: {draft_shard.is_last_layer()}")
    
    compatible = await spec_engine._check_model_compatibility(target_shard, draft_shard)
    print(f"   Result: {'✅ Compatible' if compatible else '❌ Incompatible'}")
    
    # Test incompatible models (different families)
    print(f"\n🧪 Test 2: Incompatible models (different families)")
    target_engine2 = EnhancedDummyEngine("qwen-2.5-7b")
    draft_engine2 = EnhancedDummyEngine("llama-3.2-3b")
    
    # Give them different tokenizers
    draft_engine2.tokenizer = MockTokenizer(vocab_size=60000)  # Different vocab size
    
    spec_engine2 = SpeculativeInferenceEngine(
        target_engine=target_engine2,
        draft_engine=draft_engine2,
        target_model_id="qwen-2.5-7b",
        draft_model_id="llama-3.2-3b"
    )
    
    target_shard2 = Shard("qwen-2.5-7b", 21, 27, 28)   # Final layer shard
    draft_shard2 = Shard("llama-3.2-3b", 18, 23, 24)  # Final layer shard
    
    compatible2 = await spec_engine2._check_model_compatibility(target_shard2, draft_shard2)
    print(f"   Result: {'✅ Compatible' if compatible2 else '❌ Incompatible'}")

async def test_early_exit_mode():
    """Test early exit speculative decoding mode."""
    print(f"\n{'='*60}")
    print(f"TESTING EARLY EXIT SPECULATIVE DECODING")
    print(f"{'='*60}")
    
    target_engine = EnhancedDummyEngine("llama-3.1-8b", delay=0.15)
    
    # No draft engine = early exit mode
    spec_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=None,  # This triggers early exit mode
        target_model_id="llama-3.1-8b",
        gamma=3,
        early_exit_layer=16
    )
    
    shard = Shard("llama-3.1-8b", 24, 31, 32)  # Final layer shard
    input_tokens = np.array([[1, 2, 3]])
    
    # Verify this is a final layer
    print(f"   Is final layer: {shard.is_last_layer()}")
    
    print(f"\n🧪 Early exit test:")
    print(f"   Mode: {spec_engine.use_early_exit}")
    print(f"   Early exit layer: {spec_engine.early_exit_layer}")
    
    result, state = await spec_engine.infer_tensor(
        request_id="test_early_exit",
        shard=shard,
        input_data=input_tokens
    )
    
    print(f"   Result shape: {result.shape}")
    print(f"   Target engine calls: {target_engine.call_count}")

async def run_debug_tests():
    """Run all debug tests with different DEBUG levels."""
    original_debug = os.environ.get('DEBUG', '0')
    
    for debug_level in [1, 2, 3]:
        print(f"\n" + "="*80)
        print(f"RUNNING TESTS WITH DEBUG LEVEL {debug_level}")
        print(f"="*80)
        
        # Set DEBUG level
        os.environ['DEBUG'] = str(debug_level)
        
        # Re-import to pick up new DEBUG level
        import importlib
        import exo.helpers
        importlib.reload(exo.helpers)
        
        # Run tests
        await test_speculative_phases(debug_level)
        await test_compatibility_checking()
        await test_early_exit_mode()
        
        print(f"\n✅ Debug level {debug_level} tests complete!")
    
    # Restore original DEBUG level
    os.environ['DEBUG'] = original_debug

if __name__ == "__main__":
    print("Speculative Decoding Debug Test Suite")
    print("====================================")
    print("This script demonstrates all phases of speculative decoding with detailed debug output.")
    print(f"Current DEBUG level: {DEBUG}")
    
    asyncio.run(run_debug_tests()) 