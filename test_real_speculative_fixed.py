#!/usr/bin/env python3
"""
Test REAL speculative decoding with PROPER shard configuration!

The issue was we were using partial shards that don't include the final layer.
For vocab logits, we need full model shards with is_last_layer() = True.
"""

import asyncio
import numpy as np
import os
import sys
from pathlib import Path

# Add exo to path
exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.inference.shard import Shard
from exo.inference.adaptive_speculative_engine import AdaptiveSpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine 
from exo.download.new_shard_download import NewShardDownloader

# Enable DEBUG for detailed logging
os.environ['DEBUG'] = '2'

async def test_real_speculative_fixed():
    """Test real speculative decoding with CORRECT shard configuration"""
    
    print("🚀 TESTING REAL SPECULATIVE DECODING - FIXED!")
    print("=" * 60)
    print("Using FULL MODEL SHARDS with proper layer coverage")
    print("Target: llama-3.2-3b (28 layers) → Full vocab: 128,256")
    print("Draft:  llama-3.2-1b (16 layers) → Full vocab: 128,256")
    print()
    
    # Create shard downloader
    shard_downloader = NewShardDownloader()
    
    # Create target and draft engines
    print("📥 Creating inference engines...")
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Create REAL speculative engine
    print("🔧 Creating REAL speculative engine...")
    speculative_engine = AdaptiveSpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=4,  # Try 4 draft tokens
        temperature=0.8
    )
    
    # CRITICAL FIX: Use FULL model shards covering ALL layers
    target_model_id = "llama-3.2-3b"
    target_shard = Shard(
        model_id=target_model_id, 
        start_layer=0, 
        end_layer=27,  # 3B model has 28 layers (0-27)
        n_layers=28
    )
    
    print(f"🎯 Target shard: {target_shard}")
    print(f"   is_first_layer(): {target_shard.is_first_layer()}")
    print(f"   is_last_layer(): {target_shard.is_last_layer()}")
    
    # Test vocabulary detection with correct config
    print(f"\n🧪 Testing vocabulary detection...")
    await speculative_engine._initialize_vocabulary_mapping(target_shard, target_shard)
    
    # Test simple inference 
    print(f"\n🔥 Testing multi-token inference...")
    test_input = np.array([[1, 2, 3, 4]])  # Simple test sequence
    
    try:
        result, state, tokens = await speculative_engine.infer_tensor_multi(
            "test-real", target_shard, test_input
        )
        
        print(f"✅ REAL speculative decoding completed!")
        print(f"   Result shape: {result.shape}")
        print(f"   Generated tokens: {tokens}")
        print(f"   Number of tokens: {len(tokens)}")
        
        # Get detailed statistics
        stats = speculative_engine.get_real_stats()
        print(f"\n📊 STATISTICS:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_speculative_fixed()) 