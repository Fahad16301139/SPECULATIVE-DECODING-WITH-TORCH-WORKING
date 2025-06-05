#!/usr/bin/env python3
"""
Debug shard configuration to see why output layer isn't being applied
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
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine, MODEL_PARAMS
from exo.download.new_shard_download import NewShardDownloader

os.environ['DEBUG'] = '1'

async def debug_shard_config():
    """Debug shard configuration"""
    
    print("🔍 DEBUGGING SHARD CONFIGURATION")
    print("=" * 60)
    
    shard_downloader = NewShardDownloader()
    engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    model_id = "llama-3.2-1b"
    model_size = "1B"
    
    print(f"🧪 Testing {model_id} ({model_size})")
    print("-" * 40)
    
    # Check MODEL_PARAMS
    expected_params = MODEL_PARAMS[model_size]["args"]
    print(f"   Model config:")
    print(f"     n_layers: {expected_params['n_layers']}")
    print(f"     vocab_size: {expected_params['vocab_size']:,}")
    print(f"     dim: {expected_params['dim']:,}")
    
    # Create different shard configurations
    test_shards = [
        ("Single layer 0", Shard(model_id=model_id, start_layer=0, end_layer=0, n_layers=16)),
        ("All layers", Shard(model_id=model_id, start_layer=0, end_layer=15, n_layers=16)),
        ("Last layer only", Shard(model_id=model_id, start_layer=15, end_layer=15, n_layers=16)),
    ]
    
    for desc, shard in test_shards:
        print(f"\n📋 {desc}:")
        print(f"     shard: {shard}")
        print(f"     is_first_layer(): {shard.is_first_layer()}")
        print(f"     is_last_layer(): {shard.is_last_layer()}")
        
        try:
            await engine.ensure_shard(shard)
            
            # Check model properties
            model = engine.model
            print(f"     model type: {type(model).__name__}")
            
            if hasattr(model, 'post'):
                print(f"     model.post function exists: {model.post is not None}")
            
            # Test inference
            test_input = np.array([[1, 2, 3]])
            result, _ = await engine.infer_tensor(f"debug-{desc}", shard, test_input)
            
            print(f"     inference output shape: {result.shape}")
            print(f"     last dimension: {result.shape[-1]:,}")
            
            # Check if we got vocab logits or hidden states
            if result.shape[-1] == expected_params['vocab_size']:
                print(f"     ✅ CORRECT: Full vocabulary logits!")
            elif result.shape[-1] == expected_params['dim']:
                print(f"     ❌ WRONG: Hidden states (not vocab logits)")
            else:
                print(f"     ❓ UNKNOWN: Unexpected dimension")
            
        except Exception as e:
            print(f"     ❌ ERROR: {str(e)[:100]}...")
    
    print(f"\n🔍 ANALYSIS:")
    print("=" * 60)
    print("If 'All layers' doesn't produce vocab logits, then shard")
    print("configuration is wrong. Only last layer should output vocab.")

if __name__ == "__main__":
    asyncio.run(debug_shard_config()) 