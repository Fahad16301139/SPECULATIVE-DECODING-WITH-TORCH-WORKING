#!/usr/bin/env python3
"""
Debug shard configuration to see why 3B model isn't recognizing last layer
"""

import asyncio
import sys
from pathlib import Path

exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.models import build_base_shard, build_full_shard
from exo.inference.shard import Shard

async def debug_shard_config():
    print("🔍 DEBUGGING SHARD CONFIGURATION")
    print("=" * 50)
    
    models_to_test = [
        "llama-3.2-1b",
        "llama-3.2-3b", 
    ]
    
    for model_id in models_to_test:
        print(f"\n📊 Testing {model_id}:")
        
        # Get full shard (complete model) 
        print("  🔧 Testing build_base_shard (BROKEN):")
        base_shard = build_base_shard(model_id, "TinygradDynamicShardInferenceEngine")
        if base_shard:
            print(f"    Shard: {base_shard}")
            print(f"    is_last_layer(): {base_shard.is_last_layer()} ❌")
        
        print("  ✅ Testing build_full_shard (FIXED):")
        shard = build_full_shard(model_id, "TinygradDynamicShardInferenceEngine")
        
        if shard:
            print(f"    Shard: {shard}")
            print(f"    start_layer: {shard.start_layer}")
            print(f"    end_layer: {shard.end_layer}")
            print(f"    n_layers: {shard.n_layers}")
            print(f"    is_first_layer(): {shard.is_first_layer()}")
            print(f"    is_last_layer(): {shard.is_last_layer()}")
            
            # Check layer ranges
            if shard.is_last_layer():
                print("    ✅ Correctly identified as last layer - FIXED!")
            else:
                print("    ❌ Still NOT identified as last layer!")
                print(f"       Expected: end_layer({shard.end_layer}) == n_layers-1({shard.n_layers-1})")
        else:
            print(f"    ❌ Failed to create full shard")

if __name__ == "__main__":
    asyncio.run(debug_shard_config()) 