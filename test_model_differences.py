#!/usr/bin/env python3

import asyncio
import numpy as np
import os
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard

# Reduce debug noise
os.environ['DEBUG'] = '0'

async def test_model_differences():
    print("🔍 TESTING IF 1B AND 3B MODELS ARE ACTUALLY DIFFERENT")
    print("=" * 60)
    
    shard_downloader = NewShardDownloader()
    
    # Create two separate engines
    engine_1b = TinygradDynamicShardInferenceEngine(shard_downloader)
    engine_3b = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Create shards for different models
    shard_1b = Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=15, n_layers=16)
    shard_3b = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=27, n_layers=28)
    
    print("📥 Loading models...")
    
    # Load the 1B model
    print("   Loading 1B model...")
    await engine_1b.ensure_shard(shard_1b)
    
    # Load the 3B model  
    print("   Loading 3B model...")
    await engine_3b.ensure_shard(shard_3b)
    
    print("✅ Both models loaded!")
    
    # Test identical inputs
    test_inputs = [
        np.array([[128000, 9906, 1917]]),  # "Hello world"
        np.array([[128000, 791, 3938, 315, 15592]]),  # "The future of AI"
        np.array([[1, 2, 3, 4, 5]]),  # Simple sequence
    ]
    
    print("\n🧪 TESTING MODEL OUTPUTS:")
    print("-" * 40)
    
    all_identical = True
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n🔬 Test {i+1}: {test_input.tolist()}")
        
        try:
            # Get outputs from both models
            output_1b, _ = await engine_1b.infer_tensor(f"test-1b-{i}", shard_1b, test_input)
            output_3b, _ = await engine_3b.infer_tensor(f"test-3b-{i}", shard_3b, test_input)
            
            print(f"   1B output shape: {output_1b.shape}")
            print(f"   3B output shape: {output_3b.shape}")
            
            # Check if vocab sizes match
            if output_1b.shape[-1] != output_3b.shape[-1]:
                print(f"   ❌ Different vocab sizes: {output_1b.shape[-1]} vs {output_3b.shape[-1]}")
                all_identical = False
                continue
                
            # Compare a subset of logits for the last position
            last_1b = output_1b[0, -1, :]  # Last position logits
            last_3b = output_3b[0, -1, :]  # Last position logits
            
            # Sample a few key positions
            sample_positions = [0, 1, 100, 1000, 10000, -1]
            differences = []
            
            for pos in sample_positions:
                if pos < len(last_1b) and pos < len(last_3b):
                    diff = abs(float(last_1b[pos]) - float(last_3b[pos]))
                    differences.append(diff)
                    print(f"   Position {pos}: 1B={last_1b[pos]:.4f}, 3B={last_3b[pos]:.4f}, diff={diff:.4f}")
            
            avg_diff = np.mean(differences)
            max_diff = np.max(differences)
            
            print(f"   📊 Average difference: {avg_diff:.6f}")
            print(f"   📊 Maximum difference: {max_diff:.6f}")
            
            if avg_diff < 1e-6:
                print(f"   🚨 IDENTICAL: Models produce nearly identical outputs!")
                all_identical = False
            elif avg_diff < 0.1:
                print(f"   ⚠️  VERY SIMILAR: Small differences - models might be too similar")
            else:
                print(f"   ✅ DIFFERENT: Models produce clearly different outputs")
                all_identical = False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
    print(f"\n🏁 FINAL ASSESSMENT:")
    print("=" * 40)
    
    if all_identical:
        print("🚨 PROBLEM: Models are producing identical or nearly identical outputs!")
        print("   This explains the 100% acceptance rates in speculative decoding.")
        print("   Possible causes:")
        print("   - Both models downloading the same weights")
        print("   - TinyGrad model loading bug")
        print("   - Incorrect model configuration")
    else:
        print("✅ GOOD: Models produce different outputs!")
        print("   The 100% acceptance rate issue must be elsewhere in the algorithm.")
        
    # Additional diagnostics
    print(f"\n🔍 ADDITIONAL INFO:")
    print(f"   1B model shard: {engine_1b.shard}")
    print(f"   3B model shard: {engine_3b.shard}")
    print(f"   Same shard objects? {engine_1b.shard is engine_3b.shard}")
    print(f"   Same model objects? {engine_1b.model is engine_3b.model}")

if __name__ == "__main__":
    asyncio.run(test_model_differences()) 