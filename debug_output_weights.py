#!/usr/bin/env python3
"""
Debug TinyGrad output weight dimensions to understand vocabulary truncation
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
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader

os.environ['DEBUG'] = '1'

async def debug_output_weights():
    """Debug the actual output layer dimensions in TinyGrad models"""
    
    print("🔍 DEBUGGING TINYGRAD OUTPUT LAYER WEIGHTS")
    print("=" * 60)
    
    models_to_test = [
        ("llama-3.2-1b", "1B"),
        ("llama-3.2-3b", "3B"),
    ]
    
    shard_downloader = NewShardDownloader()
    engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    for model_id, size_key in models_to_test:
        print(f"\n🔍 Inspecting {model_id} ({size_key}):")
        
        try:
            # Load the model
            shard = Shard(
                model_id=model_id,
                start_layer=0,
                end_layer=-1,
                n_layers=16 if "1b" in model_id else 28
            )
            
            print(f"  📥 Loading {model_id}...")
            await engine.ensure_shard(shard)
            
            # Inspect the model structure
            model = engine.model
            
            print(f"  📊 Model structure inspection:")
            print(f"     Type: {type(model)}")
            
            # Check if it has tok_embeddings
            if hasattr(model, 'tok_embeddings'):
                emb_weight_shape = model.tok_embeddings.weight.shape
                print(f"     tok_embeddings.weight shape: {emb_weight_shape}")
                print(f"     Vocab size from embeddings: {emb_weight_shape[0]}")
            
            # Check if it has output layer  
            if hasattr(model, 'output'):
                output_weight_shape = model.output.weight.shape
                print(f"     output.weight shape: {output_weight_shape}")
                print(f"     Output vocab size: {output_weight_shape[0]}")
                print(f"     Hidden dim: {output_weight_shape[1]}")
                
                if output_weight_shape[1] == output_weight_shape[0]:
                    print(f"     🚨 PROBLEM: Output matrix is square! ({output_weight_shape[0]}x{output_weight_shape[1]})")
                    print(f"     🚨 Should be ({output_weight_shape[1]}, 128256) for proper vocab!")
                elif output_weight_shape[0] == 128256:
                    print(f"     ✅ Output vocab size is correct: {output_weight_shape[0]}")
                else:
                    print(f"     ⚠️  Unexpected output vocab size: {output_weight_shape[0]}")
            else:
                print(f"     ❌ No output layer found!")
                
            # Test actual inference to see output shape
            print(f"  🧪 Testing inference output shape:")
            test_input = np.array([[1, 2, 3]], dtype=np.int64)  # Simple test input
            
            try:
                result, _ = await engine.infer_tensor("test", shard, test_input)
                print(f"     Inference output shape: {result.shape}")
                
                if len(result.shape) == 3:
                    batch, seq, vocab = result.shape
                    print(f"     Batch: {batch}, Sequence: {seq}, Vocab: {vocab}")
                    
                    if vocab == 3072:
                        print(f"     🚨 CONFIRMED: Vocab dimension is truncated to {vocab}!")
                        print(f"     🚨 This explains the index out of bounds error!")
                    elif vocab == 128256:
                        print(f"     ✅ Vocab dimension is correct: {vocab}")
                    else:
                        print(f"     ⚠️  Unexpected vocab dimension: {vocab}")
                        
            except Exception as e:
                print(f"     ❌ Inference test failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Error loading {model_id}: {e}")
        
        print()
    
    print("=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    asyncio.run(debug_output_weights()) 