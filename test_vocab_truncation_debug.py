#!/usr/bin/env python3
"""
Debug TinyGrad vocabulary truncation issue

The problem: TinyGrad models output logits with vocab dimension matching
the hidden dimension (2048 for 1B, 3072 for 3B) instead of the true 
vocabulary size (128,256). This breaks speculative decoding.
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

async def debug_vocab_truncation():
    """Debug why TinyGrad is truncating vocabulary sizes"""
    
    print("🔍 DEBUGGING TINYGRAD VOCABULARY TRUNCATION")
    print("=" * 60)
    print("Expected: 128,256 vocab size for all LLaMA models")
    print("Actual:   2,048 (1B) and 3,072 (3B) - matches hidden dims!")
    print()
    
    # Test models
    models_to_test = [
        ("llama-3.2-1b", "1B"),
        ("llama-3.2-3b", "3B"),
    ]
    
    shard_downloader = NewShardDownloader()
    engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    for model_id, model_size in models_to_test:
        print(f"🧪 Testing {model_id} ({model_size})")
        print("-" * 40)
        
        # Check MODEL_PARAMS
        expected_params = MODEL_PARAMS[model_size]["args"]
        print(f"   Expected vocab_size: {expected_params['vocab_size']:,}")
        print(f"   Expected dim:        {expected_params['dim']:,}")
        print(f"   Expected hidden_dim: {expected_params['hidden_dim']:,}")
        
        # Load model and check actual sizes
        shard = Shard(model_id=model_id, start_layer=0, end_layer=0, n_layers=8)
        await engine.ensure_shard(shard)
        
        # Check tokenizer
        print(f"   Tokenizer vocab:     {engine.tokenizer.vocab_size:,}")
        
        # Check model architecture
        model = engine.model
        if hasattr(model, 'output'):
            print(f"   Model output layer:  {model.output}")
            if hasattr(model.output, 'weight'):
                weight_shape = model.output.weight.shape
                print(f"   Output weight shape: {weight_shape}")
                print(f"   Actual vocab size:   {weight_shape[0]:,}")
        
        # Test actual inference
        test_input = np.array([[1, 2, 3]])
        result, _ = await engine.infer_tensor("debug-test", shard, test_input)
        actual_vocab = result.shape[-1]
        
        print(f"   Inference output:    {result.shape}")
        print(f"   Actual vocab output: {actual_vocab:,}")
        
        # Analysis
        if actual_vocab == expected_params['vocab_size']:
            print("   ✅ CORRECT: Full vocabulary preserved")
        elif actual_vocab == expected_params['dim']:
            print("   ❌ BUG: Vocab size equals hidden dimension!")
            print("       🔧 The output layer is using 'dim' instead of 'vocab_size'")
        else:
            print(f"   ❓ UNKNOWN: Unexpected vocab size {actual_vocab}")
        
        print()
    
    print("🔍 INVESTIGATION RESULTS:")
    print("=" * 60)
    print("If actual vocab == dim, then TinyGrad model initialization")
    print("is incorrectly using the hidden dimension instead of vocab_size.")
    print("This explains the truncation and breaks speculative decoding!")

if __name__ == "__main__":
    asyncio.run(debug_vocab_truncation()) 