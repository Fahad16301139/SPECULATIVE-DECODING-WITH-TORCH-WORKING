#!/usr/bin/env python3
"""
Debug script to check actual vocabulary sizes of llama-3.2 models
"""

import asyncio
import numpy as np
import os
import sys
from pathlib import Path

# Add the exo directory to Python path
exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.helpers import DEBUG
from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine 
from exo.download.new_shard_download import NewShardDownloader

# Enable DEBUG for detailed logging
os.environ['DEBUG'] = '2'

async def debug_vocab_sizes():
    """Debug actual vocabulary sizes of models"""
    
    print("🔍 Debugging Vocabulary Sizes")
    print("=" * 50)
    
    # Create shard downloader
    shard_downloader = NewShardDownloader()
    
    # Create target and draft engines
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Test models
    target_model_id = "llama-3.2-3b"
    draft_model_id = "llama-3.2-1b"
    
    print(f"Target model: {target_model_id}")
    print(f"Draft model: {draft_model_id}")
    
    # Create shards
    target_shard = Shard(model_id=target_model_id, start_layer=0, end_layer=0, n_layers=28)
    draft_shard = Shard(model_id=draft_model_id, start_layer=0, end_layer=0, n_layers=16)
    
    try:
        # Load target model and check vocab size
        print("\n🎯 Loading target model...")
        await target_engine.ensure_shard(target_shard)
        
        # Test with dummy input to see output shape
        dummy_input = np.array([[1, 2, 3, 4, 5]])  # 5 tokens
        target_result, _ = await target_engine.infer_tensor("debug-target", target_shard, dummy_input)
        target_vocab_size = target_result.shape[-1]
        
        print(f"   Target model loaded successfully")
        print(f"   Target logits shape: {target_result.shape}")
        print(f"   Target vocab size: {target_vocab_size}")
        
        # Load draft model and check vocab size
        print("\n📝 Loading draft model...")
        await draft_engine.ensure_shard(draft_shard)
        
        draft_result, _ = await draft_engine.infer_tensor("debug-draft", draft_shard, dummy_input)
        draft_vocab_size = draft_result.shape[-1]
        
        print(f"   Draft model loaded successfully")
        print(f"   Draft logits shape: {draft_result.shape}")
        print(f"   Draft vocab size: {draft_vocab_size}")
        
        # Compare
        print(f"\n📊 COMPARISON:")
        print(f"   Target vocab size: {target_vocab_size}")
        print(f"   Draft vocab size: {draft_vocab_size}")
        print(f"   Compatible: {target_vocab_size == draft_vocab_size}")
        
        if target_vocab_size != draft_vocab_size:
            print(f"\n❌ VOCABULARY SIZE MISMATCH!")
            print(f"   This explains the speculative decoding error")
            print(f"   Vanilla speculative decoding requires identical vocab sizes")
            
            # Check if these match expected sizes from MODEL_PARAMS
            expected_vocab = 128256
            print(f"\n🔧 Expected vocab size (from MODEL_PARAMS): {expected_vocab}")
            print(f"   Target matches expected: {target_vocab_size == expected_vocab}")
            print(f"   Draft matches expected: {draft_vocab_size == expected_vocab}")
            
            # Find compatible model pairs
            print(f"\n💡 SOLUTION:")
            print(f"   We need model pairs with IDENTICAL vocabulary sizes")
            print(f"   Current models have different vocabularies and cannot be used together")
            return False
        else:
            print(f"\n✅ Vocabulary sizes match! Should work for speculative decoding.")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_vocab_sizes())
    if success:
        print("\n✅ Vocabulary sizes are compatible!")
    else:
        print("\n❌ Vocabulary compatibility issues found.")
        sys.exit(1) 