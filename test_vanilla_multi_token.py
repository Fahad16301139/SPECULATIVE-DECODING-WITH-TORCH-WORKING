#!/usr/bin/env python3
"""
Test script for TRUE VANILLA Speculative Decoding with multi-token architecture
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
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine 
from exo.download.new_shard_download import NewShardDownloader

# Enable DEBUG for detailed logging
os.environ['DEBUG'] = '2'

async def test_vanilla_multi_token():
    """Test vanilla speculative decoding with multi-token generation"""
    
    print("🚀 Testing TRUE VANILLA Multi-Token Speculative Decoding")
    print("=" * 60)
    
    # Use correct model names from exo's supported models
    target_model_id = "llama-3.2-3b"  # Fixed: removed -instruct
    draft_model_id = "llama-3.2-1b"   # Fixed: removed -instruct
    
    print(f"Target model: {target_model_id}")
    print(f"Draft model: {draft_model_id}")
    
    # Create shard downloader
    shard_downloader = NewShardDownloader()
    
    # Create target and draft engines
    print("\n📥 Creating inference engines...")
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Create speculative engine with multi-token support
    print("🎯 Creating speculative engine...")
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=6,  # Generate 6 draft tokens
        temperature=0.8
    )
    
    # Create shards
    target_shard = Shard(model_id=target_model_id, start_layer=0, end_layer=0, n_layers=28)
    
    # Test prompt
    test_prompt = "The future of artificial intelligence is"
    
    print(f"\n🧪 Testing with prompt: '{test_prompt}'")
    print("=" * 60)
    
    try:
        # Test multi-token generation
        print("🚀 Testing infer_prompt_multi (multi-token generation)...")
        result, inference_state, generated_tokens = await speculative_engine.infer_prompt_multi(
            request_id="test-001",
            shard=target_shard,
            prompt=test_prompt
        )
        
        print(f"✅ Multi-token generation successful!")
        print(f"   Result shape: {result.shape}")
        print(f"   Generated tokens: {len(generated_tokens) if generated_tokens else 0}")
        print(f"   Tokens: {generated_tokens}")
        
        # Decode the result
        if generated_tokens and len(generated_tokens) > 0:
            decoded_text = await speculative_engine.decode(target_shard, np.array(generated_tokens))
            print(f"   Decoded text: '{decoded_text}'")
        
        print("\n🎉 SUCCESS: Multi-token vanilla speculative decoding works!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_vanilla_multi_token())
    if success:
        print("\n✅ All tests passed! Multi-token architecture is working.")
    else:
        print("\n❌ Tests failed.")
        sys.exit(1) 