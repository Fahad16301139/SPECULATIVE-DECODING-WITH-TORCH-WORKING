#!/usr/bin/env python3
"""
Quick test for PyTorch speculative decoding tensor dimension issue
"""
import os
import asyncio
import numpy as np
import torch

# Set debug level
os.environ['DEBUG'] = '2'

from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.download.new_shard_download import new_shard_downloader
from exo.inference.shard import Shard

async def test_torch_speculative():
    print("🧪 Testing PyTorch Speculative Decoding Logic")
    
    # Setup
    downloader = new_shard_downloader()
    engine = TorchDynamicShardInferenceEngine(downloader)
    
    # Create a shard for llama-3.2-3b
    shard = Shard(
        model_id="llama-3.2-3b",
        start_layer=0,
        end_layer=27,
        n_layers=28
    )
    
    print("\n1️⃣ Setting up engine with shard...")
    await engine.ensure_shard(shard)
    
    print("\n2️⃣ Encoding initial prompt (simulating target engine encode)...")
    prompt = "<|begin_of_text|>Hello"
    encoded_tokens = await engine.encode(shard, prompt)
    print(f"   Encoded tokens shape: {encoded_tokens.shape}")
    
    print("\n3️⃣ Testing token concatenation (simulating speculative verification)...")
    
    # First draft token (sequence length changes: ~6 → 7)
    print("   Adding first draft token...")
    draft_token_1 = np.array([[9906]], dtype=np.int64)  # Shape: (1, 1)
    
    try:
        result1, state1 = await engine.infer_tensor("test", shard, draft_token_1)
        print(f"   ✅ First draft token successful: {result1.shape}")
    except Exception as e:
        print(f"   ❌ First draft token failed: {e}")
        return
    
    # Second draft token (sequence length changes: 7 → 8) 
    print("   Adding second draft token...")
    draft_token_2 = np.array([[0]], dtype=np.int64)  # Shape: (1, 1)
    
    try:
        result2, state2 = await engine.infer_tensor("test", shard, draft_token_2)
        print(f"   ✅ Second draft token successful: {result2.shape}")
    except Exception as e:
        print(f"   ❌ Second draft token failed: {e}")
        return
    
    print("\n🎉 Test completed successfully!")
    print("   Our smart logic is working!")

if __name__ == "__main__":
    asyncio.run(test_torch_speculative()) 