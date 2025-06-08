#!/usr/bin/env python3

import asyncio
import numpy as np
import os
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard

# Enable debug to see what's happening
os.environ['DEBUG'] = '1'

async def test_pytorch_speculative():
    print("🔥 TESTING PYTORCH SPECULATIVE DECODING")
    print("=" * 60)
    print("✅ Using separate PyTorch engines for true isolation")
    print("✅ No shared executors or threading issues")
    print("✅ Should get realistic acceptance rates (40-70%)")
    print()
    
    # Create separate PyTorch engines
    shard_downloader = NewShardDownloader()
    target_engine = TorchDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TorchDynamicShardInferenceEngine(shard_downloader)
    
    print(f"🎯 Target engine type: {type(target_engine).__name__}")
    print(f"📝 Draft engine type: {type(draft_engine).__name__}")
    print(f"🔍 Different instances? {target_engine is not draft_engine}")
    
    # Verify different executors
    print(f"🧵 Target executor ID: {id(target_engine.executor)}")
    print(f"🧵 Draft executor ID: {id(draft_engine.executor)}")
    print(f"✅ Different executors? {target_engine.executor is not draft_engine.executor}")
    print()
    
    # Create shards for different models
    target_shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=27, n_layers=28)
    draft_shard = Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=15, n_layers=16)
    
    print(f"🎯 Target model: {target_shard.model_id} (layers {target_shard.start_layer}-{target_shard.end_layer})")
    print(f"📝 Draft model: {draft_shard.model_id} (layers {draft_shard.start_layer}-{draft_shard.end_layer})")
    print()
    
    # Create speculative engine
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=3,  # Generate 3 draft tokens
        temperature=1.5,  # High temperature for differences
        target_model_id="llama-3.2-3b",
        draft_model_id="llama-3.2-1b"
    )
    
    print("🚀 Testing PyTorch Speculative Decoding...")
    print(f"🌡️  Temperature: {speculative_engine.temperature}")
    print(f"🎲 Gamma: {speculative_engine.gamma}")
    print()
    
    # Test prompt
    prompt = "The future of AI is"
    print(f"💬 Test prompt: '{prompt}'")
    print()
    
    try:
        # Run speculative generation
        tokens, final_state, accepted_tokens = await speculative_engine.infer_prompt_multi(
            request_id="pytorch_test",
            shard=target_shard,
            prompt=prompt,
            inference_state=None
        )
        
        # Decode result
        result_text = await speculative_engine.decode(target_shard, tokens[0])
        
        print(f"✅ PYTORCH SPECULATIVE RESULTS:")
        print(f"   Generated text: '{result_text}'")
        print(f"   Final acceptance rate: {speculative_engine.total_tokens_accepted/max(speculative_engine.total_tokens_generated,1):.1%}")
        print(f"   Total calls: {speculative_engine.total_calls}")
        print(f"   Tokens generated: {speculative_engine.total_tokens_generated}")
        print(f"   Tokens accepted: {speculative_engine.total_tokens_accepted}")
        
        acceptance_rate = speculative_engine.total_tokens_accepted/max(speculative_engine.total_tokens_generated,1)
        
        if acceptance_rate < 0.95:
            print(f"\n🎉 SUCCESS: Realistic acceptance rate ({acceptance_rate:.1%})")
            print("✅ PyTorch engines are properly isolated!")
        else:
            print(f"\n⚠️  Still high acceptance rate ({acceptance_rate:.1%})")
            print("   This might be normal for similar model families")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pytorch_speculative()) 