#!/usr/bin/env python3

import asyncio
import numpy as np
import os
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard

# Enable debug
os.environ['DEBUG'] = '1'

async def test_simple_speculative():
    print("🧪 SIMPLE SPECULATIVE TEST WITH FORCED DIFFERENCES")
    print("=" * 60)
    print("Testing if we can get realistic acceptance rates...")
    print()
    
    # Create engines
    shard_downloader = NewShardDownloader()
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Create speculative engine with HIGHER temperature to increase randomness
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=3,  # Smaller gamma for easier testing
        temperature=2.0,  # HIGH temperature to increase randomness
        target_model_id="llama-3.2-3b",
        draft_model_id="llama-3.2-1b"
    )
    
    # Create target shard
    target_shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=27, n_layers=28)
    
    # Test with a short, simple prompt
    prompt = "Hi"
    
    print(f"🚀 Testing with prompt: '{prompt}'")
    print(f"🌡️  Using HIGH temperature: {speculative_engine.temperature}")
    print(f"🎲 Using gamma: {speculative_engine.gamma}")
    print("-" * 40)
    
    try:
        # Run speculative decoding
        result, state, generated_tokens = await speculative_engine.infer_prompt_multi(
            "simple-test",
            target_shard,
            prompt
        )
        
        print(f"\n✅ Generation completed!")
        print(f"Generated tokens: {generated_tokens}")
        
        if generated_tokens:
            decoded = await speculative_engine.decode(target_shard, np.array(generated_tokens))
            print(f"Generated text: '{decoded}'")
            print(f"Full result: '{prompt} {decoded}'")
        
        # Check final stats
        acceptance_rate = speculative_engine.total_tokens_accepted / max(speculative_engine.total_tokens_generated, 1)
        print(f"\n📊 RESULTS:")
        print(f"   Final acceptance rate: {acceptance_rate:.1%}")
        print(f"   Total calls: {speculative_engine.total_calls}")
        print(f"   Tokens generated: {speculative_engine.total_tokens_generated}")
        print(f"   Tokens accepted: {speculative_engine.total_tokens_accepted}")
        
        # Analysis
        if acceptance_rate >= 0.95:
            print(f"\n🚨 STILL TOO HIGH: {acceptance_rate:.1%}")
            print("   Possible remaining issues:")
            print("   - Models are just naturally very similar")
            print("   - Temperature not working properly")
            print("   - Algorithm bug still exists")
        elif acceptance_rate >= 0.6:
            print(f"\n✅ REALISTIC: {acceptance_rate:.1%} is a good acceptance rate!")
            print("   This suggests speculative decoding is working correctly.")
        elif acceptance_rate >= 0.3:
            print(f"\n⚠️  LOW: {acceptance_rate:.1%} but at least it's rejecting some tokens!")
            print("   This proves the algorithm can reject, just maybe too aggressively.")
        else:
            print(f"\n🔴 VERY LOW: {acceptance_rate:.1%}")
            print("   Models might be too different or algorithm too strict.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_speculative()) 