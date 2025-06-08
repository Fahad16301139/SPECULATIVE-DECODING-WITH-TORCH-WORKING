#!/usr/bin/env python3

import asyncio
import numpy as np
import os
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard

# Enable debug to see acceptance rates
os.environ['DEBUG'] = '1'

async def test_fixed_speculative():
    print("🧪 TESTING FIXED SPECULATIVE DECODING")
    print("=" * 60)
    print("After fixing executor sharing, we should see:")
    print("✅ Realistic acceptance rates (40-70%, NOT 100%)")
    print("✅ Some tokens rejected (proving models are different)")
    print()
    
    # Create separate engines
    shard_downloader = NewShardDownloader()
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Verify separate executors
    print(f"Target executor: {id(target_engine.executor)}")
    print(f"Draft executor: {id(draft_engine.executor)}")
    print(f"Same executor? {target_engine.executor is draft_engine.executor}")
    print()
    
    # Create speculative engine
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=5,
        temperature=1.0,  # Use temperature=1.0 for realistic sampling
        target_model_id="llama-3.2-3b",
        draft_model_id="llama-3.2-1b"
    )
    
    # Create target shard
    target_shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=27, n_layers=28)
    
    # Test prompt
    prompt = "The future of AI is"
    
    print(f"🚀 Testing with prompt: '{prompt}'")
    print("-" * 40)
    
    try:
        # Run speculative decoding
        result, state, generated_tokens = await speculative_engine.infer_prompt_multi(
            "test-fixed",
            target_shard,
            prompt
        )
        
        print(f"\n✅ Generation completed!")
        print(f"Generated tokens: {generated_tokens}")
        
        if generated_tokens:
            decoded = await speculative_engine.decode(target_shard, np.array(generated_tokens))
            print(f"Generated text: '{decoded}'")
            print(f"Full result: '{prompt} {decoded}'")
        
        # Check stats
        acceptance_rate = speculative_engine.total_tokens_accepted / max(speculative_engine.total_tokens_generated, 1)
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Acceptance rate: {acceptance_rate:.1%}")
        print(f"   Total calls: {speculative_engine.total_calls}")
        print(f"   Tokens generated: {speculative_engine.total_tokens_generated}")
        print(f"   Tokens accepted: {speculative_engine.total_tokens_accepted}")
        
        # Analysis
        if acceptance_rate >= 0.95:
            print(f"\n🚨 STILL TOO HIGH: {acceptance_rate:.1%} suggests models are still too similar!")
            print("   Possible causes:")
            print("   - Models trained on same data")
            print("   - Architecture too similar")
            print("   - TinyGrad state still interfering")
        elif acceptance_rate >= 0.4:
            print(f"\n✅ GOOD: {acceptance_rate:.1%} is realistic for speculative decoding!")
            print("   The fix worked - models are now properly different!")
        else:
            print(f"\n⚠️  LOW: {acceptance_rate:.1%} suggests models may be too different")
            print("   This is actually better than 100% - shows real algorithm!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_speculative()) 