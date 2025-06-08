#!/usr/bin/env python3

import asyncio
import numpy as np
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.inference.shard import Shard

async def test_balanced_speculative():
    print("🧪 Testing BALANCED Speculative Decoding Parameters")
    print("🎯 Goal: Better output quality with reasonable acceptance rate")
    print("=" * 80)
    
    # Create target and draft engines
    target_engine = TorchDynamicShardInferenceEngine()
    draft_engine = TorchDynamicShardInferenceEngine()
    
    # Create speculative engine with BALANCED parameters
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=3,  # Reduced from 8 to 3 draft tokens
        temperature=0.7,
        top_k_threshold=0.9,
        lenience=2.0,  # Much reduced from 10.0
        target_model_id="llama-3.2-3b",  # 3B model
        draft_model_id="llama-3.2-1b"    # 1B model
    )
    
    print(f"📋 BALANCED PARAMETERS:")
    print(f"   🎲 Gamma (draft tokens): {speculative_engine.gamma}")
    print(f"   🌡️  Temperature: {speculative_engine.temperature}")
    print(f"   🎯 Lenience factor: {speculative_engine.lenience}")
    print(f"   📊 Top-K threshold: {speculative_engine.top_k_threshold}")
    print(f"   🎯 Target model: {speculative_engine.target_model_id}")
    print(f"   📝 Draft model: {speculative_engine.draft_model_id}")
    print()
    
    # Create shard for testing  
    shard = Shard(
        model_id="llama-3.2-3b",
        start_layer=0,
        end_layer=27,
        n_layers=28
    )
    
    # Test prompt: Simple greeting
    test_prompt = "Hello, how are you?"
    print(f"🗣️  Test prompt: '{test_prompt}'")
    print()
    
    try:
        # Encode prompt
        input_tokens = await speculative_engine.encode(shard, test_prompt)
        print(f"📥 Encoded input shape: {input_tokens.shape}")
        print(f"📥 Input tokens: {input_tokens.tolist()}")
        print()
        
        # Generate 3 rounds of speculative decoding to see pattern
        current_tokens = input_tokens.copy()
        all_generated_tokens = []
        
        for round_num in range(3):
            print(f"🚀 ROUND {round_num + 1}/3: Generating with balanced parameters")
            print("-" * 60)
            
            # Generate tokens
            output_tokens, _, _ = await speculative_engine.infer_tensor_multi(
                f"test_balanced_round_{round_num}",
                shard,
                current_tokens,
                None
            )
            
            # Get newly generated tokens
            new_tokens = output_tokens[:, current_tokens.shape[1]:]
            all_generated_tokens.extend(new_tokens[0].tolist())
            
            # Decode current response
            current_response = await speculative_engine.decode(shard, output_tokens)
            print(f"🗨️  Current response: '{current_response}'")
            print()
            
            # Update current tokens for next round
            current_tokens = output_tokens
            
            # Check if we got reasonable output (not just gibberish)
            words = current_response.split()
            if len(words) >= 5:
                print(f"✅ Generated enough words ({len(words)}), stopping test")
                break
        
        print("=" * 80)
        print(f"📊 FINAL RESULTS:")
        final_response = await speculative_engine.decode(shard, current_tokens)
        print(f"🗨️  Final response: '{final_response}'")
        print(f"🔢 Total generated tokens: {len(all_generated_tokens)}")
        print(f"📈 Average acceptance rate: {speculative_engine.total_tokens_accepted/max(speculative_engine.total_tokens_generated,1):.1%}")
        print(f"⏱️  Total calls: {speculative_engine.total_calls}")
        
        # Quality assessment
        response_words = final_response.split()
        quality_score = "GOOD" if len(response_words) >= 3 and not any(word.count("'") > 2 for word in response_words) else "POOR"
        print(f"🎯 Quality assessment: {quality_score}")
        
        if quality_score == "GOOD":
            print("✅ BALANCED PARAMETERS SUCCESS: Better output quality achieved!")
        else:
            print("❌ Still need more tuning - output quality needs improvement")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_balanced_speculative()) 