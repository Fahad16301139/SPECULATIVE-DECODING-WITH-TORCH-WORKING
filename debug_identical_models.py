#!/usr/bin/env python3

import numpy as np
import asyncio
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard

async def debug_model_behavior():
    """
    Investigate why models sometimes behave identically in speculative decoding.
    """
    print("🔍 DEBUGGING IDENTICAL MODEL BEHAVIOR")
    print("=" * 60)
    
    # Create shard downloader
    shard_downloader = NewShardDownloader()
    
    # Create target and draft engines separately 
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Create shards
    target_shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=27, n_layers=28)
    draft_shard = Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=15, n_layers=16)
    
    print(f"Target shard: {target_shard}")
    print(f"Draft shard: {draft_shard}")
    
    # Test inputs - use the same sequence that showed identical behavior
    test_inputs = [
        np.array([[40, 2846, 1120, 264, 4221]]),  # From debug output
        np.array([[128000, 9906, 1917]]),         # "Hello world"
        np.array([[1, 2, 3, 4, 5]]),              # Simple sequence
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n🧪 TEST {i+1}: Input shape {test_input.shape}")
        print(f"   Input tokens: {test_input[0].tolist()}")
        
        try:
            # Get target model output
            print(f"   🎯 Getting target output...")
            target_output, _ = await target_engine.infer_tensor(f"test_target_{i}", target_shard, test_input)
            
            # Get draft model output  
            print(f"   📝 Getting draft output...")
            draft_output, _ = await draft_engine.infer_tensor(f"test_draft_{i}", draft_shard, test_input)
            
            print(f"   🎯 Target output shape: {target_output.shape}")
            print(f"   📝 Draft output shape: {draft_output.shape}")
            
            if target_output.shape != draft_output.shape:
                print(f"   ❌ SHAPE MISMATCH!")
                continue
                
            # Compare the outputs
            diff = np.abs(target_output - draft_output)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            print(f"   📊 Mean absolute difference: {mean_diff:.6f}")
            print(f"   📊 Max absolute difference: {max_diff:.6f}")
            
            # Check if outputs are suspiciously similar
            if mean_diff < 0.01:
                print(f"   🚨 SUSPICIOUS: Models outputs are too similar!")
                print(f"   🔍 Target sample: {target_output[0, -1, :5]}")
                print(f"   🔍 Draft sample: {draft_output[0, -1, :5]}")
            
            # Test probability calculation at a specific position
            if target_output.shape[1] >= 1:
                target_logits = target_output[0, -1, :]  # Last position
                draft_logits = draft_output[0, -1, :]
                
                # Apply softmax
                target_probs = np.exp(target_logits - np.max(target_logits))
                target_probs = target_probs / np.sum(target_probs)
                
                draft_probs = np.exp(draft_logits - np.max(draft_logits))
                draft_probs = draft_probs / np.sum(draft_probs)
                
                # Check top tokens
                target_top_tokens = np.argsort(target_probs)[-5:][::-1]
                draft_top_tokens = np.argsort(draft_probs)[-5:][::-1]
                
                print(f"   🎯 Target top 5 tokens: {target_top_tokens}")
                print(f"   📝 Draft top 5 tokens: {draft_top_tokens}")
                
                # Calculate acceptance ratios for top tokens
                print(f"   ⚖️  ACCEPTANCE ANALYSIS:")
                for j, token in enumerate(target_top_tokens):
                    target_prob = target_probs[token]
                    draft_prob = draft_probs[token]
                    if draft_prob > 0:
                        ratio = target_prob / draft_prob
                        acceptance_prob = min(1.0, ratio)
                        print(f"      Token {token}: target={target_prob:.6f}, draft={draft_prob:.6f}, ratio={ratio:.6f}, accept_prob={acceptance_prob:.6f}")
                        
                        if acceptance_prob > 0.99:
                            print(f"         🚨 SUSPICIOUS: Near 100% acceptance for token {token}!")
            
        except Exception as e:
            print(f"   ❌ Error in test {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🔍 MEMORY AND OBJECT INVESTIGATION:")
    print(f"   🎯 Target engine id: {id(target_engine)}")
    print(f"   📝 Draft engine id: {id(draft_engine)}")
    print(f"   🎯 Target engine class: {target_engine.__class__}")
    print(f"   📝 Draft engine class: {draft_engine.__class__}")
    
    # Check if engines have the same model loaded
    if hasattr(target_engine, 'model') and hasattr(draft_engine, 'model'):
        print(f"   🎯 Target model id: {id(target_engine.model) if target_engine.model else 'None'}")
        print(f"   📝 Draft model id: {id(draft_engine.model) if draft_engine.model else 'None'}")
        
        if target_engine.model is not None and draft_engine.model is not None:
            if id(target_engine.model) == id(draft_engine.model):
                print(f"   🚨 CRITICAL: Both engines share the same model object!")
            else:
                print(f"   ✅ Models are different objects")
    
    # Check shards
    if hasattr(target_engine, 'shard') and hasattr(draft_engine, 'shard'):
        print(f"   🎯 Target shard: {target_engine.shard}")
        print(f"   📝 Draft shard: {draft_engine.shard}")
        
        if target_engine.shard == draft_engine.shard:
            print(f"   🚨 CRITICAL: Both engines have the same shard!")
        else:
            print(f"   ✅ Shards are different")
            
    print(f"\n🏁 Investigation complete!")

if __name__ == "__main__":
    asyncio.run(debug_model_behavior()) 