#!/usr/bin/env python3
"""
Debug script specifically for LLaMA model alignment issues.
Since you're using LLaMA+LLaMA, let's diagnose the specific misalignment causes.
"""

import asyncio
import numpy as np
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.topology.ring_memory_pool import RingMemoryPool
from exo.inference.shard import Shard

async def diagnose_llama_alignment():
    """
    Diagnose specific LLaMA model alignment issues that could cause
    severe misalignment despite using compatible model families.
    """
    print('🔍 DIAGNOSING LLaMA MODEL ALIGNMENT ISSUES')
    print('=' * 60)
    
    # Test different LLaMA combinations to find the issue
    test_combinations = [
        {
            'name': 'LLaMA 3.2-1B → 3.2-3B (same version)',
            'target': 'llama-3.2-3b',
            'draft': 'llama-3.2-1b', 
            'expected': 'HIGH alignment (same training)'
        },
        {
            'name': 'LLaMA 3.2-1B → 3.1-8B (cross-version)',
            'target': 'llama-3.1-8b',
            'draft': 'llama-3.2-1b',
            'expected': 'MEDIUM alignment (different versions)'
        },
        {
            'name': 'LLaMA 3.1-8B → 3.1-70B (same version)',
            'target': 'llama-3.1-70b', 
            'draft': 'llama-3.1-8b',
            'expected': 'HIGH alignment (same training)'
        }
    ]
    
    for combo in test_combinations:
        print(f"\n🧪 Testing: {combo['name']}")
        print(f"   Expected: {combo['expected']}")
        
        try:
            # Create engines
            target_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
            draft_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
            
            # Create shards
            target_shard = Shard(
                model_id=combo['target'],
                start_layer=0,
                end_layer=27,  # Adjust based on model
                n_layers=28
            )
            
            draft_shard = Shard(
                model_id=combo['draft'],
                start_layer=0, 
                end_layer=15,  # Smaller model
                n_layers=16
            )
            
            print(f"   📊 Target: {combo['target']} ({target_shard.n_layers} layers)")
            print(f"   📊 Draft: {combo['draft']} ({draft_shard.n_layers} layers)")
            
            # Load models
            await target_engine.set_shard(target_shard)
            await draft_engine.set_shard(draft_shard)
            
            # Test specific prompt that often causes issues
            test_prompt = "The quick brown fox"
            
            # Encode prompt
            tokens = await target_engine.encode(target_shard, test_prompt)
            print(f"   🔤 Test tokens: {tokens}")
            
            # Get predictions from both models
            target_out, _ = await target_engine.infer_tensor("test_target", target_shard, tokens, None)
            draft_out, _ = await draft_engine.infer_tensor("test_draft", draft_shard, tokens, None)
            
            # Compare predictions
            target_logits = target_out[0, -1, :] if target_out.ndim == 3 else target_out[0]
            draft_logits = draft_out[0, -1, :] if draft_out.ndim == 3 else draft_out[0]
            
            # Apply same temperature
            temp = 1.0
            target_probs = np.exp(target_logits / temp) / np.sum(np.exp(target_logits / temp))
            draft_probs = np.exp(draft_logits / temp) / np.sum(np.exp(draft_logits / temp))
            
            # Get top tokens from each
            target_top_idx = np.argsort(target_probs)[-10:][::-1]
            draft_top_idx = np.argsort(draft_probs)[-10:][::-1]
            
            print(f"   🎯 Target top token: {target_top_idx[0]} (prob: {target_probs[target_top_idx[0]]:.4f})")
            print(f"   📝 Draft top token: {draft_top_idx[0]} (prob: {draft_probs[draft_top_idx[0]]:.4f})")
            
            # Check alignment
            overlap = len(set(target_top_idx).intersection(set(draft_top_idx)))
            print(f"   🔗 Top-10 overlap: {overlap}/10 tokens")
            
            # Check if top tokens match
            if target_top_idx[0] == draft_top_idx[0]:
                print(f"   ✅ Top tokens MATCH - good alignment")
            else:
                print(f"   ❌ Top tokens DIFFER - potential misalignment")
                
                # Check what target thinks of draft's top choice
                draft_top_token = draft_top_idx[0]
                target_prob_for_draft_token = target_probs[draft_top_token]
                draft_prob_for_draft_token = draft_probs[draft_top_token]
                
                print(f"   🔍 Draft's top choice ({draft_top_token}):")
                print(f"      Draft confidence: {draft_prob_for_draft_token:.6f}")
                print(f"      Target confidence: {target_prob_for_draft_token:.6f}")
                print(f"      Ratio: {target_prob_for_draft_token/draft_prob_for_draft_token:.6f}")
                
                if target_prob_for_draft_token < 0.001:
                    print(f"   🚨 SEVERE MISALIGNMENT: Target assigns <0.1% to draft's choice!")
                elif target_prob_for_draft_token < 0.01:
                    print(f"   ⚠️  MODERATE MISALIGNMENT: Target assigns <1% to draft's choice")
                else:
                    print(f"   🟡 MILD MISALIGNMENT: Acceptable disagreement")
                    
        except Exception as e:
            print(f"   ❌ Failed to test combination: {str(e)[:100]}...")

async def check_llama_model_details():
    """
    Check specific LLaMA model configuration details that could cause issues.
    """
    print('\n🔧 CHECKING LLaMA MODEL CONFIGURATION DETAILS')
    print('=' * 60)
    
    # Common LLaMA misalignment causes
    potential_issues = [
        "🔸 Different chat templates (base vs instruct)",
        "🔸 Different tokenizer versions", 
        "🔸 Different quantization (fp16 vs int8 vs int4)",
        "🔸 Different context lengths (2048 vs 4096 vs 8192)",
        "🔸 Different RoPE scaling",
        "🔸 One model fine-tuned, other base model",
        "🔸 Different attention mechanisms",
        "🔸 Cache size mismatches"
    ]
    
    print("Common LLaMA alignment issues to check:")
    for issue in potential_issues:
        print(f"   {issue}")
    
    print(f"\n💡 SPECIFIC DEBUGGING STEPS:")
    print(f"   1. Check if both models are base models (not instruct/chat)")
    print(f"   2. Verify both use same tokenizer version")
    print(f"   3. Check quantization settings match")
    print(f"   4. Verify context length settings")
    print(f"   5. Test with temperature=0 (greedy) first")
    print(f"   6. Try smaller gamma (1-2) instead of default")

async def test_temperature_sensitivity():
    """
    Test how temperature affects LLaMA alignment.
    LLaMA models can be very sensitive to temperature differences.
    """
    print('\n🌡️  TESTING TEMPERATURE SENSITIVITY')
    print('=' * 60)
    
    temperatures = [0.1, 0.5, 0.7, 1.0, 1.2]
    
    print("LLaMA models are often sensitive to temperature.")
    print("Testing different temperatures to find optimal alignment:")
    
    for temp in temperatures:
        print(f"\n   🌡️  Temperature: {temp}")
        print(f"      Expected behavior:")
        
        if temp < 0.3:
            print(f"      - Very deterministic, high agreement likely")
        elif temp < 0.8:
            print(f"      - Balanced, good for most tasks")
        elif temp < 1.1:
            print(f"      - More creative, possible disagreement")
        else:
            print(f"      - Very creative, high disagreement likely")

if __name__ == "__main__":
    print('🚀 LLaMA Model Alignment Diagnostic')
    print('Identifying why LLaMA+LLaMA is showing severe misalignment\n')
    
    asyncio.run(diagnose_llama_alignment())
    asyncio.run(check_llama_model_details())
    asyncio.run(test_temperature_sensitivity()) 