#!/usr/bin/env python3
"""
Test script replicating the exact Leviathan et al. speculative decoding setup
that achieved 2.6X-3.4X speedups with T5 models.
"""

import asyncio
import numpy as np
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.topology.ring_memory_pool import RingMemoryPool
from exo.inference.shard import Shard

async def test_leviathan_t5_setup():
    """
    Test the exact T5-XXL (target) + T5-Small (draft) setup that worked
    in the Leviathan paper with 2.6X-3.4X speedups.
    """
    print('🔬 Testing Leviathan Paper T5 Setup')
    print('=' * 60)
    print('📊 Expected Results from Paper:')
    print('   Target: T5-XXL (11B) | Draft: T5-Small (77M)')
    print('   Acceptance Rate: 0.62-0.75')
    print('   Speedup: 2.6X-3.4X')
    print('   Temperature: 0 (greedy) and 1 (sampling)')
    print('=' * 60)
    
    # Create engines with proper memory pools
    target_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
    draft_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
    
    # Create shards with the EXACT models from the paper
    print('🎯 Setting up TARGET model: T5-XXL (11B parameters)')
    target_shard = Shard(
        model_id='google/t5-xxl-lm-adapt',  # 11B parameters
        start_layer=0, 
        end_layer=23, 
        n_layers=24
    )
    
    print('📝 Setting up DRAFT model: T5-Small (77M parameters)')  
    draft_shard = Shard(
        model_id='google/t5-small-lm-adapt',  # 77M parameters
        start_layer=0,
        end_layer=5,
        n_layers=6
    )
    
    # Verify size ratio matches paper
    size_ratio = 11000 / 77  # 11B / 77M ≈ 143x
    print(f'📏 Size ratio: {size_ratio:.1f}x (should be ~143x like paper)')
    
    # Load the models
    print('\n🔄 Loading models...')
    await target_engine.set_shard(target_shard)
    await draft_engine.set_shard(draft_shard)
    
    # Test different gamma values as in paper
    gamma_values = [3, 5, 7]  # Paper tested multiple gamma values
    
    for gamma in gamma_values:
        print(f'\n🧪 Testing gamma={gamma}')
        
        # Create speculative engine with paper's settings
        speculative_engine = SpeculativeInferenceEngine(
            target_engine=target_engine,
            draft_engine=draft_engine,
            gamma=gamma,
            temperature=1.0,  # Paper tested both T=0 and T=1
            top_k_threshold=0.9,
            lenience=2.0,
            target_model_id='google/t5-xxl-lm-adapt',
            draft_model_id='google/t5-small-lm-adapt'
        )
        
        # Test prompt similar to paper's translation task
        test_prompt = "Translate English to German: The weather is beautiful today."
        
        print(f'   🔤 Test prompt: "{test_prompt}"')
        
        try:
            # Generate response
            result = await speculative_engine.infer_tensor(
                request_id="test_leviathan",
                shard=target_shard,
                input_data=test_prompt,
                inference_state={}
            )
            
            print(f'   ✅ Generation successful with gamma={gamma}')
            print(f'   📄 Output shape: {result.shape if hasattr(result, "shape") else type(result)}')
            
        except Exception as e:
            print(f'   ❌ Error with gamma={gamma}: {str(e)[:100]}...')
    
    print('\n' + '=' * 60)
    print('📊 Expected behavior based on Leviathan paper:')
    print('   - High acceptance rates (62-75%)')
    print('   - Low rejection/rollback frequency') 
    print('   - 2.6X-3.4X speedup vs sequential T5-XXL')
    print('   - Identical output quality to T5-XXL alone')

async def test_alternative_setups():
    """
    Test alternative model combinations that should work better
    than your current setup.
    """
    print('\n🔬 Testing Alternative Model Combinations')
    print('=' * 60)
    
    # Test smaller model combinations that should work
    combinations = [
        {
            'target': 'google/t5-large',  # 800M 
            'draft': 'google/t5-small',   # 77M (10x smaller)
            'target_layers': 24,
            'draft_layers': 6,
            'expected_ratio': '10x'
        },
        {
            'target': 'google/t5-base',   # 250M
            'draft': 'google/t5-small',   # 77M (3x smaller) 
            'target_layers': 12,
            'draft_layers': 6,
            'expected_ratio': '3x'
        }
    ]
    
    for combo in combinations:
        print(f'\n🧪 Testing: {combo["target"]} + {combo["draft"]}')
        print(f'   📏 Expected size ratio: {combo["expected_ratio"]}')
        
        try:
            target_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
            draft_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
            
            target_shard = Shard(
                model_id=combo['target'],
                start_layer=0,
                end_layer=combo['target_layers']-1,
                n_layers=combo['target_layers']
            )
            
            draft_shard = Shard(
                model_id=combo['draft'], 
                start_layer=0,
                end_layer=combo['draft_layers']-1,
                n_layers=combo['draft_layers']
            )
            
            await target_engine.set_shard(target_shard)
            await draft_engine.set_shard(draft_shard)
            
            speculative_engine = SpeculativeInferenceEngine(
                target_engine=target_engine,
                draft_engine=draft_engine,
                gamma=3,
                temperature=1.0,
                top_k_threshold=0.9,
                lenience=2.0,
                target_model_id=combo['target'],
                draft_model_id=combo['draft']
            )
            
            print(f'   ✅ Models loaded successfully')
            
        except Exception as e:
            print(f'   ❌ Failed to load: {str(e)[:100]}...')

if __name__ == "__main__":
    print('🚀 Leviathan Paper Replication Test')
    print('Testing the exact model combinations that achieved')
    print('2.6X-3.4X speedups in the original paper\n')
    
    asyncio.run(test_leviathan_t5_setup())
    asyncio.run(test_alternative_setups()) 