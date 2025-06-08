import asyncio
import numpy as np
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.topology.ring_memory_pool import RingMemoryPool
from exo.inference.shard import Shard

async def test_fixed_speculative():
    print('🔧 Testing Fixed Speculative Decoding')
    print('=' * 50)
    
    # Create engines
    target_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
    draft_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
    
    # Create shards
    target_shard = Shard(model_id='llama-3.2-3b', start_layer=0, end_layer=27, n_layers=28)
    
    # Create speculative engine with fixed algorithm
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=2,  # Conservative
        temperature=0.7,
        target_model_id='llama-3.2-3b',
        draft_model_id='llama-3.2-1b'
    )
    
    # Test prompt
    prompt = 'The capital of Bangladesh is'
    
    try:
        print(f'🧪 Testing prompt: {prompt}')
        
        # Run fixed speculative decoding  
        result, state, generated_tokens = await speculative_engine.infer_prompt_multi(
            'test-fixed',
            target_shard,
            prompt
        )
        
        print(f'✅ Generation completed!')
        print(f'Generated tokens: {generated_tokens}')
        
        if generated_tokens:
            # Convert result back to tokens for decoding
            if isinstance(result, np.ndarray) and result.ndim == 2:
                all_tokens = result[0].astype(np.int64)
                decoded = await speculative_engine.decode(target_shard, all_tokens)
                print(f'Full result: {decoded}')
                
                # Extract just the generated part
                prompt_tokens = await speculative_engine.encode(target_shard, prompt)
                generated_part = all_tokens[len(prompt_tokens[0]):]
                if len(generated_part) > 0:
                    generated_text = await speculative_engine.decode(target_shard, generated_part.reshape(1, -1))
                    print(f'Generated text: {generated_text}')
        
        # Check stats
        acceptance_rate = speculative_engine.total_tokens_accepted / max(speculative_engine.total_tokens_generated, 1)
        print(f'Acceptance rate: {acceptance_rate:.1%}')
        
        if acceptance_rate >= 0.8:
            print('🚨 Still very high acceptance - may need more model differentiation')
        elif acceptance_rate >= 0.3:
            print('✅ Good acceptance rate - algorithm working properly!')
        else:
            print('📉 Low acceptance rate - models very different (this is actually good!)')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_speculative()) 