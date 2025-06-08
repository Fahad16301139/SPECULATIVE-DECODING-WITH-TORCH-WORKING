#!/usr/bin/env python3
"""
PyTorch-specific speculative decoding debug script.
Since vocab sizes are compatible (both 128,000), the issue is likely
in PyTorch cache management or tensor handling.
"""

import asyncio
import numpy as np
import torch
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.topology.ring_memory_pool import RingMemoryPool
from exo.inference.shard import Shard

async def debug_pytorch_speculative():
    """Debug PyTorch-specific speculative decoding issues"""
    
    print('🔧 PYTORCH SPECULATIVE DECODING DEBUG')
    print('=' * 60)
    print('✅ Vocab compatibility confirmed: both models have 128,000 tokens')
    print('🎯 Diagnosing PyTorch-specific cache/tensor issues')
    print()
    
    # Create PyTorch engines
    target_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
    draft_engine = TorchDynamicShardInferenceEngine(memory_pool=RingMemoryPool())
    
    # Create shards for compatible models
    target_shard = Shard(
        model_id='llama-3.2-3b',
        start_layer=0,
        end_layer=27,
        n_layers=28
    )
    
    draft_shard = Shard(
        model_id='llama-3.2-1b', 
        start_layer=0,
        end_layer=15,
        n_layers=16
    )
    
    print(f'🎯 Target: {target_shard.model_id} ({target_shard.n_layers} layers)')
    print(f'📝 Draft: {draft_shard.model_id} ({draft_shard.n_layers} layers)')
    print()
    
    # Load models
    print('📥 Loading models...')
    await target_engine.ensure_shard(target_shard)
    await draft_engine.ensure_shard(draft_shard)
    
    # Verify tokenizers are identical
    print('🔍 Verifying tokenizer compatibility:')
    target_vocab = target_engine.tokenizer.vocab_size
    draft_vocab = draft_engine.tokenizer.vocab_size
    print(f'   Target vocab: {target_vocab:,}')
    print(f'   Draft vocab: {draft_vocab:,}')
    
    if target_vocab != draft_vocab:
        print('❌ VOCAB MISMATCH - this should not happen!')
        return
    else:
        print('✅ Vocabularies match perfectly')
    print()
    
    # Test with simple prompt
    test_prompt = "The quick brown fox"
    print(f'🧪 Testing with prompt: "{test_prompt}"')
    
    # Encode with target engine
    tokens = await target_engine.encode(target_shard, test_prompt)
    print(f'📝 Encoded tokens: {tokens}')
    print(f'📊 Token shape: {tokens.shape}')
    print()
    
    # Test individual model predictions
    print('🔍 Testing individual model predictions:')
    
    # Target model prediction
    print('🎯 Target model prediction:')
    target_out, target_state = await target_engine.infer_tensor(
        'test_target', target_shard, tokens, None
    )
    print(f'   Output shape: {target_out.shape}')
    
    # Extract logits
    if target_out.ndim == 3:
        target_logits = target_out[0, -1, :]
    else:
        target_logits = target_out[0]
    
    print(f'   Logits shape: {target_logits.shape}')
    print(f'   Logits range: [{target_logits.min():.3f}, {target_logits.max():.3f}]')
    
    # Apply softmax
    target_probs = torch.softmax(torch.from_numpy(target_logits), dim=-1).numpy()
    target_top_tokens = np.argsort(target_probs)[-10:][::-1]
    print(f'   Top token: {target_top_tokens[0]} (prob: {target_probs[target_top_tokens[0]]:.6f})')
    print()
    
    # Draft model prediction  
    print('📝 Draft model prediction:')
    draft_out, draft_state = await draft_engine.infer_tensor(
        'test_draft', draft_shard, tokens, None
    )
    print(f'   Output shape: {draft_out.shape}')
    
    # Extract logits
    if draft_out.ndim == 3:
        draft_logits = draft_out[0, -1, :]
    else:
        draft_logits = draft_out[0]
    
    print(f'   Logits shape: {draft_logits.shape}')
    print(f'   Logits range: [{draft_logits.min():.3f}, {draft_logits.max():.3f}]')
    
    # Apply softmax
    draft_probs = torch.softmax(torch.from_numpy(draft_logits), dim=-1).numpy()
    draft_top_tokens = np.argsort(draft_probs)[-10:][::-1]
    print(f'   Top token: {draft_top_tokens[0]} (prob: {draft_probs[draft_top_tokens[0]]:.6f})')
    print()
    
    # Compare predictions
    print('🔍 Prediction comparison:')
    print(f'   Target top token: {target_top_tokens[0]}')
    print(f'   Draft top token: {draft_top_tokens[0]}')
    
    if target_top_tokens[0] == draft_top_tokens[0]:
        print('✅ Top tokens MATCH - good alignment!')
    else:
        print('❌ Top tokens DIFFER - this explains the misalignment')
        
        # Check if draft top token is in target top-10
        if draft_top_tokens[0] in target_top_tokens:
            rank = np.where(target_top_tokens == draft_top_tokens[0])[0][0] + 1
            print(f'   Draft top token ranks #{rank} in target top-10')
        else:
            print('   Draft top token not in target top-10 - severe misalignment')
    
    # Check overlap in top tokens
    overlap = len(set(target_top_tokens).intersection(set(draft_top_tokens)))
    print(f'   Top-10 overlap: {overlap}/10 tokens')
    print()
    
    # Test speculative decoding
    print('🚀 Testing actual speculative decoding:')
    speculative_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=2,  # Conservative gamma
        temperature=0.7,
        target_model_id='llama-3.2-3b',
        draft_model_id='llama-3.2-1b'
    )
    
    try:
        result, final_state, accepted_tokens = await speculative_engine.infer_tensor_multi(
            'test_speculative', target_shard, tokens, None
        )
        
        print(f'✅ Speculative decoding completed')
        print(f'   Result shape: {result.shape}')
        print(f'   Accepted tokens: {accepted_tokens}')
        print(f'   Acceptance rate: {len(accepted_tokens)/2:.1%}')
        
    except Exception as e:
        print(f'❌ Speculative decoding failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(debug_pytorch_speculative()) 