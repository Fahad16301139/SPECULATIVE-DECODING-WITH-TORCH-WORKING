#!/usr/bin/env python3
"""
Debug script to find the root cause of severe misalignment between LLaMA 3.2 models.
This shouldn't happen - let's find what's actually wrong.
"""

import asyncio
import numpy as np
import os

# Set debug level
os.environ['DEBUG'] = '2'

async def debug_model_mismatch():
    """Find why LLaMA 3.2 models are severely misaligned"""
    
    print('🔍 DEBUGGING SEVERE MODEL MISALIGNMENT')
    print('=' * 60)
    print('❓ Question: Why are LLaMA 3.2-1B and 3.2-3B so misaligned?')
    print('📊 Expected: Reasonable alignment (both same family)')  
    print('🚨 Observed: Extreme misalignment (target=0.0, draft=0.99)')
    print()
    
    # Test the simplest possible case
    test_prompts = [
        "The",
        "Hello",
        "The quick brown",
        "1 + 1 ="
    ]
    
    for prompt in test_prompts:
        print(f'🧪 Testing prompt: "{prompt}"')
        
        try:
            # Simple tokenizer test first
            from transformers import AutoTokenizer
            
            tokenizer_1b = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
            tokenizer_3b = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-3B-Instruct')
            
            tokens_1b = tokenizer_1b.encode(prompt)
            tokens_3b = tokenizer_3b.encode(prompt) 
            
            print(f'   📝 1B tokens: {tokens_1b}')
            print(f'   📝 3B tokens: {tokens_3b}')
            
            if tokens_1b != tokens_3b:
                print('   🚨 TOKENIZER MISMATCH - this explains everything!')
                print(f'   🔍 Difference: {set(tokens_1b) - set(tokens_3b)} vs {set(tokens_3b) - set(tokens_1b)}')
                return
            else:
                print('   ✅ Tokenizers match perfectly')
            
            # Test vocab sizes
            vocab_1b = tokenizer_1b.vocab_size
            vocab_3b = tokenizer_3b.vocab_size
            print(f'   📊 1B vocab: {vocab_1b}, 3B vocab: {vocab_3b}')
            
            if vocab_1b != vocab_3b:
                print('   🚨 VOCAB SIZE MISMATCH!')
                return
            
            # Check if we can load model configs
            try:
                from transformers import AutoConfig
                config_1b = AutoConfig.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
                config_3b = AutoConfig.from_pretrained('unsloth/Llama-3.2-3B-Instruct')
                
                print(f'   📋 1B config: {config_1b.architectures}, vocab={config_1b.vocab_size}')
                print(f'   📋 3B config: {config_3b.architectures}, vocab={config_3b.vocab_size}')
                
                if config_1b.vocab_size != config_3b.vocab_size:
                    print('   🚨 CONFIG VOCAB MISMATCH!')
                    return
                    
            except Exception as e:
                print(f'   ⚠️  Could not load configs: {e}')
            
        except Exception as e:
            print(f'   ❌ Tokenizer test failed: {e}')
            continue
        
        print()
    
    print('🔍 ADVANCED DEBUGGING: Model Loading Issues')
    print('-' * 50)
    
    # Check what models are actually being loaded in exo
    try:
        from exo.models import model_cards
        
        card_1b = model_cards.get('llama-3.2-1b', {})
        card_3b = model_cards.get('llama-3.2-3b', {})
        
        print(f'📋 1B model card:')
        print(f'   Repo: {card_1b.get("repo", {})}')
        print(f'   Vocab: {card_1b.get("vocab_size")}')
        print(f'   Layers: {card_1b.get("layers")}')
        
        print(f'📋 3B model card:')
        print(f'   Repo: {card_3b.get("repo", {})}')
        print(f'   Vocab: {card_3b.get("vocab_size")}')
        print(f'   Layers: {card_3b.get("layers")}')
        
        # Check if PyTorch repos are different
        torch_repo_1b = card_1b.get("repo", {}).get("TorchDynamicShardInferenceEngine")
        torch_repo_3b = card_3b.get("repo", {}).get("TorchDynamicShardInferenceEngine")
        
        print(f'🔍 PyTorch repos:')
        print(f'   1B: {torch_repo_1b}')
        print(f'   3B: {torch_repo_3b}')
        
        if torch_repo_1b and torch_repo_3b:
            if torch_repo_1b != torch_repo_3b:
                print('   ✅ Different repos (expected)')
            else:
                print('   🚨 SAME REPO - this could be wrong!')
        
    except Exception as e:
        print(f'❌ Model card check failed: {e}')
    
    print()
    print('🔍 HYPOTHESIS TESTING:')
    print('1. ✅ Same tokenizer (checked)')
    print('2. ✅ Same vocab size (checked)') 
    print('3. ❓ Different model variants (base vs instruct)?')
    print('4. ❓ Model loading bug in PyTorch engine?')
    print('5. ❓ Cache state corruption?')
    print('6. ❓ Logits extraction bug?')
    print('7. ❓ Temperature application bug?')
    
    print()
    print('💡 NEXT STEPS:')
    print('- Check if both models are actually instruct-tuned')
    print('- Verify models load correctly in PyTorch')
    print('- Test with minimal prompt outside speculative decoding')
    print('- Check if cache state is corrupting predictions')

if __name__ == '__main__':
    asyncio.run(debug_model_mismatch()) 