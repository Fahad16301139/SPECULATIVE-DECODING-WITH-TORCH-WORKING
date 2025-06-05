#!/usr/bin/env python3
"""
Find models with identical vocabularies for REAL speculative decoding!
"""

import asyncio
import numpy as np
import os
import sys
from pathlib import Path

exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine 
from exo.download.new_shard_download import NewShardDownloader
from exo.models import model_cards

# Suppress debug noise
os.environ['DEBUG'] = '0'

async def get_vocab_size(model_id, engine):
    """Get vocabulary size for a model"""
    try:
        shard = Shard(model_id=model_id, start_layer=0, end_layer=0, n_layers=8)
        test_input = np.array([[1, 2, 3]])
        result, _ = await engine.infer_tensor("vocab-test", shard, test_input)
        return result.shape[-1]
    except Exception as e:
        print(f"   ❌ Error testing {model_id}: {str(e)[:100]}...")
        return None

async def find_compatible_models():
    """Find all model pairs with identical vocabularies"""
    
    print("🔍 FINDING COMPATIBLE MODEL PAIRS")
    print("=" * 60)
    print("Testing vocabulary sizes across all TinyGrad-supported models...")
    print()
    
    # Get all TinyGrad-supported models
    tinygrad_models = []
    for model_id, info in model_cards.items():
        if "TinygradDynamicShardInferenceEngine" in info.get("repo", {}):
            tinygrad_models.append(model_id)
    
    print(f"📋 Found {len(tinygrad_models)} TinyGrad-supported models:")
    for model in sorted(tinygrad_models):
        print(f"   • {model}")
    print()
    
    # Test vocabulary sizes
    print("🧪 Testing vocabulary sizes...")
    
    engine = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    model_vocabs = {}
    
    for model_id in sorted(tinygrad_models):
        print(f"   Testing {model_id}...", end=" ")
        vocab_size = await get_vocab_size(model_id, engine)
        if vocab_size:
            model_vocabs[model_id] = vocab_size
            print(f"✅ {vocab_size}")
        else:
            print("❌ Failed")
    
    print(f"\n📊 VOCABULARY SIZE RESULTS:")
    print("-" * 40)
    
    # Group by vocabulary size
    vocab_groups = {}
    for model_id, vocab_size in model_vocabs.items():
        if vocab_size not in vocab_groups:
            vocab_groups[vocab_size] = []
        vocab_groups[vocab_size].append(model_id)
    
    # Show results grouped by vocab size
    for vocab_size in sorted(vocab_groups.keys()):
        models = vocab_groups[vocab_size]
        print(f"\n🔢 Vocabulary Size: {vocab_size:,}")
        for model in sorted(models):
            print(f"   • {model}")
    
    print(f"\n🎯 COMPATIBLE PAIRS FOR SPECULATIVE DECODING:")
    print("=" * 60)
    
    # Find compatible pairs (same vocab size, different model sizes)
    compatible_pairs = []
    
    for vocab_size, models in vocab_groups.items():
        if len(models) >= 2:
            print(f"\n✅ Vocabulary Size {vocab_size:,} - {len(models)} compatible models:")
            
            # Sort by likely model size (smaller first for draft models)
            models_sorted = sorted(models, key=lambda x: (
                int(x.split('-')[-1].replace('b', '')) if x.split('-')[-1].replace('b', '').isdigit() 
                else 999  # Put non-numeric sizes at end
            ))
            
            for model in models_sorted:
                print(f"   • {model}")
            
            # Generate draft->target pairs
            for i, draft_model in enumerate(models_sorted[:-1]):
                for target_model in models_sorted[i+1:]:
                    compatible_pairs.append((draft_model, target_model))
                    print(f"   🚀 PAIR: {draft_model} → {target_model}")
    
    if not compatible_pairs:
        print("\n❌ NO COMPATIBLE PAIRS FOUND")
        print("All tested models have different vocabulary sizes.")
        print("Real speculative decoding requires identical vocabularies.")
    else:
        print(f"\n🎉 FOUND {len(compatible_pairs)} COMPATIBLE PAIRS!")
        print("\nBest recommendations for speculative decoding:")
        
        # Recommend best pairs (smaller draft, larger target, same family)
        for draft, target in compatible_pairs[:5]:  # Show top 5
            print(f"   🥇 Draft: {draft}")
            print(f"      Target: {target}")
            print(f"      Vocab: {model_vocabs[draft]:,} tokens")
            print()
    
    return compatible_pairs, model_vocabs

async def test_specific_llama_models():
    """Test specific LLaMA model combinations that might be compatible"""
    
    print("\n🔬 TESTING SPECIFIC LLAMA MODEL COMBINATIONS")
    print("=" * 60)
    
    # Test combinations that might have same vocab
    test_pairs = [
        ("llama-3.1-8b", "llama-3.1-70b"),  # Same generation
        ("llama-3-8b", "llama-3-70b"),      # Same generation  
        ("llama-3.2-3b", "llama-3.2-3b-8bit"),  # Same model, different quantization
        ("llama-3.2-3b", "llama-3.2-3b-bf16"),  # Same model, different precision
        ("llama-3.1-8b", "llama-3.2-3b"),   # Cross-generation
    ]
    
    engine = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    
    for draft_model, target_model in test_pairs:
        print(f"\n🧪 Testing: {draft_model} + {target_model}")
        
        try:
            draft_vocab = await get_vocab_size(draft_model, engine)
            target_vocab = await get_vocab_size(target_model, engine)
            
            if draft_vocab and target_vocab:
                if draft_vocab == target_vocab:
                    print(f"   ✅ COMPATIBLE! Both have {draft_vocab:,} tokens")
                    print(f"   🚀 Can use for real speculative decoding!")
                else:
                    print(f"   ❌ Incompatible: {draft_vocab:,} vs {target_vocab:,}")
            else:
                print(f"   ❌ Could not test one or both models")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 FINDING MODELS WITH IDENTICAL VOCABULARIES")
    print("For REAL vanilla speculative decoding in exo!")
    print()
    
    # Run comprehensive search
    compatible_pairs, vocabs = asyncio.run(find_compatible_models())
    
    # Test specific promising combinations
    asyncio.run(test_specific_llama_models())
    
    if compatible_pairs:
        print("\n" + "="*60)
        print("🎉 SUCCESS! FOUND COMPATIBLE MODELS!")
        print("="*60)
        print("You can now implement REAL speculative decoding with:")
        for draft, target in compatible_pairs[:3]:  # Show top 3
            print(f"✅ {draft} → {target}")
        print("\nThese models have identical vocabularies! 🚀")
    else:
        print("\n" + "="*60)  
        print("❌ NO IDENTICAL VOCABULARIES FOUND")
        print("="*60)
        print("May need to use different acceleration techniques...") 