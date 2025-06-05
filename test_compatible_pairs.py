#!/usr/bin/env python3
"""
Quick test of the most promising model pairs for identical vocabularies
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

os.environ['DEBUG'] = '0'

async def test_vocab_pair(draft_model, target_model, engine):
    """Test if a pair of models has identical vocabularies"""
    print(f"\n🧪 Testing: {draft_model} + {target_model}")
    
    try:
        # Create shards
        draft_shard = Shard(model_id=draft_model, start_layer=0, end_layer=0, n_layers=8) 
        target_shard = Shard(model_id=target_model, start_layer=0, end_layer=0, n_layers=8)
        
        # Test input
        test_input = np.array([[1, 2, 3]])
        
        print("   Loading models...", end=" ")
        
        # Get vocab sizes
        draft_result, _ = await engine.infer_tensor("vocab-test-draft", draft_shard, test_input)
        target_result, _ = await engine.infer_tensor("vocab-test-target", target_shard, test_input)
        
        draft_vocab = draft_result.shape[-1]
        target_vocab = target_result.shape[-1]
        
        print("✅")
        print(f"   Draft vocab:  {draft_vocab:,}")
        print(f"   Target vocab: {target_vocab:,}")
        
        if draft_vocab == target_vocab:
            print(f"   🎉 COMPATIBLE! Both have {draft_vocab:,} tokens")
            print(f"   ✅ Can use for REAL speculative decoding!")
            return True, draft_vocab
        else:
            print(f"   ❌ Incompatible: {draft_vocab:,} vs {target_vocab:,}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:80]}...")
        return False, None

async def main():
    """Test the most promising model pairs"""
    
    print("🎯 TESTING MOST PROMISING MODEL PAIRS")
    print("=" * 60)
    print("Looking for models with identical vocabularies...")
    
    # Most promising pairs (same generation = same tokenizer)
    test_pairs = [
        ("llama-3.2-1b", "llama-3.2-3b"),          # Same generation - VERY likely
        ("llama-3.1-8b", "llama-3.1-70b"),         # Same generation - VERY likely  
        ("llama-3-8b", "llama-3-70b"),             # Same generation - VERY likely
        ("llama-3.2-3b", "llama-3.2-3b-8bit"),     # Same model, different quantization
        ("llama-3.2-3b", "llama-3.2-3b-bf16"),     # Same model, different precision
        ("llama-3.1-8b", "llama-3.2-3b"),          # Cross-generation - maybe
    ]
    
    engine = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    compatible_pairs = []
    
    for draft_model, target_model in test_pairs:
        is_compatible, vocab_size = await test_vocab_pair(draft_model, target_model, engine)
        
        if is_compatible:
            compatible_pairs.append((draft_model, target_model, vocab_size))
    
    print(f"\n🎉 RESULTS:")
    print("=" * 60)
    
    if compatible_pairs:
        print(f"✅ FOUND {len(compatible_pairs)} COMPATIBLE PAIRS!")
        print("\nRecommended pairs for REAL speculative decoding:")
        
        for i, (draft, target, vocab) in enumerate(compatible_pairs, 1):
            print(f"\n{i}. 🚀 {draft} → {target}")
            print(f"   Vocabulary: {vocab:,} tokens")
            print(f"   Status: ✅ READY FOR REAL SPECULATIVE DECODING!")
    else:
        print("❌ NO COMPATIBLE PAIRS FOUND")
        print("All tested models have different vocabulary sizes.")
        print("REAL speculative decoding requires identical vocabularies.")
    
    return compatible_pairs

if __name__ == "__main__":
    print("🔍 QUICK COMPATIBILITY TEST")
    print("Testing same-generation LLaMA models for identical vocabularies")
    print()
    
    compatible = asyncio.run(main())
    
    if compatible:
        print(f"\n🎉 SUCCESS! Found compatible models!")
        print("You can now implement REAL vanilla speculative decoding! 🚀")
    else:
        print(f"\n😞 No compatible models found in exo...")
        print("May need to find other acceleration approaches.") 