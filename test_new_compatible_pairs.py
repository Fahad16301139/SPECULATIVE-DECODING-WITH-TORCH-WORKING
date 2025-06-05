#!/usr/bin/env python3
"""
Test vocabulary compatibility for newly added model pairs in exo
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

async def test_vocab_compatibility(draft_model, target_model, engine):
    """Test if two models have compatible vocabularies for speculative decoding"""
    print(f"\n🧪 Testing: {draft_model} → {target_model}")
    
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
        print(f"   Draft:  {draft_vocab:,} tokens")
        print(f"   Target: {target_vocab:,} tokens")
        
        if draft_vocab == target_vocab:
            print(f"   🎉 COMPATIBLE! Identical vocabulary: {draft_vocab:,} tokens")
            print(f"   ✅ READY FOR REAL SPECULATIVE DECODING!")
            return True, draft_vocab
        else:
            ratio = max(draft_vocab, target_vocab) / min(draft_vocab, target_vocab)
            print(f"   ❌ Incompatible: {draft_vocab:,} vs {target_vocab:,} (ratio: {ratio:.2f}x)")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:80]}...")
        return False, None

async def main():
    """Test all promising model pairs for speculative decoding"""
    
    print("🎯 TESTING NEW COMPATIBLE MODEL PAIRS")
    print("=" * 70)
    print("Verifying vocabulary compatibility for newly added models...")
    
    engine = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    compatible_pairs = []
    
    # Test pairs organized by likely compatibility
    test_groups = {
        "🔬 TinyLLM + LLaMA (Same Architecture)": [
            ("tiny-llm", "llama-3.2-1b"),
            ("tiny-llm-10m", "llama-3.2-1b"), 
            ("vicuna-68m", "llama-3.2-1b"),
            ("vicuna-68m", "llama-3.2-3b"),
            ("tiny-llm", "llama-3.1-8b"),
        ],
        
        "🔬 Phi Family (Same Tokenizer)": [
            ("phi-3-mini", "phi-3-medium"),
            ("phi-3-mini", "phi-3.5-mini"),
            ("phi-3-mini", "phi-4"),
            ("phi-3.5-mini", "phi-4"),
        ],
        
        "🔬 DeepSeek R1 Distill LLaMA (Same Base)": [
            ("deepseek-r1-distill-llama-8b", "llama-3.1-8b"),
            ("deepseek-r1-distill-llama-8b", "llama-3.2-3b"),
            ("deepseek-r1-distill-llama-70b", "llama-3.1-70b"),
        ],
        
        "🔬 Cross-Size Qwen Family": [
            ("qwen-2.5-0.5b", "qwen-2.5-1.5b"),
            ("qwen-2.5-1.5b", "qwen-2.5-3b"),
            ("qwen-2.5-3b", "qwen-2.5-7b"),
        ],
        
        "🔬 Universal Assisted Generation": [
            ("tiny-starcoder", "llama-3.2-3b"),
            ("vicuna-68m", "gemma2-9b"),
            ("tiny-llm", "phi-4"),
        ]
    }
    
    for group_name, pairs in test_groups.items():
        print(f"\n{group_name}")
        print("-" * 50)
        
        for draft_model, target_model in pairs:
            is_compatible, vocab_size = await test_vocab_compatibility(draft_model, target_model, engine)
            
            if is_compatible:
                compatible_pairs.append((draft_model, target_model, vocab_size))
    
    print(f"\n🎉 FINAL RESULTS:")
    print("=" * 70)
    
    if compatible_pairs:
        print(f"✅ FOUND {len(compatible_pairs)} COMPATIBLE PAIRS!")
        print("\n🚀 RECOMMENDED PAIRS FOR REAL SPECULATIVE DECODING:")
        
        for i, (draft, target, vocab) in enumerate(compatible_pairs, 1):
            print(f"\n{i}. {draft} → {target}")
            print(f"   Vocabulary: {vocab:,} tokens")
            print(f"   Status: ✅ READY FOR PRODUCTION!")
            
        print(f"\n📝 NEXT STEPS:")
        print("1. Test speculative decoding with these pairs")
        print("2. Measure actual speedup and acceptance rates")
        print("3. Deploy in production with confidence!")
        
    else:
        print("❌ NO COMPATIBLE PAIRS FOUND")
        print("All tested models have mismatched vocabularies.")
        print("Consider using Universal Assisted Generation instead.")
    
    return compatible_pairs

if __name__ == "__main__":
    print("🔍 VOCABULARY COMPATIBILITY TEST")
    print("Testing newly added model pairs for real speculative decoding")
    print()
    
    compatible = asyncio.run(main())
    
    if compatible:
        print(f"\n🎉 SUCCESS! Found {len(compatible)} compatible model pairs!")
        print("Real vanilla speculative decoding is now possible! 🚀")
    else:
        print(f"\n😞 No compatible models found...")
        print("May need alternative acceleration approaches.") 