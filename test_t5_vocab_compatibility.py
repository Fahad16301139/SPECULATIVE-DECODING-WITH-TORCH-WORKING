#!/usr/bin/env python3
"""
Test T5 model vocabulary compatibility for real speculative decoding
Based on original Leviathan et al. research
"""

import asyncio
import sys
from pathlib import Path

# Add exo to path
exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine

async def test_t5_vocab_compatibility():
    """Test that T5 models have identical vocabularies for speculative decoding"""
    
    print("🧪 Testing T5 Model Vocabulary Compatibility")
    print("=" * 60)
    
    # T5 model pairs to test (original Leviathan et al. used T5-Small + T5-XXL)
    test_pairs = [
        ("t5-small", "t5-base"),
        ("t5-small", "t5-large"), 
        ("t5-base", "t5-large"),
        # Add 3B and 11B if available
    ]
    
    engine = TinygradDynamicShardInferenceEngine()
    
    for draft_model, target_model in test_pairs:
        print(f"\n🔍 Testing: {draft_model} → {target_model}")
        
        try:
            # Create shards for both models
            draft_shard = Shard(
                model_id=draft_model,
                start_layer=0,
                end_layer=-1,
                n_layers=6 if "small" in draft_model else 12 if "base" in draft_model else 24
            )
            
            target_shard = Shard(
                model_id=target_model, 
                start_layer=0,
                end_layer=-1,
                n_layers=12 if "base" in target_model else 24
            )
            
            # Load models and check vocab sizes
            print(f"  📥 Loading {draft_model}...")
            await engine.ensure_shard(draft_shard)
            draft_vocab = getattr(engine.model, 'vocab_size', None)
            
            print(f"  📥 Loading {target_model}...")
            await engine.ensure_shard(target_shard)
            target_vocab = getattr(engine.model, 'vocab_size', None)
            
            # Check compatibility
            print(f"  📊 Vocab sizes:")
            print(f"    {draft_model}: {draft_vocab}")
            print(f"    {target_model}: {target_vocab}")
            
            if draft_vocab == target_vocab == 32000:
                print(f"  ✅ COMPATIBLE: Identical vocab size ({draft_vocab})")
                print(f"  🎯 Perfect for speculative decoding!")
            elif draft_vocab == target_vocab:
                print(f"  ✅ COMPATIBLE: Identical vocab size ({draft_vocab})")
                print(f"  ⚠️  Unexpected size (expected 32000)")
            else:
                print(f"  ❌ INCOMPATIBLE: Different vocab sizes")
                print(f"  🚫 Cannot use for speculative decoding")
                
        except Exception as e:
            print(f"  ❌ Error testing {draft_model} → {target_model}: {e}")
            
    print("\n" + "=" * 60)
    print("🎉 T5 Vocabulary Compatibility Test Complete!")
    print("💡 T5 models should all have 32k vocab (WordPiece tokenizer)")
    print("📚 Based on original Leviathan et al. speculative decoding research")

if __name__ == "__main__":
    asyncio.run(test_t5_vocab_compatibility()) 