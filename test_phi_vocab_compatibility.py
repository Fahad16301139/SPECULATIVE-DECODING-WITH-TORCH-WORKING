#!/usr/bin/env python3
"""
Test Phi-1.5 and Phi-2 vocabulary compatibility for speculative decoding
Testing the claim that both use WordPiece-like vocab with ~50K tokens
"""

import asyncio
import sys
from pathlib import Path

# Add exo to path
exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader

async def test_phi_vocab_compatibility():
    """Test that Phi-1.5 and Phi-2 have compatible vocabularies"""
    
    print("🧪 Testing Phi Model Vocabulary Compatibility")
    print("=" * 60)
    print("🎯 Testing Phi-1.5 and Phi-2 for speculative decoding")
    print("📝 Expected: ~50K WordPiece-like tokens")
    print("⚠️  Note: User mentioned 'slight tokenizer variance between versions'")
    print()
    
    # Initialize with proper shard downloader
    shard_downloader = NewShardDownloader()
    engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    models_to_test = [
        ("phi-1.5", 24),  # 24 layers
        ("phi-2", 32),    # 32 layers  
    ]
    
    vocab_info = {}
    
    for model_id, n_layers in models_to_test:
        print(f"🔍 Testing: {model_id}")
        
        try:
            # Create shard
            shard = Shard(
                model_id=model_id,
                start_layer=0,
                end_layer=-1,
                n_layers=n_layers
            )
            
            print(f"  📥 Loading {model_id}...")
            await engine.ensure_shard(shard)
            
            # Try to get vocab size from different places
            vocab_size = None
            vocab_source = "Unknown"
            
            # Check model config
            if hasattr(engine.model, 'config'):
                if hasattr(engine.model.config, 'vocab_size'):
                    vocab_size = engine.model.config.vocab_size
                    vocab_source = "model.config.vocab_size"
                elif hasattr(engine.model.config, 'vocabulary_size'):
                    vocab_size = engine.model.config.vocabulary_size
                    vocab_source = "model.config.vocabulary_size"
            
            # Check model directly
            if vocab_size is None and hasattr(engine.model, 'vocab_size'):
                vocab_size = engine.model.vocab_size
                vocab_source = "model.vocab_size"
            
            # Check tokenizer if available
            if hasattr(engine, 'tokenizer'):
                if hasattr(engine.tokenizer, 'vocab_size'):
                    tokenizer_vocab = engine.tokenizer.vocab_size
                    print(f"  📖 Tokenizer vocab size: {tokenizer_vocab}")
                    if vocab_size is None:
                        vocab_size = tokenizer_vocab
                        vocab_source = "tokenizer.vocab_size"
            
            vocab_info[model_id] = vocab_size
            
            print(f"  📊 Vocab size: {vocab_size} (from {vocab_source})")
            
            # Check if it matches expected ~50K
            if vocab_size:
                if 49000 <= vocab_size <= 51000:
                    print(f"  ✅ Expected range: ~50K tokens ✓")
                else:
                    print(f"  ⚠️  Unexpected size: expected ~50K, got {vocab_size}")
            else:
                print(f"  ❌ Could not determine vocab size")
                
        except Exception as e:
            print(f"  ❌ Error loading {model_id}: {e}")
            vocab_info[model_id] = None
            
        print()
    
    # Check compatibility
    print("🤝 Compatibility Analysis:")
    print("-" * 40)
    
    phi_15_vocab = vocab_info.get("phi-1.5")
    phi_2_vocab = vocab_info.get("phi-2") 
    
    if phi_15_vocab and phi_2_vocab:
        if phi_15_vocab == phi_2_vocab:
            print(f"✅ COMPATIBLE: Both models have identical vocab size ({phi_15_vocab})")
            print(f"🎯 Perfect for speculative decoding!")
            print(f"🚀 Phi-1.5 (draft) → Phi-2 (target) should work!")
        else:
            print(f"❌ INCOMPATIBLE: Different vocab sizes")
            print(f"   Phi-1.5: {phi_15_vocab}")
            print(f"   Phi-2: {phi_2_vocab}")
            print(f"   Difference: {abs(phi_15_vocab - phi_2_vocab)} tokens")
            
            # Check if they're close (might work with minor patching)
            diff_percent = abs(phi_15_vocab - phi_2_vocab) / max(phi_15_vocab, phi_2_vocab) * 100
            if diff_percent < 1:
                print(f"⚠️  Very close ({diff_percent:.2f}% difference) - might work with patching")
            elif diff_percent < 5:
                print(f"⚠️  Close ({diff_percent:.2f}% difference) - might work with alignment")
            else:
                print(f"🚫 Too different ({diff_percent:.2f}% difference) - incompatible")
    else:
        print(f"❌ Could not test compatibility - missing vocab info")
        if not phi_15_vocab:
            print(f"   Phi-1.5: Failed to load")
        if not phi_2_vocab:
            print(f"   Phi-2: Failed to load")
    
    print()
    print("=" * 60)
    print("🎉 Phi Vocabulary Compatibility Test Complete!")
    print("💡 Results will show if Phi models can be used for speculative decoding")
    print("📚 Unlike LLaMA models, these should have more similar vocabularies")

if __name__ == "__main__":
    asyncio.run(test_phi_vocab_compatibility()) 