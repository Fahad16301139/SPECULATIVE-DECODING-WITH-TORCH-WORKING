#!/usr/bin/env python3
"""
Test LLaMA 3.2 model vocabulary compatibility for speculative decoding
Testing if llama-3.2-1b and llama-3.2-8b have identical vocabularies
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

async def test_llama32_vocab_compatibility():
    """Test that LLaMA 3.2 1B and 8B have compatible vocabularies"""
    
    print("🧪 Testing LLaMA 3.2 Model Vocabulary Compatibility")
    print("=" * 60)
    print("🎯 Testing llama-3.2-1b and llama-3.2-8b for speculative decoding")
    print("🤔 Question: Do they have the same vocabulary size?")
    print("💭 Previous results: 1B=2048, 3B=3072 (incompatible)")
    print("🤞 Hoping: 1B and 8B might have same vocab!")
    print()
    
    # Initialize with proper shard downloader
    shard_downloader = NewShardDownloader()
    engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    models_to_test = [
        ("llama-3.2-1b", 16),  # 16 layers
        ("llama-3.2-8b", 32),  # 32 layers  
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
            
            # Compare to our known values
            if model_id == "llama-3.2-1b" and vocab_size == 2048:
                print(f"  ✅ Matches previous result: 2048 tokens")
            elif model_id == "llama-3.2-8b" and vocab_size == 2048:
                print(f"  🎯 Same as 1B model: 2048 tokens - PROMISING!")
            elif model_id == "llama-3.2-8b" and vocab_size != 2048:
                print(f"  ⚠️  Different from 1B model: {vocab_size} vs 2048")
            else:
                print(f"  ℹ️  New data point: {vocab_size} tokens")
                
        except Exception as e:
            print(f"  ❌ Error loading {model_id}: {e}")
            vocab_info[model_id] = None
            
        print()
    
    # Check compatibility
    print("🤝 Compatibility Analysis:")
    print("-" * 40)
    
    llama_1b_vocab = vocab_info.get("llama-3.2-1b")
    llama_8b_vocab = vocab_info.get("llama-3.2-8b") 
    
    if llama_1b_vocab and llama_8b_vocab:
        if llama_1b_vocab == llama_8b_vocab:
            print(f"✅ COMPATIBLE: Both models have identical vocab size ({llama_1b_vocab})")
            print(f"🎯 Perfect for speculative decoding!")
            print(f"🚀 llama-3.2-1b (draft) → llama-3.2-8b (target) should work!")
            print(f"🎉 FINALLY! A working LLaMA pair for real speculative decoding!")
        else:
            print(f"❌ INCOMPATIBLE: Different vocab sizes")
            print(f"   llama-3.2-1b: {llama_1b_vocab}")
            print(f"   llama-3.2-8b: {llama_8b_vocab}")
            print(f"   😞 Same problem as 1B vs 3B...")
    else:
        print(f"❌ Could not test compatibility - missing vocab info")
        if not llama_1b_vocab:
            print(f"   llama-3.2-1b: Failed to load")
        if not llama_8b_vocab:
            print(f"   llama-3.2-8b: Failed to load")
    
    print()
    print("=" * 60)
    print("🎉 LLaMA 3.2 Vocabulary Compatibility Test Complete!")
    print("💡 Results will show if we finally found working LLaMA models")
    print("🤞 Fingers crossed for identical vocabularies!")

if __name__ == "__main__":
    asyncio.run(test_llama32_vocab_compatibility()) 