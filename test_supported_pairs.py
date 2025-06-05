#!/usr/bin/env python3
"""
Test vocabulary compatibility between llama-3.2-1b and other supported LLaMA models
Focus only on models that are confirmed to work in exo with TinyGrad
"""

import asyncio
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.shard_download import NewShardDownloader
from exo.inference.shard import Shard

async def get_vocab_size(model_id, engine):
    """Get vocabulary size for a model"""
    try:
        print(f"  📥 Loading {model_id}...")
        shard = Shard(model_id=model_id, start_layer=0, end_layer=0, n_layers=1)
        await engine.ensure_shard(shard)
        
        vocab_size = engine.tokenizer.vocab_size
        print(f"  ✅ {model_id}: {vocab_size:,} tokens")
        return vocab_size
        
    except Exception as e:
        print(f"  ❌ {model_id}: Error - {e}")
        return None

async def test_supported_pairs():
    """Test llama-3.2-1b with other confirmed supported models"""
    
    print("🧪 Testing LLaMA 3.2-1B with Other Supported Models")
    print("=" * 60)
    print("✅ llama-3.2-1b: Confirmed 128,000 vocab")
    print("🎯 Testing pairs with other TinyGrad-supported LLaMA models")
    print()
    
    # Models we know support TinyGrad from exo/models.py
    test_models = [
        "llama-3.2-3b",     # Most likely to work - same generation
        "llama-3.1-8b",     # Cross-generation test
        "llama-3-8b",       # Older generation test
    ]
    
    engine = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    
    # Get llama-3.2-1b vocab (we know it's 128k but let's confirm)
    print("🔍 Confirming llama-3.2-1b vocabulary:")
    base_vocab = await get_vocab_size("llama-3.2-1b", engine)
    
    if not base_vocab:
        print("❌ Could not load llama-3.2-1b - stopping test")
        return
    
    print(f"✅ Base model vocab: {base_vocab:,} tokens")
    print()
    
    # Test each potential target model
    compatible_pairs = []
    
    for target_model in test_models:
        print(f"🔍 Testing compatibility with {target_model}:")
        target_vocab = await get_vocab_size(target_model, engine)
        
        if target_vocab:
            if target_vocab == base_vocab:
                print(f"  ✅ COMPATIBLE! Both have {base_vocab:,} tokens")
                compatible_pairs.append(target_model)
                print(f"  🚀 llama-3.2-1b + {target_model} = READY FOR SPECULATIVE DECODING!")
            else:
                print(f"  ❌ Incompatible: {base_vocab:,} vs {target_vocab:,}")
        else:
            print(f"  ❌ Could not test {target_model}")
        print()
    
    # Summary
    print("=" * 60)
    print("🎉 FINAL RESULTS:")
    print(f"✅ Base model: llama-3.2-1b ({base_vocab:,} tokens)")
    
    if compatible_pairs:
        print("🚀 COMPATIBLE TARGET MODELS:")
        for model in compatible_pairs:
            print(f"   ✅ {model}")
        print()
        print("🎯 RECOMMENDED SPECULATIVE PAIRS:")
        for model in compatible_pairs:
            print(f"   🤖 llama-3.2-1b (draft) + {model} (target)")
    else:
        print("❌ No compatible models found")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_supported_pairs()) 