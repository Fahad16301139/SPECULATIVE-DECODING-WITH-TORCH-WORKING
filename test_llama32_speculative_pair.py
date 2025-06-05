#!/usr/bin/env python3
"""
Test llama-3.2-1b + llama-3.2-8b speculative decoding pair
This should be our best candidate since both are from the same LLaMA generation
"""

import asyncio
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.shard_download import NewShardDownloader
from exo.inference.shard import Shard

async def test_llama32_pair():
    """Test llama-3.2-1b (draft) + llama-3.2-8b (target) for speculative decoding"""
    
    print("🚀 Testing LLaMA 3.2 Speculative Decoding Pair")
    print("=" * 60)
    print("🎯 Draft Model: llama-3.2-1b (16 layers)")
    print("🎯 Target Model: llama-3.2-8b (32 layers)")
    print("💡 Expected: Both should have 128,000 vocab tokens")
    print()
    
    draft_model = "llama-3.2-1b"
    target_model = "llama-3.2-8b"
    
    engine = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    
    # Test draft model
    print("🔍 Testing Draft Model (llama-3.2-1b):")
    try:
        print(f"  📥 Loading {draft_model}...")
        shard = Shard(model_id=draft_model, start_layer=0, end_layer=0, n_layers=1)
        await engine.ensure_shard(shard)
        
        draft_vocab = engine.tokenizer.vocab_size
        print(f"  ✅ {draft_model}: {draft_vocab:,} tokens")
        
    except Exception as e:
        print(f"  ❌ {draft_model}: Error - {e}")
        return
    
    print()
    
    # Test target model
    print("🔍 Testing Target Model (llama-3.2-8b):")
    try:
        print(f"  📥 Loading {target_model}...")
        shard = Shard(model_id=target_model, start_layer=0, end_layer=0, n_layers=1)
        await engine.ensure_shard(shard)
        
        target_vocab = engine.tokenizer.vocab_size
        print(f"  ✅ {target_model}: {target_vocab:,} tokens")
        
    except Exception as e:
        print(f"  ❌ {target_model}: Error - {e}")
        return
    
    print()
    
    # Compatibility check
    print("🔍 Compatibility Analysis:")
    print("-" * 40)
    
    if draft_vocab == target_vocab:
        print(f"✅ COMPATIBLE! Both models have {draft_vocab:,} tokens")
        print("🎉 This pair is READY for speculative decoding!")
        print()
        print("🚀 SPECULATIVE DECODING SETUP:")
        print(f"   📝 Draft Model: {draft_model} ({draft_vocab:,} vocab)")
        print(f"   🎯 Target Model: {target_model} ({target_vocab:,} vocab)")
        print("   ⚡ Expected speedup: 2-3x faster inference")
        print()
        print("🎯 NEXT STEPS:")
        print("   1. Implement speculative decoding logic")
        print("   2. Test with actual text generation")
        print("   3. Measure speedup vs single model")
        
    else:
        print(f"❌ INCOMPATIBLE: {draft_vocab:,} vs {target_vocab:,} tokens")
        print("   Cannot use for speculative decoding - vocab mismatch")
    
    print("=" * 60)

async def test_tokenizer_compatibility():
    """Test if tokenizers produce identical outputs"""
    
    print("\n🔍 Testing Tokenizer Compatibility:")
    print("-" * 40)
    
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning.",
    ]
    
    engine1 = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    engine2 = TinygradDynamicShardInferenceEngine(NewShardDownloader())
    
    # Load both models
    shard1 = Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=0, n_layers=1)
    shard2 = Shard(model_id="llama-3.2-8b", start_layer=0, end_layer=0, n_layers=1)
    
    await engine1.ensure_shard(shard1)
    await engine2.ensure_shard(shard2)
    
    all_compatible = True
    
    for text in test_texts:
        print(f"\n📝 Testing: '{text}'")
        
        try:
            tokens1 = await engine1.encode(shard1, text)
            tokens2 = await engine2.encode(shard2, text)
            
            if len(tokens1) == len(tokens2) and (tokens1 == tokens2).all():
                print(f"  ✅ Identical tokenization: {len(tokens1)} tokens")
            else:
                print(f"  ❌ Different tokenization: {len(tokens1)} vs {len(tokens2)} tokens")
                all_compatible = False
                
        except Exception as e:
            print(f"  ❌ Error testing tokenization: {e}")
            all_compatible = False
    
    if all_compatible:
        print("\n🎉 TOKENIZER COMPATIBILITY: CONFIRMED!")
        print("   Both models use identical tokenization")
        print("   ✅ Ready for speculative decoding implementation")
    else:
        print("\n❌ TOKENIZER COMPATIBILITY: FAILED!")
        print("   Models use different tokenization")
        print("   ❌ Cannot use for speculative decoding")

if __name__ == "__main__":
    asyncio.run(test_llama32_pair())
    asyncio.run(test_tokenizer_compatibility()) 