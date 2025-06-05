#!/usr/bin/env python3
"""
DEBUG: Why is acceptance rate still 100%?

Let's trace through the actual acceptance sampling logic step by step.
"""

import asyncio
import numpy as np
import os
import sys
from pathlib import Path

exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.inference.shard import Shard
from exo.inference.adaptive_speculative_engine import AdaptiveSpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine 
from exo.download.new_shard_download import NewShardDownloader

os.environ['DEBUG'] = '3'  # Maximum debug

async def debug_acceptance_sampling():
    """Debug exactly what's happening in acceptance sampling"""
    
    print("🔍 DEBUGGING ACCEPTANCE SAMPLING")
    print("=" * 50)
    
    # Create engines
    shard_downloader = NewShardDownloader()
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Create custom debug engine
    debug_engine = AdaptiveSpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=3,  # Just 3 tokens for easier debugging
        temperature=1.0  # Higher temperature for more variation
    )
    
    target_shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=0, n_layers=28)
    
    print("🧪 Running single debug test...")
    
    try:
        # Just run one test to see detailed debug output
        result, state, tokens = await debug_engine.infer_prompt_multi(
            "debug-test",
            target_shard,
            "Hello world"
        )
        
        print(f"\n📊 Results:")
        print(f"Generated tokens: {tokens}")
        if tokens:
            text = await debug_engine.decode(target_shard, np.array(tokens))
            print(f"Generated text: '{text}'")
        
        stats = debug_engine.get_real_stats()
        print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
        print(f"Vocab compatibility: {stats['vocab_compatibility']}")
        
        # Check if we're actually using the right method
        print(f"\n🔍 Method Resolution:")
        print(f"Using _real_speculative_acceptance: {hasattr(debug_engine, '_real_speculative_acceptance')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def manual_acceptance_test():
    """Manually test the acceptance logic with known values"""
    print("\n🔧 MANUAL ACCEPTANCE LOGIC TEST")
    print("=" * 50)
    
    # Create fake probability distributions for testing
    print("Testing with manual probability values...")
    
    # Target model says token 100 has probability 0.3
    target_probs = np.zeros((1, 3, 3072))
    target_probs[0, 0, 100] = 0.3
    target_probs[0, 1, 101] = 0.4  
    target_probs[0, 2, 102] = 0.2
    
    # Draft model says same tokens have different probabilities
    draft_probs = np.zeros((1, 3, 2048))
    draft_probs[0, 0, 100] = 0.8  # Much higher confidence
    draft_probs[0, 1, 101] = 0.1  # Much lower confidence  
    draft_probs[0, 2, 102] = 0.5  # Moderate confidence
    
    # Draft tokens selected
    draft_tokens = np.array([[100, 101, 102]])
    
    print(f"Draft tokens: {draft_tokens[0]}")
    print(f"Target probs: [0.3, 0.4, 0.2]") 
    print(f"Draft probs:  [0.8, 0.1, 0.5]")
    print(f"Expected acceptance ratios: [0.375, 4.0, 0.4]")
    print(f"Expected acceptance ratios (capped): [0.375, 1.0, 0.4]")
    
    # Manually calculate what SHOULD happen
    for t in range(3):
        token_id = draft_tokens[0, t]
        p_target = [0.3, 0.4, 0.2][t]
        p_draft = [0.8, 0.1, 0.5][t]
        
        acceptance_ratio = min(1.0, p_target / p_draft)
        print(f"Token {t} (id={token_id}): ratio = min(1.0, {p_target}/{p_draft}) = {acceptance_ratio:.3f}")
        
        # With acceptance ratio, what percentage should be accepted?
        expected_acceptance = acceptance_ratio * 100
        print(f"  Expected acceptance rate: {expected_acceptance:.1f}%")

if __name__ == "__main__":
    print("🐛 DEBUGGING SPECULATIVE DECODING ACCEPTANCE")
    print("Why are we getting 100% acceptance rates?")
    print()
    
    # Run debug
    asyncio.run(debug_acceptance_sampling())
    
    # Run manual test
    asyncio.run(manual_acceptance_test()) 