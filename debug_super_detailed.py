#!/usr/bin/env python3
"""
SUPER DETAILED DEBUG: Trace every step of the speculative decoding
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

os.environ['DEBUG'] = '3'


class SuperDebugSpeculativeEngine(AdaptiveSpeculativeInferenceEngine):
    """Debug version with extreme logging"""
    
    def _real_speculative_acceptance(self, target_probs, draft_probs, draft_tokens):
        print("🚨 ENTERING _real_speculative_acceptance")
        print(f"   target_probs.shape: {target_probs.shape}")
        print(f"   draft_probs.shape: {draft_probs.shape}")
        print(f"   draft_tokens.shape: {draft_tokens.shape}")
        print(f"   draft_tokens: {draft_tokens}")
        
        batch, gamma = draft_tokens.shape
        acceptance_ratios = []
        
        print(f"   Processing {batch} batches, {gamma} tokens each")
        
        for b in range(batch):
            print(f"   🔍 Batch {b}:")
            accepted_count = 0
            
            for t in range(gamma):
                token_id = draft_tokens[b, t]
                print(f"      🎯 Token {t} (id={token_id}):")
                
                if token_id < target_probs.shape[-1] and token_id < draft_probs.shape[-1]:
                    p_target = target_probs[b, t, token_id]  
                    p_draft = draft_probs[b, t, token_id]
                    
                    print(f"         p_target = {p_target:.8f}")
                    print(f"         p_draft = {p_draft:.8f}")
                    
                    # Check for weird probability values
                    if p_target <= 0 or p_draft <= 0:
                        print(f"         ⚠️  WARNING: Zero or negative probabilities!")
                    
                    acceptance_ratio = min(1.0, p_target / max(p_draft, 1e-12))
                    acceptance_ratios.append(acceptance_ratio)
                    
                    r = np.random.random()
                    
                    print(f"         Acceptance ratio: {acceptance_ratio:.6f}")
                    print(f"         Random value: {r:.6f}")
                    print(f"         Accept condition: {r} <= {acceptance_ratio} = {r <= acceptance_ratio}")
                    
                    if r <= acceptance_ratio:
                        accepted_count += 1
                        print(f"         ✅ ACCEPTED (total: {accepted_count})")
                    else:
                        print(f"         ❌ REJECTED - stopping at token {t}")
                        break
                else:
                    print(f"         ❌ OUT OF VOCAB - token {token_id} not in both vocabs")
                    print(f"         target_vocab_size: {target_probs.shape[-1]}")
                    print(f"         draft_vocab_size: {draft_probs.shape[-1]}")
                    break
        
        print(f"   🏁 FINAL RESULT: accepted {accepted_count}/{gamma} tokens")
        print(f"   Acceptance ratios: {acceptance_ratios}")
        return accepted_count, acceptance_ratios


async def super_debug_test():
    """Run super detailed debug test"""
    
    print("🔬 SUPER DETAILED DEBUG")
    print("=" * 60)
    
    # Create engines
    shard_downloader = NewShardDownloader()
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    # Use debug engine
    debug_engine = SuperDebugSpeculativeEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        gamma=2,  # Just 2 tokens for simpler debugging
        temperature=1.0
    )
    
    target_shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=0, n_layers=28)
    
    print("🧪 Running super debug test with gamma=2...")
    
    try:
        result, state, tokens = await debug_engine.infer_prompt_multi(
            "super-debug",
            target_shard,
            "Test"
        )
        
        print(f"\n🎉 RESULTS:")
        print(f"Generated tokens: {tokens}")
        print(f"Length: {len(tokens)}")
        
        if tokens:
            text = await debug_engine.decode(target_shard, np.array(tokens))
            print(f"Generated text: '{text}'")
        
        stats = debug_engine.get_real_stats()
        print(f"\n📊 Final Stats:")
        print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
        print(f"Tokens per call: {stats['avg_tokens_per_call']:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚨 SUPER DETAILED SPECULATIVE DECODING DEBUG")
    print("This will show EVERY step of the acceptance process")
    print()
    
    asyncio.run(super_debug_test()) 