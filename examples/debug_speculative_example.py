#!/usr/bin/env python3
"""
Example: Running Speculative Decoding with Debug Output

This example shows how to use speculative decoding with comprehensive debug output
to see all phases of the algorithm in action.

Usage:
    # Basic debug output
    DEBUG=1 python examples/debug_speculative_example.py
    
    # Detailed phase logging  
    DEBUG=2 python examples/debug_speculative_example.py
    
    # Token-level debugging
    DEBUG=3 python examples/debug_speculative_example.py
"""

import os
import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import exo modules
sys.path.insert(0, str(Path(__file__).parent.parent))

async def run_example_with_debug():
    """Run a practical example of speculative decoding with debug output."""
    print("🎯 Speculative Decoding Debug Example")
    print("="*50)
    
    # Show current debug level
    debug_level = int(os.environ.get('DEBUG', 0))
    print(f"Debug Level: {debug_level}")
    
    if debug_level == 0:
        print("\n💡 No debug output enabled. Try:")
        print("   DEBUG=1 python examples/debug_speculative_example.py  # Basic logs")
        print("   DEBUG=2 python examples/debug_speculative_example.py  # Detailed logs") 
        print("   DEBUG=3 python examples/debug_speculative_example.py  # Token-level logs")
        return
    
    print(f"\n🚀 Starting exo with speculative decoding...")
    print(f"   This will show all {debug_level}-level debug output for speculative decoding phases:")
    
    if debug_level >= 1:
        print("   ✅ PHASE 1: Model Compatibility Check")
        print("   ✅ PHASE 2: Draft Token Generation") 
        print("   ✅ PHASE 3: Target Model Verification")
        print("   ✅ PHASE 4: Acceptance/Rejection Sampling")
        print("   ✅ PHASE 5: Statistics Update")
    
    if debug_level >= 2:
        print("   ✅ Detailed operation logs")
        print("   ✅ Cache management")
        print("   ✅ Token shapes and timing")
    
    if debug_level >= 3:
        print("   ✅ Individual token processing")
        print("   ✅ Probability calculations")
        print("   ✅ Sampling details")
    
    print(f"\n📋 Example Commands to Run:")
    print(f"   # Use LLaMA family models with auto-config")
    print(f"   DEBUG={debug_level} exo --inference-engine speculative \\")
    print(f"     --speculative-target-model llama-3.1-8b \\")
    print(f"     --speculative-auto-config \\")
    print(f"     --prompt \"Explain quantum computing\"")
    
    print(f"\n   # Manual configuration with specific models")
    print(f"   DEBUG={debug_level} exo --inference-engine speculative \\")
    print(f"     --speculative-target-model llama-3.1-8b \\")
    print(f"     --speculative-draft-model llama-3.2-3b \\")
    print(f"     --speculative-gamma 5 \\")
    print(f"     --prompt \"Write a story about AI\"")
    
    print(f"\n   # Early exit mode (no draft model)")
    print(f"   DEBUG={debug_level} exo --inference-engine speculative \\")
    print(f"     --speculative-target-model llama-3.1-8b \\")
    print(f"     --prompt \"Hello world\"")
    
    print(f"\n🔍 What You'll See in the Debug Output:")
    
    if debug_level >= 1:
        print(f"\n   🔧 Initialization:")
        print(f"   🔧 Initializing SpeculativeInferenceEngine:")
        print(f"      Target model: llama-3.1-8b")
        print(f"      Draft model: llama-3.2-3b")
        print(f"      Gamma: 5")
        print(f"   🎯 Speculative mode: draft-target")
        
        print(f"\n   🔍 PHASE 1: Model Compatibility Check")
        print(f"      Checking: llama-3.1-8b (target) vs llama-3.2-3b (draft)")
        print(f"   ✅ Model compatibility check PASSED: llama-3.1-8b + llama-3.2-3b")
        
        print(f"\n   🎨 PHASE 2: Draft Token Generation (5 tokens)")
        print(f"   ✅ Generated draft token 1: 42")
        print(f"   ✅ Generated draft token 2: 100")
        print(f"   ... (up to gamma tokens)")
        
        print(f"\n   🎯 PHASE 3: Target Model Verification")
        print(f"      Verifying 5 draft tokens")
        
        print(f"\n   ⚖️  PHASE 4: Acceptance/Rejection Sampling")
        print(f"   ✅ ACCEPTED token 1: 42 (p=0.875)")
        print(f"   ❌ REJECTED token 2: 100 (p=0.234)")
        print(f"   🔄 Sampled adjusted token: 150")
        
        print(f"\n   📊 PHASE 5: Statistics Update")
        print(f"   ✅ Speculative decoding complete: 2/5 tokens accepted")
        print(f"      Acceptance rate: 0.400")
        print(f"      Potential speedup: ~2x")
    
    if debug_level >= 2:
        print(f"\n   🔧 Cache Management:")
        print(f"   🔧 Initialized target cache for test_request")
        print(f"   🔧 Initialized draft cache for test_request")
        
        print(f"\n   ⏱️  Timing Information:")
        print(f"      Draft inference time: 45.32ms")
        print(f"      Target inference time: 123.45ms")
        print(f"      Total time: 234.56ms")
        
        print(f"\n   🔢 Token Details:")
        print(f"      Initial tokens shape: (1, 5)")
        print(f"      Initial tokens: [1, 2, 3, 4, 5]...")
        print(f"      Updated tokens shape: (1, 6)")
    
    if debug_level >= 3:
        print(f"\n   🔍 Token-Level Processing:")
        print(f"      Evaluating draft token 1/5")
        print(f"      Token ID: 42")
        print(f"      Target prob: 0.123456")
        print(f"      Draft prob: 0.098765")
        print(f"      Acceptance prob: 0.875432")
        print(f"      Random value: 0.234567")
        
        print(f"\n   🎲 Sampling Details:")
        print(f"      Applying top-k filtering (threshold: 0.9)")
        print(f"      Sampling from draft logits with temperature: 1.0")
        print(f"      Gumbel sampling with temperature: 1.0")
    
    print(f"\n💡 Tips for Effective Debugging:")
    print(f"   • Start with DEBUG=1 to see the overall flow")
    print(f"   • Use DEBUG=2 to investigate timing and caching issues")
    print(f"   • Use DEBUG=3 when you need to debug acceptance/rejection logic")
    print(f"   • Check acceptance rates - should be > 0.3 for good performance")
    print(f"   • Look for model compatibility warnings")
    print(f"   • Monitor token generation vs acceptance patterns")

if __name__ == "__main__":
    asyncio.run(run_example_with_debug()) 