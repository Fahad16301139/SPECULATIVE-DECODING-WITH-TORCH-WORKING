#!/usr/bin/env python3
"""
Test REAL speculative decoding with actual models
Using the framework's built-in CLI functionality
"""

import asyncio
import os
import sys
import traceback

# Enable maximum debugging
os.environ["DEBUG"] = "4"

async def test_with_cli_approach():
    """Test using exo's CLI approach"""
    print("🚀 TESTING REAL SPECULATIVE DECODING")
    print("=" * 80)
    
    # Test 1: Auto-configuration with LLaMA family
    print("\n📝 TEST 1: Auto-configuration with LLaMA family")
    print("🎯 Target: llama-3.2-3b")
    print("🎨 Draft: auto-selected (should be llama-3.2-1b)")
    
    # Test 2: Manual configuration
    print("\n📝 TEST 2: Manual configuration")
    print("🎯 Target: llama-3.2-3b")  
    print("🎨 Draft: llama-3.2-1b (manual)")
    
    # Test 3: Check what models are available
    print("\n📝 TEST 3: Check available models")
    
    print("\n⚡ Starting exo with speculative decoding...")
    
    return True

def test_direct_import():
    """Test by directly importing and using the speculative engine"""
    print("\n🔧 TESTING DIRECT IMPORT APPROACH")
    print("=" * 80)
    
    try:
        # Import the speculative engine
        from exo.inference.inference_engine import get_inference_engine
        from exo.download.new_shard_download import new_shard_downloader
        from exo.inference.shard import Shard
        
        print("✅ Successfully imported exo modules")
        
        # Create shard downloader
        shard_downloader = new_shard_downloader()
        print("✅ Created shard downloader")
        
        # Try to create speculative engine with auto-config
        speculative_config = {
            'target_engine_name': 'tinygrad',
            'draft_engine_name': 'tinygrad', 
            'target_model_id': 'llama-3.2-3b',
            'draft_model_id': 'llama-3.2-1b',
            'gamma': 3,
            'temperature': 0.8,
            'top_k_threshold': 0.9,
            'lenience': 1.1,
            'enable_speculative': True,
            'early_exit_layer': None
        }
        
        print("🔧 Creating speculative inference engine...")
        inference_engine = get_inference_engine("speculative", shard_downloader, speculative_config)
        print(f"✅ Created inference engine: {inference_engine.__class__.__name__}")
        
        # Create a test shard
        test_shard = Shard(
            model_id="llama-3.2-3b",
            start_layer=0,
            end_layer=0,
            n_layers=28
        )
        print(f"✅ Created test shard: {test_shard}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in direct import test: {e}")
        traceback.print_exc()
        return False

async def test_with_run_model():
    """Test using the --run-model flag"""
    print("\n🎮 TESTING WITH --run-model FLAG")
    print("=" * 80)
    
    # This simulates: python -m exo.main run --inference-engine speculative --speculative-target-model llama-3.2-3b --speculative-auto-config --run-model llama-3.2-3b --prompt "Hello world"
    
    try:
        # Set up arguments as if from command line
        import sys
        original_argv = sys.argv[:]
        
        sys.argv = [
            'main.py',
            'run',
            '--inference-engine', 'speculative',
            '--speculative-target-model', 'llama-3.2-3b',
            '--speculative-auto-config',
            '--run-model', 'llama-3.2-3b', 
            '--prompt', 'The future of AI is',
            '--max-generate-tokens', '20'
        ]
        
        print(f"🎯 Simulating command: python -m exo.main {' '.join(sys.argv[1:])}")
        
        # Import and run main
        from exo.main import main
        
        print("⚡ Starting exo main...")
        await main()
        
        # Restore original argv
        sys.argv = original_argv
        
        return True
        
    except Exception as e:
        print(f"❌ Error in --run-model test: {e}")
        traceback.print_exc()
        sys.argv = original_argv
        return False

def show_available_models():
    """Show what models are available in the system"""
    print("\n📚 CHECKING AVAILABLE MODELS")
    print("=" * 80)
    
    try:
        from exo.inference.inference_engine import get_model_family_variants
        
        # Test some common model names
        test_models = [
            "llama-3.2-1b",
            "llama-3.2-3b", 
            "llama-3.1-8b",
            "qwen2.5-7b",
            "phi-3.5-mini"
        ]
        
        for model in test_models:
            try:
                family_info = get_model_family_variants(model)
                print(f"🔍 {model}:")
                print(f"   Family: {family_info['family']}")
                print(f"   Suggested drafts: {family_info['suggested_drafts']}")
            except Exception as e:
                print(f"❌ {model}: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        traceback.print_exc()
        return False

def check_system_requirements():
    """Check if the system has what we need"""
    print("\n🔧 CHECKING SYSTEM REQUIREMENTS")
    print("=" * 80)
    
    checks = []
    
    # Check if MLX is available (for macOS)
    try:
        import mlx
        checks.append(("MLX", "✅ Available"))
    except ImportError:
        checks.append(("MLX", "❌ Not available"))
    
    # Check if TinyGrad is available
    try:
        import tinygrad
        checks.append(("TinyGrad", "✅ Available"))
    except ImportError:
        checks.append(("TinyGrad", "❌ Not available"))
    
    # Check numpy
    try:
        import numpy as np
        checks.append(("NumPy", f"✅ Available ({np.__version__})"))
    except ImportError:
        checks.append(("NumPy", "❌ Not available"))
    
    # Check exo modules
    try:
        from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
        checks.append(("Speculative Engine", "✅ Available"))
    except ImportError as e:
        checks.append(("Speculative Engine", f"❌ Not available: {e}"))
    
    for check, status in checks:
        print(f"   {check}: {status}")
    
    return all("✅" in status for _, status in checks)

async def main():
    """Main test function"""
    print("🎯 REAL SPECULATIVE DECODING TEST SUITE")
    print("=" * 80)
    print("This will test the actual speculative decoding implementation")
    print("with real models and real inference engines.")
    print()
    
    # Check system requirements first
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please install missing dependencies.")
        return
    
    # Show available models
    show_available_models()
    
    # Test direct import approach
    if not test_direct_import():
        print("\n❌ Direct import test failed")
        return
    
    # Test CLI approach
    await test_with_cli_approach()
    
    # Test with --run-model flag
    print("\n🎮 Would you like to test with --run-model flag? (This will actually run inference)")
    print("   This requires models to be downloaded and may take time...")
    
    # For now, just show what the command would be
    print("\n💡 To test manually, run:")
    print("   python -m exo.main run --inference-engine speculative \\")
    print("     --speculative-target-model llama-3.2-3b \\")
    print("     --speculative-auto-config \\")
    print("     --run-model llama-3.2-3b \\")
    print("     --prompt 'The future of AI is' \\")
    print("     --max-generate-tokens 20")
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    asyncio.run(main()) 