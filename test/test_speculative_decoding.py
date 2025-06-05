#!/usr/bin/env python3
"""
Test script for speculative decoding integration with exo.
"""

import asyncio
import numpy as np
import time
from pathlib import Path
import sys

# Add the exo directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exo.inference.inference_engine import get_inference_engine
from exo.inference.shard import Shard
from exo.download.shard_download import NoopShardDownloader


async def test_speculative_inference_engine():
    """Test the speculative inference engine."""
    print("Testing Speculative Decoding Integration with exo...")
    
    # Test with dummy engines for quick verification
    print("\n1. Testing with dummy engines...")
    
    # Config for speculative decoding with dummy engines
    speculative_config = {
        'target_engine_name': 'dummy',
        'draft_engine_name': 'dummy',
        'gamma': 3,
        'temperature': 1.0,
        'top_k_threshold': 0.9,
        'lenience': 1.0,
        'enable_speculative': True,
        'early_exit_layer': None
    }
    
    shard_downloader = NoopShardDownloader()
    
    # Create speculative engine
    spec_engine = get_inference_engine('speculative', shard_downloader, speculative_config)
    print(f"Created speculative engine: {spec_engine.__class__.__name__}")
    
    # Test basic functionality
    test_shard = Shard(
        model_id="dummy", 
        start_layer=0, 
        end_layer=7, 
        n_layers=8
    )
    
    # Test encoding
    print("\n2. Testing encoding...")
    prompt = "Hello, world!"
    tokens = await spec_engine.encode(test_shard, prompt)
    print(f"Encoded '{prompt}' to tokens: {tokens}")
    
    # Test inference
    print("\n3. Testing inference...")
    start_time = time.perf_counter()
    
    input_data = tokens.reshape(1, -1)
    output, state = await spec_engine.infer_tensor(
        "test_request", 
        test_shard, 
        input_data
    )
    
    end_time = time.perf_counter()
    print(f"Inference completed in {(end_time - start_time)*1000:.2f}ms")
    print(f"Output shape: {output.shape}")
    
    # Test decoding
    print("\n4. Testing decoding...")
    decoded = await spec_engine.decode(test_shard, output[0])
    print(f"Decoded output: '{decoded}'")
    
    # Test speculation stats
    print("\n5. Testing speculation statistics...")
    stats = spec_engine.get_speculation_stats()
    print(f"Speculation stats: {stats}")
    
    # Test enabling/disabling speculation
    print("\n6. Testing speculation toggle...")
    spec_engine.enable_speculation(False)
    print("Disabled speculation")
    
    output_no_spec, _ = await spec_engine.infer_tensor(
        "test_request_no_spec", 
        test_shard, 
        input_data
    )
    print(f"Output without speculation: {output_no_spec.shape}")
    
    spec_engine.enable_speculation(True)
    print("Re-enabled speculation")
    
    # Test gamma adjustment
    print("\n7. Testing gamma adjustment...")
    original_gamma = spec_engine.gamma
    spec_engine.set_gamma(7)
    print(f"Changed gamma from {original_gamma} to {spec_engine.gamma}")
    
    print("\n✅ All tests passed! Speculative decoding integration is working.")
    
    return True


async def test_early_exit_mode():
    """Test early exit mode (no draft engine)."""
    print("\n\n8. Testing early exit mode...")
    
    # Config for early exit (no draft engine)
    early_exit_config = {
        'target_engine_name': 'dummy',
        'draft_engine_name': None,  # No draft engine = early exit mode
        'gamma': 4,
        'temperature': 1.0,
        'top_k_threshold': 0.9,
        'lenience': 1.0,
        'enable_speculative': True,
        'early_exit_layer': 4
    }
    
    shard_downloader = NoopShardDownloader()
    
    # Create early exit speculative engine
    early_exit_engine = get_inference_engine('speculative', shard_downloader, early_exit_config)
    print(f"Created early exit engine: {early_exit_engine.__class__.__name__}")
    print(f"Using early exit: {early_exit_engine.use_early_exit}")
    
    # Test inference with early exit
    test_shard = Shard(
        model_id="dummy", 
        start_layer=0, 
        end_layer=7, 
        n_layers=8
    )
    
    tokens = await early_exit_engine.encode(test_shard, "Test early exit")
    input_data = tokens.reshape(1, -1)
    
    start_time = time.perf_counter()
    output, state = await early_exit_engine.infer_tensor(
        "early_exit_test", 
        test_shard, 
        input_data
    )
    end_time = time.perf_counter()
    
    print(f"Early exit inference completed in {(end_time - start_time)*1000:.2f}ms")
    print(f"Output shape: {output.shape}")
    
    print("✅ Early exit mode test passed!")
    
    return True


async def test_integration_with_exo_workflow():
    """Test integration with exo's typical workflow."""
    print("\n\n9. Testing integration with exo workflow...")
    
    # Simulate the typical exo inference workflow
    speculative_config = {
        'target_engine_name': 'dummy',
        'draft_engine_name': 'dummy',
        'gamma': 5,
        'temperature': 0.7,
        'top_k_threshold': 0.8,
        'lenience': 1.1,
        'enable_speculative': True,
    }
    
    shard_downloader = NoopShardDownloader()
    engine = get_inference_engine('speculative', shard_downloader, speculative_config)
    
    # Test with different shard configurations
    shards = [
        Shard(model_id="dummy", start_layer=0, end_layer=3, n_layers=8),  # First shard
        Shard(model_id="dummy", start_layer=4, end_layer=7, n_layers=8),  # Last shard
    ]
    
    for i, shard in enumerate(shards):
        print(f"\nTesting shard {i+1}: layers {shard.start_layer}-{shard.end_layer}")
        print(f"Is last layer: {shard.is_last_layer()}")
        
        # Test prompt inference
        if shard.is_first_layer():
            tokens, state = await engine.infer_prompt(
                f"test_prompt_{i}", 
                shard, 
                "This is a test prompt"
            )
            print(f"Prompt inference output shape: {tokens.shape}")
        
        # Test tensor inference
        dummy_input = np.array([[1, 2, 3, 4, 5]])
        output, state = await engine.infer_tensor(
            f"test_tensor_{i}", 
            shard, 
            dummy_input
        )
        print(f"Tensor inference output shape: {output.shape}")
        
        # Check if speculative decoding was used (only on last layer)
        if shard.is_last_layer():
            stats = engine.get_speculation_stats()
            print(f"Speculation was used: {stats['total_speculations'] > 0}")
    
    print("✅ Integration with exo workflow test passed!")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting Speculative Decoding Tests for exo\n")
    
    try:
        # Run all tests
        await test_speculative_inference_engine()
        await test_early_exit_mode()
        await test_integration_with_exo_workflow()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo use speculative decoding with exo, run:")
        print("  exo --inference-engine speculative --speculative-target-engine mlx --speculative-gamma 5")
        print("  # or for early exit mode:")
        print("  exo --inference-engine speculative --speculative-target-engine mlx --speculative-draft-engine None")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 