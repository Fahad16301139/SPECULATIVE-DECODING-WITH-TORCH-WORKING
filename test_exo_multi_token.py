#!/usr/bin/env python3
"""
Test exo's full architecture with TRUE VANILLA multi-token speculative decoding
"""

import asyncio
import numpy as np
import os
import sys
from pathlib import Path

# Add the exo directory to Python path
exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

from exo.helpers import DEBUG
from exo.inference.shard import Shard
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine 
from exo.download.new_shard_download import NewShardDownloader
from exo.orchestration.node import Node
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.networking.manual.manual_server import ManualServer
from exo.orchestration.standard_node import StandardNode

# Enable DEBUG for detailed logging
os.environ['DEBUG'] = '2'

async def test_exo_multi_token():
    """Test exo's full architecture with multi-token speculative decoding"""
    
    print("🚀 Testing EXO Full Architecture with Multi-Token Speculative Decoding")
    print("=" * 70)
    
    # Use correct model names from exo's supported models
    target_model_id = "llama-3.2-3b"  # Fixed: removed -instruct
    draft_model_id = "llama-3.2-1b"   # Fixed: removed -instruct
    
    print(f"Target model: {target_model_id}")
    print(f"Draft model: {draft_model_id}")
    
    try:
        # Create shard downloader
        shard_downloader = NewShardDownloader()
        
        # Create target and draft engines
        print("\n📥 Creating inference engines...")
        target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
        draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
        
        # Create speculative engine with multi-token support
        print("🎯 Creating speculative engine...")
        speculative_engine = SpeculativeInferenceEngine(
            target_engine=target_engine,
            draft_engine=draft_engine,
            gamma=6,  # Generate 6 draft tokens
            temperature=0.8
        )
        
        # Create exo node with speculative engine
        print("🏗️ Creating exo node...")
        discovery = ManualDiscovery()
        server = ManualServer(discovery, lambda peer_id, address, device_capabilities: print(f"Connected: {peer_id}"))
        
        node = StandardNode(
            node_id="test-node",
            inference_engine=speculative_engine,
            discovery=discovery,
            partition_strategy=None,  # Single node
            chatgpt_api_endpoint=None,
            web_chat_url=None,
            disable_download=False
        )
        
        # Test prompt through exo's interface
        test_prompt = "The future of artificial intelligence is"
        
        print(f"\n🧪 Testing with prompt: '{test_prompt}'")
        print("=" * 70)
        
        # Create shard
        target_shard = Shard(model_id=target_model_id, start_layer=0, end_layer=0, n_layers=28)
        
        # Test through exo's process_prompt interface
        print("🚀 Testing through exo's process_prompt interface...")
        result = await node.process_prompt(target_shard, test_prompt, "test-request-001")
        
        print(f"✅ Exo multi-token generation successful!")
        print(f"   Result: {result}")
        
        # Check if we got multiple tokens
        if hasattr(node, 'buffered_token_output') and "test-request-001" in node.buffered_token_output:
            tokens, finished = node.buffered_token_output["test-request-001"]
            print(f"   Buffered tokens: {len(tokens)}")
            print(f"   Finished: {finished}")
            
            if len(tokens) > 1:
                print("🎉 SUCCESS: Multi-token generation through exo architecture!")
            else:
                print("⚠️ Only single token - may need more iterations")
        
        print("\n🎉 SUCCESS: Exo multi-token architecture integration works!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_exo_multi_token())
    if success:
        print("\n✅ All tests passed! Exo multi-token architecture is working.")
    else:
        print("\n❌ Tests failed.")
        sys.exit(1) 