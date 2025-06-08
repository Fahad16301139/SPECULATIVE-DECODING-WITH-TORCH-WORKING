#!/usr/bin/env python3

import asyncio
from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader

async def debug_executor_sharing():
    print("🔍 DEBUGGING EXECUTOR SHARING")
    print("=" * 50)
    
    shard_downloader = NewShardDownloader()
    target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    draft_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    
    print(f"Target engine executor ID: {id(target_engine.executor)}")
    print(f"Draft engine executor ID: {id(draft_engine.executor)}")
    print(f"Are they the same executor? {target_engine.executor is draft_engine.executor}")
    
    if target_engine.executor is draft_engine.executor:
        print("🚨 CRITICAL: Both engines share the same executor!")
        print("   This means they're running on the same thread and may interfere!")
    else:
        print("✅ Different executors - good isolation")

if __name__ == "__main__":
    asyncio.run(debug_executor_sharing()) 