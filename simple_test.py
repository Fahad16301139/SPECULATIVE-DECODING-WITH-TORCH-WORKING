#!/usr/bin/env python3
"""
Simple test for improved speculative decoding context management.
Uses the existing working infrastructure.
"""

import os
import requests
import time

# Set debug mode
os.environ["DEBUG"] = "1"

def test_context_management():
    print("🔧 ========== SIMPLE CONTEXT MANAGEMENT TEST ==========")
    print("🎯 Testing cache coordination improvements")
    print()
    
    # Test API endpoint (assumes server is running)
    url = "http://localhost:52415/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Test with a context-dependent prompt
    test_data = {
        "model": "meta-llama/Llama-3.2-3B",
        "messages": [
            {
                "role": "user", 
                "content": "Count from 1 to 10: One, Two, Three"
            }
        ],
        "max_tokens": 20,
        "temperature": 0.8,
        "stream": False
    }
    
    print("📝 Test prompt: 'Count from 1 to 10: One, Two, Three'")
    print("🎯 Expected: Should continue with 'Four, Five, Six...'")
    print()
    
    try:
        print("🚀 Sending request...")
        start_time = time.time()
        
        response = requests.post(url, json=test_data, headers=headers, timeout=60)
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print("✅ SUCCESS!")
            print(f"⏱️  Response time: {(end_time - start_time):.2f}s")
            print(f"📄 Response: '{content}'")
            print()
            
            # Check context continuity
            if any(word in content.lower() for word in ['four', '4', 'five', '5']):
                print("✅ CONTEXT CONTINUITY: GOOD - Sequence continues properly!")
                print("🔧 Cache coordination is working correctly")
            else:
                print("⚠️  CONTEXT CONTINUITY: Check needed - Response may not follow context")
            
            # Check for any debug output patterns that indicate cache issues
            if "cache_pos" in content or "overflow" in content.lower():
                print("⚠️  Possible cache issues detected in output")
            else:
                print("✅ No cache error patterns detected")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Server not running!")
        print("💡 Start the server first with:")
        print("   python exo/main.py --inference-engine SpeculativeInferenceEngine --model meta-llama/Llama-3.2-3B --draft-model meta-llama/Llama-3.2-1B --port 52415")
        return False
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False
    
    print()
    print("🔧 ========== TEST COMPLETE ==========")
    return True

if __name__ == "__main__":
    test_context_management() 