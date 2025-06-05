#!/usr/bin/env python3
"""
Simple test to debug TinyGrad Linear layer behavior
"""

import sys
from pathlib import Path
exo_path = Path(__file__).parent
sys.path.insert(0, str(exo_path))

# Import TinyGrad
from tinygrad.tensor import Tensor
from tinygrad import nn

print("🔍 DEBUGGING TINYGRAD LINEAR LAYER")
print("=" * 50)

# Test case 1: Simple Linear layer
print("\n📊 Test 1: Simple Linear Layer")
linear = nn.Linear(3072, 128256, bias=False)
print(f"Weight shape: {linear.weight.shape}")

# Create test input
x = Tensor.ones(1, 5, 3072)  # (batch, seq, hidden)
print(f"Input shape: {x.shape}")

# Forward pass
output = linear(x)
print(f"Output shape: {output.shape}")
print(f"Expected: (1, 5, 128256)")

if tuple(output.shape) == (1, 5, 128256):
    print("✅ Linear layer working correctly!")
else:
    print("❌ Linear layer BUG detected!")
    
    # Debug the internal operations
    print(f"\n🔧 Debugging internal operations:")
    print(f"Weight original: {linear.weight.shape}")
    weight_transposed = linear.weight.transpose()
    print(f"Weight transposed: {weight_transposed.shape}")
    
    # Manual dot product
    manual_output = x.dot(weight_transposed)
    print(f"Manual x.dot(weight.T): {manual_output.shape}")
    
    # Check what linear actually does
    linear_output = x.linear(linear.weight.transpose(), None)
    print(f"x.linear(weight.T, None): {linear_output.shape}")

print("\n" + "=" * 50) 