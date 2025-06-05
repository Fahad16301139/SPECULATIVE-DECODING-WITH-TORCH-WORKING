#!/usr/bin/env python3
"""
Minimal test to verify our speculative decoding implementation.
This will help us debug any remaining issues.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_first_true_index_vanilla(bool_tensor, dim=-1):
    """Exact vanilla implementation: (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)"""
    cumsum = np.cumsum(bool_tensor, axis=dim)
    return np.sum(cumsum == 0, axis=dim)

def test_find_first_true_index():
    """Test our find_first_true_index implementation"""
    print("Testing find_first_true_index implementation...")
    
    # Test case 1: [False, False, True, False, True] -> should return 2
    test1 = np.array([[False, False, True, False, True]])
    result1 = find_first_true_index_vanilla(test1, dim=-1)
    print(f"Test 1: {test1[0]} -> {result1[0]} (expected: 2)")
    
    # Test case 2: [False, False, False, False, False] -> should return 5
    test2 = np.array([[False, False, False, False, False]])
    result2 = find_first_true_index_vanilla(test2, dim=-1)
    print(f"Test 2: {test2[0]} -> {result2[0]} (expected: 5)")
    
    # Test case 3: [True, False, False, False, False] -> should return 0
    test3 = np.array([[True, False, False, False, False]])
    result3 = find_first_true_index_vanilla(test3, dim=-1)
    print(f"Test 3: {test3[0]} -> {result3[0]} (expected: 0)")
    
    # Test case 4: Batch of 2
    test4 = np.array([
        [False, False, True, False, True],  # -> 2
        [False, True, False, False, False]  # -> 1
    ])
    result4 = find_first_true_index_vanilla(test4, dim=-1)
    print(f"Test 4 batch: {result4} (expected: [2, 1])")

def test_acceptance_logic():
    """Test the acceptance/rejection logic"""
    print("\nTesting acceptance/rejection logic...")
    
    # Create simple test case
    batch_size = 1
    gamma = 3
    vocab_size = 5
    
    # Mock probabilities
    p = np.array([0.7, 0.2, 0.1])  # Target model probs for sampled tokens
    q = np.array([0.5, 0.3, 0.2])  # Draft model probs for sampled tokens
    
    # Mock random values
    r = np.array([0.5, 0.8, 0.3])  # Random values
    
    # Vanilla condition: r > (p / q)
    ratios = p / q
    rejection_condition = r > ratios
    
    print(f"p: {p}")
    print(f"q: {q}")
    print(f"p/q ratios: {ratios}")
    print(f"r: {r}")
    print(f"r > (p/q): {rejection_condition}")
    
    # Find first true index
    accepted = find_first_true_index_vanilla(rejection_condition.reshape(1, -1), dim=-1)
    print(f"Accepted tokens: {accepted[0]}")

if __name__ == "__main__":
    test_find_first_true_index()
    test_acceptance_logic() 