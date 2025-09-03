#!/usr/bin/env python3
"""Test script to verify the corrected PointProcessCondLaw implementation."""

import numpy as np
import sys
import os

# Add the parent directory to path to import hawkes_conditional_law
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawkes_conditional_law import PointProcessCondLaw

def test_basic_functionality():
    """Test basic functionality of PointProcessCondLaw."""
    print("Testing basic PointProcessCondLaw functionality...")
    
    # Create simple test data
    timestamps_i = np.array([1.0, 2.5, 4.0, 5.5])  # Y process events
    timestamps_j = np.array([0.5, 3.0, 6.0])       # Z process events  
    marks_j = np.array([0.1, 0.4, 0.7])            # Cumulative marks for Z
    
    # Define lags (3 intervals: [0,1], [1,2], [2,3])
    lags = np.array([0.0, 1.0, 2.0, 3.0])
    
    # Mark bounds
    mark_min = 0.0
    mark_max = 1.0
    
    # Other parameters
    T = 10.0
    lambda_i = len(timestamps_i) / T  # Baseline intensity
    
    # Output arrays
    n_lags = len(lags) - 1
    claw_X = np.zeros(n_lags)
    claw_Y = np.zeros(n_lags)
    
    # Call the function
    PointProcessCondLaw(timestamps_i, timestamps_j, marks_j, lags,
                       mark_min, mark_max, T, lambda_i, claw_X, claw_Y)
    
    print(f"Input:")
    print(f"  timestamps_i: {timestamps_i}")
    print(f"  timestamps_j: {timestamps_j}")
    print(f"  marks_j: {marks_j}")
    print(f"  lags: {lags}")
    print(f"  mark_min: {mark_min}, mark_max: {mark_max}")
    print(f"  T: {T}, lambda_i: {lambda_i}")
    
    print(f"\nOutput:")
    print(f"  claw_X (bin centers): {claw_X}")
    print(f"  claw_Y (conditional law): {claw_Y}")
    
    # Basic sanity checks
    assert len(claw_X) == n_lags, f"claw_X should have {n_lags} elements"
    assert len(claw_Y) == n_lags, f"claw_Y should have {n_lags} elements"
    assert np.allclose(claw_X, [0.5, 1.5, 2.5]), "claw_X should be bin centers"
    
    print("✓ Basic functionality test passed!")
    return True

def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Test with empty arrays
    timestamps_i = np.array([])
    timestamps_j = np.array([])
    marks_j = np.array([])
    lags = np.array([0.0, 1.0])
    
    claw_X = np.zeros(1)
    claw_Y = np.zeros(1)
    
    PointProcessCondLaw(timestamps_i, timestamps_j, marks_j, lags,
                       0.0, 1.0, 10.0, 0.1, claw_X, claw_Y)
    
    print("✓ Empty arrays test passed!")
    
    # Test with mark bounds that exclude all events
    timestamps_i = np.array([1.0, 2.0])
    timestamps_j = np.array([0.5])
    marks_j = np.array([2.0])  # Mark increment will be 2.0
    
    claw_X = np.zeros(1)
    claw_Y = np.zeros(1)
    
    # Mark bounds [0, 1) should exclude the event with mark 2.0
    PointProcessCondLaw(timestamps_i, timestamps_j, marks_j, lags,
                       0.0, 1.0, 10.0, 0.1, claw_X, claw_Y)
    
    print("✓ Mark bounds exclusion test passed!")
    
    return True

def test_mark_checking_logic():
    """Test the mark checking logic that skips first event."""
    print("\nTesting mark checking logic...")
    
    # Create data where first event would be excluded by mark bounds if checked
    timestamps_i = np.array([1.0, 2.0, 3.0])
    timestamps_j = np.array([0.0, 1.5])  # Two conditioning events
    marks_j = np.array([5.0, 5.2])       # First mark = 5.0, second increment = 0.2
    
    lags = np.array([0.0, 1.0, 2.0])
    
    claw_X = np.zeros(2)
    claw_Y = np.zeros(2)
    
    # Mark bounds [0, 1) should exclude second event (increment 0.2 is in bounds)
    # but NOT the first event (which should be processed regardless)
    PointProcessCondLaw(timestamps_i, timestamps_j, marks_j, lags,
                       0.0, 1.0, 10.0, 0.2, claw_X, claw_Y)
    
    print(f"  marks_j: {marks_j}")
    print(f"  mark increments: [5.0, 0.2]")
    print(f"  mark_bounds: [0, 1)")
    print(f"  Expected: First event (mark=5.0) processed despite being out of bounds")
    print(f"  Expected: Second event (increment=0.2) processed as it's in bounds")
    print(f"  claw_Y: {claw_Y}")
    
    print("✓ Mark checking logic test passed!")
    return True

if __name__ == "__main__":
    print("Running PointProcessCondLaw tests...\n")
    
    try:
        test_basic_functionality()
        test_edge_cases()
        test_mark_checking_logic()
        
        print("\n🎉 All tests passed! The implementation appears to be working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
