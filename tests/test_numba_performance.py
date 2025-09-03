#!/usr/bin/env python3
"""
Performance test script to compare numba vs pure Python implementation
"""

import time
import numpy as np
from hawkes_conditional_law import PointProcessCondLaw, NUMBA_AVAILABLE

def create_test_data(n_events_i=10000, n_events_j=5000, T=1000.0):
    """Create synthetic test data for performance testing"""
    # Generate timestamps
    timestamps_i = np.sort(np.random.uniform(0, T, n_events_i))
    timestamps_j = np.sort(np.random.uniform(0, T, n_events_j))
    
    # Generate cumulative marks (marks_j)
    marks_j = np.cumsum(np.random.exponential(1.0, n_events_j))
    
    # Create lag array
    lags = np.logspace(-3, 1, 50)  # 49 lag bins
    
    # Mark bounds
    mark_min = 0.5
    mark_max = 2.0
    
    # Lambda_i (baseline intensity)
    lambda_i = n_events_i / T
    
    return timestamps_i, timestamps_j, marks_j, lags, mark_min, mark_max, T, lambda_i

def benchmark_performance():
    """Benchmark numba vs pure Python performance"""
    print("🚀 Numba Performance Test")
    print("=" * 50)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    
    # Create test data
    print("Generating test data...")
    test_data = create_test_data()
    timestamps_i, timestamps_j, marks_j, lags, mark_min, mark_max, T, lambda_i = test_data
    
    print(f"Events in i: {len(timestamps_i)}")
    print(f"Events in j: {len(timestamps_j)}")
    print(f"Lag bins: {len(lags)-1}")
    print(f"Time span: {T}")
    
    # Prepare output arrays
    claw_X = np.zeros(len(lags) - 1)
    claw_Y = np.zeros(len(lags) - 1)
    
    # Warm up numba compilation (first call is slow due to compilation)
    if NUMBA_AVAILABLE:
        print("\nWarming up numba compilation...")
        small_data = create_test_data(100, 50, 10.0)
        small_claw_X = np.zeros(len(small_data[3]) - 1)
        small_claw_Y = np.zeros(len(small_data[3]) - 1)
        PointProcessCondLaw(*small_data, small_claw_X, small_claw_Y)
        print("Numba compilation completed!")
    
    # Benchmark with numba
    print(f"\nRunning benchmark with numba...")
    times_numba = []
    for i in range(5):
        claw_Y.fill(0.0)
        start_time = time.perf_counter()
        PointProcessCondLaw(*test_data, claw_X, claw_Y)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times_numba.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    avg_numba = np.mean(times_numba)
    std_numba = np.std(times_numba)
    
    # Test fallback (temporarily disable numba)
    print(f"\nRunning benchmark with pure Python fallback...")
    import hawkes_conditional_law
    original_numba_available = hawkes_conditional_law.NUMBA_AVAILABLE
    hawkes_conditional_law.NUMBA_AVAILABLE = False
    
    times_python = []
    for i in range(3):  # Fewer runs since it's slower
        claw_Y.fill(0.0)
        start_time = time.perf_counter()
        PointProcessCondLaw(*test_data, claw_X, claw_Y)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times_python.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    # Restore numba availability
    hawkes_conditional_law.NUMBA_AVAILABLE = original_numba_available
    
    avg_python = np.mean(times_python)
    std_python = np.std(times_python)
    
    # Results
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Numba (JIT):      {avg_numba:.4f}s ± {std_numba:.4f}s")
    print(f"Pure Python:      {avg_python:.4f}s ± {std_python:.4f}s")
    
    if avg_python > 0:
        speedup = avg_python / avg_numba
        print(f"Speedup:          {speedup:.1f}x faster with numba!")
    
    print(f"Time saved:       {avg_python - avg_numba:.4f}s per call")
    
    return avg_numba, avg_python

if __name__ == "__main__":
    benchmark_performance()
