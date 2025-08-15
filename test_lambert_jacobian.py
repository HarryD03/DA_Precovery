#!/usr/bin/env python3
"""
Utility functions for comparing Jacobians between DA automatic differentiation
and finite difference methods for Lambert solver validation.
"""

import numpy as np
from scipy.optimize import approx_fprime
from poliastro.iod import izzo
from utils.lambert_izzo import lambert_izzo
from daceypy import DA, array

def compare_lambert_jacobians(R1, R2, DT, MU, M=0, prograde=True, h=1e-6, tolerance=1e-4):
    """
    Compare Jacobians of Lambert solver velocities between DA and finite differences.
    
    Args:
        R1: Initial position vector [km]
        R2: Final position vector [km]
        DT: Time of flight [s]
        MU: Gravitational parameter [km³/s²]
        M: Number of revolutions
        prograde: Direction of motion
        h: Step size for finite differences
        tolerance: Tolerance for Jacobian comparison
    
    Returns:
        dict: Results containing Jacobians, errors, and comparison status
    """
    
    # Initialize DA system
    NVar = len(R1) + len(R2)  # 6 variables (R1x, R1y, R1z, R2x, R2y, R2z)
    DA.init(4, NVar)
    
    # Create DA position vectors
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])
    
    # Solve Lambert problem with DA
    print("Computing DA Lambert solution...")
    velocities_da = lambert_izzo(R1_DA, R2_DA, DT, MU, M, prograde=prograde)
    solution_da = velocities_da[0]
    v1_da = solution_da[:, 0]
    v2_da = solution_da[:, 1]
    
    # Extract DA Jacobians
    print("Extracting DA Jacobians...")
    v1_jacobian_da = np.zeros((3, 6))
    v2_jacobian_da = np.zeros((3, 6))
    
    for i in range(3):  # Velocity components (x, y, z)
        for j in range(6):  # Position components (R1x, R1y, R1z, R2x, R2y, R2z)
            v1_jacobian_da[i, j] = v1_da[i].deriv(j+1)
            v2_jacobian_da[i, j] = v2_da[i].deriv(j+1)
    
    # Define wrapper for finite difference computation
    def lambert_wrapper(positions):
        """Wrapper for reference Lambert solver"""
        r1_fd = positions[:3]
        r2_fd = positions[3:6]
        v1_fd, v2_fd = izzo(MU, r1_fd, r2_fd, DT, M, prograde=prograde, lowpath=False, numiter=30, rtol=1e-9)
        return np.concatenate([v1_fd, v2_fd])
    
    # Compute finite difference Jacobian
    print("Computing finite difference Jacobian...")
    pos_vector = np.concatenate([R1, R2])
    
    jac_fd = np.zeros((6, 6))
    for j in range(6):
        def func_j(pos):
            return lambert_wrapper(pos)[j]
        jac_fd[j, :] = approx_fprime(pos_vector, func_j, h)
    
    # Split finite difference Jacobian
    v1_jacobian_fd = jac_fd[:3, :]  # First 3 rows (v1)
    v2_jacobian_fd = jac_fd[3:, :]  # Last 3 rows (v2)
    
    # Compare Jacobians
    v1_error = np.abs(v1_jacobian_da - v1_jacobian_fd)
    v2_error = np.abs(v2_jacobian_da - v2_jacobian_fd)
    
    max_v1_error = np.max(v1_error)
    max_v2_error = np.max(v2_error)
    max_total_error = max(max_v1_error, max_v2_error)
    
    # Check accuracy
    v1_accurate = max_v1_error < tolerance
    v2_accurate = max_v2_error < tolerance
    overall_accurate = v1_accurate and v2_accurate
    
    # Results
    results = {
        'v1_jacobian_da': v1_jacobian_da,
        'v2_jacobian_da': v2_jacobian_da,
        'v1_jacobian_fd': v1_jacobian_fd,
        'v2_jacobian_fd': v2_jacobian_fd,
        'v1_error': v1_error,
        'v2_error': v2_error,
        'max_v1_error': max_v1_error,
        'max_v2_error': max_v2_error,
        'max_total_error': max_total_error,
        'v1_accurate': v1_accurate,
        'v2_accurate': v2_accurate,
        'overall_accurate': overall_accurate,
        'tolerance': tolerance,
        'v1_nominal': v1_da[0].cons(),  # Nominal values for reference
        'v2_nominal': v2_da[0].cons()
    }
    
    return results

def print_jacobian_comparison(results):
    """
    Print detailed comparison results from compare_lambert_jacobians.
    
    Args:
        results: Dictionary returned by compare_lambert_jacobians
    """
    
    print("\n=== LAMBERT JACOBIAN COMPARISON ===")
    
    print(f"\nNominal velocities:")
    print(f"v1 = {results['v1_nominal']}")
    print(f"v2 = {results['v2_nominal']}")
    
    print(f"\nDA Jacobian for v1 (∂v1/∂[R1,R2]):")
    print(results['v1_jacobian_da'])
    
    print(f"\nFinite Difference Jacobian for v1:")
    print(results['v1_jacobian_fd'])
    
    print(f"\nDA Jacobian for v2 (∂v2/∂[R1,R2]):")
    print(results['v2_jacobian_da'])
    
    print(f"\nFinite Difference Jacobian for v2:")
    print(results['v2_jacobian_fd'])
    
    print(f"\nError Analysis:")
    print(f"Max error in ∂v1/∂[R1,R2]: {results['max_v1_error']:.2e}")
    print(f"Max error in ∂v2/∂[R1,R2]: {results['max_v2_error']:.2e}")
    print(f"Max total error: {results['max_total_error']:.2e}")
    print(f"Tolerance: {results['tolerance']:.1e}")
    
    print(f"\nAccuracy Assessment:")
    print(f"v1 Jacobian accurate: {results['v1_accurate']}")
    print(f"v2 Jacobian accurate: {results['v2_accurate']}")
    print(f"Overall accurate: {results['overall_accurate']}")
    
    if results['overall_accurate']:
        print("✅ DA Jacobians match finite differences within tolerance")
    else:
        print("❌ DA Jacobians differ from finite differences")
        print("   This indicates potential issues with DA automatic differentiation")
        
        # Identify worst errors
        v1_worst_idx = np.unravel_index(np.argmax(results['v1_error']), results['v1_error'].shape)
        v2_worst_idx = np.unravel_index(np.argmax(results['v2_error']), results['v2_error'].shape)
        
        print(f"\nWorst v1 error: {results['v1_error'][v1_worst_idx]:.2e} at element [{v1_worst_idx[0]}, {v1_worst_idx[1]}]")
        print(f"Worst v2 error: {results['v2_error'][v2_worst_idx]:.2e} at element [{v2_worst_idx[0]}, {v2_worst_idx[1]}]")

def test_lambert_jacobian_accuracy():
    """
    Test function to validate Lambert solver Jacobian accuracy
    """
    
    # Test data (Vallado Example 5.2)
    R1 = np.array([15945.34, 0.0, 0.0])
    R2 = np.array([12214.83399, 10249.46731, 0.0])
    DT = 76 * 60  # seconds
    MU = 3.986e5  # km³/s²
    
    print("Testing Lambert Jacobian accuracy...")
    print(f"R1 = {R1}")
    print(f"R2 = {R2}")
    print(f"DT = {DT} s")
    print(f"MU = {MU} km³/s²")
    
    # Compare Jacobians
    results = compare_lambert_jacobians(R1, R2, DT, MU, M=0, prograde=True)
    
    # Print results
    print_jacobian_comparison(results)
    
    return results

if __name__ == "__main__":
    # Run the test
    results = test_lambert_jacobian_accuracy()
    
    # Summary
    print(f"\n=== SUMMARY ===")
    if results['overall_accurate']:
        print("✅ Lambert solver DA Jacobians are accurate")
    else:
        print("❌ Lambert solver DA Jacobians have errors")
        print("   This could explain Newton divergence in DAIOD algorithms")
        print(f"   Maximum error: {results['max_total_error']:.2e}")
