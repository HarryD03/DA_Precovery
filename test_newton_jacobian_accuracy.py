#!/usr/bin/env python3

import numpy as np
import pickle
from daceypy import DA, array
import daceypy.op as op
from scipy.optimize import fsolve, approx_fprime
from utils.iod import newton_nominal_DAVec, DAIOD_1, DAIOD_1Scipy
from utils.lambert_izzo import lambert_izzo, householder_iter_DA_Map
from utils.observation import observing_station_vectors
from utils.data import sample_earth_data

def test_newton_jacobian_accuracy():
    """
    Test if the Newton Jacobian computed using DA is accurate
    by comparing with finite difference Jacobian
    """
    
    print("=== TESTING NEWTON JACOBIAN ACCURACY IN DAIOD ===\n")
    
    # Get sample test data (same as in test_iod.py)
    sample_ra, sample_dec, time_ref = sample_earth_data()
    
    # Select three observations
    obs_indices = [0, 10, 20]
    ra_sample = sample_ra[obs_indices]
    dec_sample = sample_dec[obs_indices]
    time_sample = [time_ref[i] for i in obs_indices]
    
    print(f"Using observations at indices: {obs_indices}")
    print(f"RA: {ra_sample}")
    print(f"DEC: {dec_sample}")
    
    # Get observing station vectors
    station_vectors = observing_station_vectors(time_sample)
    print(f"Station vectors computed")
    
    # Get initial Gauss solution (this should work)
    from utils.iod import gauss_iod_angles
    
    try:
        r2_gauss, v2_gauss = gauss_iod_angles(
            ra_sample, dec_sample, time_sample, station_vectors
        )
        print(f"Gauss initial solution:")
        print(f"  r2 = {r2_gauss.flatten()}")
        print(f"  v2 = {v2_gauss.flatten()}")
        
    except Exception as e:
        print(f"Gauss solution failed: {e}")
        return None, None
    
    # Get initial range magnitudes from Gauss solution
    from utils.iod import create_da_los_vectors
    los_vectors = create_da_los_vectors(ra_sample, dec_sample)
    
    # Estimate initial range magnitudes
    r_mag_init = np.array([1000.0, 1100.0, 1200.0])  # km
    
    print(f"\nInitial range magnitude estimates: {r_mag_init}")
    
    # Define the DAIOD function for testing
    def daiod_function(range_mags):
        """
        The function that DAIOD tries to zero
        Returns velocity differences between Lambert and observations
        """
        try:
            # Create position vectors from range magnitudes
            positions = []
            for i, r_mag in enumerate(range_mags):
                r_vec = station_vectors[i] + r_mag * los_vectors[i].cons()
                positions.append(r_vec)
            
            # Solve Lambert problem between consecutive positions
            dt1 = (time_sample[1] - time_sample[0]).total_seconds()
            dt2 = (time_sample[2] - time_sample[1]).total_seconds()
            
            # Lambert solution 1->2
            velocities_12 = lambert_izzo(positions[0], positions[1], dt1, 398600.4418, 0)
            v1_lambert, v2_lambert = velocities_12[0][:, 0], velocities_12[0][:, 1]
            
            # Lambert solution 2->3
            velocities_23 = lambert_izzo(positions[1], positions[2], dt2, 398600.4418, 0)
            v2_lambert_23, v3_lambert = velocities_23[0][:, 0], velocities_23[0][:, 1]
            
            # Velocity consistency constraint
            velocity_diff = v2_lambert - v2_lambert_23
            
            return velocity_diff.flatten()
            
        except Exception as e:
            print(f"Error in DAIOD function: {e}")
            return np.array([1e6, 1e6, 1e6])  # Large error if Lambert fails
    
    print(f"\n=== STEP 1: EVALUATE FUNCTION AT INITIAL GUESS ===")
    
    f_initial = daiod_function(r_mag_init)
    print(f"Function value at initial guess: {f_initial}")
    print(f"Function norm: {np.linalg.norm(f_initial)}")
    
    print(f"\n=== STEP 2: COMPUTE DA JACOBIAN ===")
    
    # Method 1: DA-based Jacobian (current DAIOD approach)
    DA.init(2, 3)  # Order 2, 3 variables
    
    try:
        # Create DA variables
        r_mag_da = array([r_mag_init[i] + DA(i+1) for i in range(3)])
        
        # Evaluate function with DA
        f_da = daiod_function(r_mag_da)
        
        # Extract Jacobian from DA linear terms
        jacobian_da = np.zeros((3, 3))
        if hasattr(f_da, '__len__'):
            for i in range(3):
                if hasattr(f_da[i], 'linear'):
                    jacobian_da[i, :] = f_da[i].linear()
                else:
                    # If DA object doesn't have linear method, try deriv
                    for j in range(3):
                        if hasattr(f_da[i], 'deriv'):
                            jacobian_da[i, j] = f_da[i].deriv(j+1)
        
        print(f"DA Jacobian computed successfully:")
        print(jacobian_da)
        
        # Check if Jacobian is reasonable
        jacobian_condition = np.linalg.cond(jacobian_da)
        print(f"DA Jacobian condition number: {jacobian_condition:.2e}")
        
    except Exception as e:
        print(f"DA Jacobian computation failed: {e}")
        jacobian_da = None
    
    print(f"\n=== STEP 3: COMPUTE FINITE DIFFERENCE JACOBIAN ===")
    
    # Method 2: Finite Difference Jacobian (scipy.optimize approach)
    h = 1e-6  # Step size for finite differences
    
    try:
        jacobian_fd = np.zeros((3, 3))
        
        # Compute each column of Jacobian via finite differences
        for j in range(3):
            r_mag_plus = r_mag_init.copy()
            r_mag_minus = r_mag_init.copy()
            r_mag_plus[j] += h
            r_mag_minus[j] -= h
            
            f_plus = daiod_function(r_mag_plus)
            f_minus = daiod_function(r_mag_minus)
            
            jacobian_fd[:, j] = (f_plus - f_minus) / (2 * h)
        
        print(f"Finite Difference Jacobian computed successfully:")
        print(jacobian_fd)
        
        # Check if Jacobian is reasonable
        jacobian_fd_condition = np.linalg.cond(jacobian_fd)
        print(f"FD Jacobian condition number: {jacobian_fd_condition:.2e}")
        
    except Exception as e:
        print(f"Finite Difference Jacobian computation failed: {e}")
        jacobian_fd = None
    
    print(f"\n=== STEP 4: COMPARE JACOBIANS ===")
    
    if jacobian_da is not None and jacobian_fd is not None:
        # Compare Jacobians
        error_matrix = np.abs(jacobian_da - jacobian_fd)
        max_error = np.max(error_matrix)
        relative_error = max_error / (np.max(np.abs(jacobian_fd)) + 1e-15)
        
        print(f"Jacobian comparison:")
        print(f"  Maximum absolute error: {max_error:.2e}")
        print(f"  Maximum relative error: {relative_error:.2e}")
        print(f"  Error matrix:")
        print(error_matrix)
        
        # Check if they match within tolerance
        tolerance = 1e-4  # Reasonable tolerance for numerical derivatives
        if max_error < tolerance:
            print(f"✅ DA and FD Jacobians match (within {tolerance:.1e})")
            print("   DA higher-order terms appear correct")
        else:
            print(f"❌ DA and FD Jacobians differ significantly")
            print("   This confirms DA higher-order terms are causing Newton divergence!")
            
        # Test Newton step predictions
        print(f"\n=== STEP 5: TEST NEWTON STEP PREDICTIONS ===")
        
        # Compute Newton steps using both Jacobians
        try:
            newton_step_da = -np.linalg.solve(jacobian_da, f_initial)
            newton_step_fd = -np.linalg.solve(jacobian_fd, f_initial)
            
            print(f"Newton step with DA Jacobian: {newton_step_da}")
            print(f"Newton step with FD Jacobian: {newton_step_fd}")
            
            step_difference = np.linalg.norm(newton_step_da - newton_step_fd)
            print(f"Difference in Newton steps: {step_difference:.2e}")
            
            if step_difference > 1.0:  # If steps differ by more than 1 km
                print("❌ Newton steps differ significantly - this explains divergence!")
            else:
                print("✅ Newton steps are similar")
                
        except np.linalg.LinAlgError as e:
            print(f"Newton step computation failed: {e}")
    
    return jacobian_da, jacobian_fd


def test_scipy_vs_newton_comparison():
    """
    Direct comparison between DAIOD_1Scipy (which works) and DAIOD_1 (which fails)
    Uses the same test data as the working test case
    """
    
    print(f"\n\n=== SCIPY vs NEWTON DA COMPARISON ===\n")
    
    # Use the exact same test setup as in test_iod.py
    from tests.test_iod import sample_data_Earth
    
    # Get the test data
    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth()
    
    print(f"Test data loaded:")
    print(f"  Initial range magnitudes: {range_mag}")
    print(f"  Observer positions shape: {pos_obs.shape}")
    print(f"  Observation directions shape: {obs_dir.shape}")
    print(f"  Times: {t}")
    
    # Test both methods on the same problem
    print("\nTesting both methods on the same DAIOD problem...")
    
    try:
        # Method 1: DAIOD_1Scipy (known to work)
        print("\n--- Testing DAIOD_1Scipy ---")
        result_scipy = DAIOD_1Scipy(range_mag, obs_dir, t, 4, pos_obs, mu=3.986e5, tol=1e-3)
        print(f"DAIOD_1Scipy result: SUCCESS")
        print(f"Final range magnitudes (constant): {result_scipy.cons()}")
        
    except Exception as e:
        print(f"DAIOD_1Scipy failed: {e}")
        result_scipy = None
    
    try:
        # Method 2: DAIOD_1 (known to fail)
        print("\n--- Testing DAIOD_1 (Newton DA) ---")
        result_newton = DAIOD_1(range_mag, obs_dir, t, 4, pos_obs, mu=3.986e5, tol=1e-3)
        print(f"DAIOD_1 (Newton DA) result: SUCCESS")
        print(f"Final range magnitudes (constant): {result_newton.cons()}")
        
    except Exception as e:
        print(f"DAIOD_1 (Newton DA) failed: {e}")
        print(f"Error details: {str(e)}")
        result_newton = None
    
    # Compare results if both succeeded
    if result_scipy is not None and result_newton is not None:
        difference = np.linalg.norm(result_scipy.cons() - result_newton.cons())
        print(f"\nDifference between methods: {difference:.2e}")
        
        if difference < 1.0:
            print("✅ Both methods converge to same solution")
        else:
            print("❌ Methods converge to different solutions")
    
    elif result_scipy is not None and result_newton is None:
        print(f"\n❌ DAIOD_1Scipy works but DAIOD_1 (Newton DA) fails!")
        print(f"   This confirms the DA Jacobian/higher-order terms issue!")
    
    return result_scipy, result_newton


if __name__ == "__main__":
    print("Testing DA Map accuracy and Newton Jacobian in DAIOD...\n")
    
    # Test 1: Check if DA Jacobian matches finite difference Jacobian
    jacobian_da, jacobian_fd = test_newton_jacobian_accuracy()
    
    # Test 2: Direct comparison of working vs failing methods
    result_scipy, result_newton = test_scipy_vs_newton_comparison()
    
    print(f"\n\n=== FINAL DIAGNOSIS ===")
    print(f"Run this test to determine if the DA higher-order terms in")
    print(f"householder_iter_DA_Map are causing incorrect Jacobians in Newton iteration.")
    print(f"If DA and FD Jacobians differ significantly, that confirms your hypothesis!")
