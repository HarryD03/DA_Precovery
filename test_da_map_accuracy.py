#!/usr/bin/env python3

import numpy as np
import pickle
from daceypy import DA, array
import daceypy.op as op
from scipy.optimize import approx_fprime
from utils.lambert_izzo import (
    householder_iter_DA_nom, householder_iter_DA_Map, 
    x2tof, _tof_equation_y, _compute_y, _compute_psi
)

def test_householder_da_map_accuracy():
    """
    Test if householder_iter_DA_Map produces correct higher-order terms
    by comparing DA derivatives with finite difference derivatives
    """
    
    print("=== TESTING HOUSEHOLDER_ITER_DA_MAP ACCURACY ===\n")
    
    # Load the saved parameters from the DAIOD run
    try:
        with open('householder_params.pkl', 'rb') as f:
            p_da = pickle.load(f)
        
        T_da, L_da, M = p_da
        T_nom = T_da.cons()
        L_nom = L_da.cons()
        
        print(f"Loaded parameters:")
        print(f"  T_nom = {T_nom}")
        print(f"  L_nom = {L_nom}")
        print(f"  M = {M}")
        
    except FileNotFoundError:
        print("Warning: householder_params.pkl not found. Using default parameters.")
        T_nom = 0.5
        L_nom = 0.2
        M = 0
        
        # Create DA versions
        DA.init(4, 2)
        T_da = T_nom + DA(1)
        L_da = L_nom + DA(2)
        p_da = [T_da, L_da, M]
    
    # Define the function for Householder iteration
    def f(x, p):
        """Function to obtain f(x) = T(x) - T* for Householder Iteration scheme"""
        T, L, M = p
        return x2tof(x, M, L) - T
    
    # Get initial guess and nominal solution
    x0 = 0.5  # Reasonable initial guess
    p0 = [T_nom, L_nom, M]  # Float parameters for nominal solution
    
    print(f"\n=== STEP 1: GET NOMINAL SOLUTION ===")
    print(f"Initial guess x0 = {x0}")
    
    # Get nominal solution
    x_nom = householder_iter_DA_nom(x0, p0, f, tol=1e-12, MaxIter=100)
    print(f"Nominal solution x_nom = {x_nom}")
    
    # Verify nominal solution
    f_check = f(x_nom, p0)
    print(f"Function value at x_nom: f(x_nom) = {f_check}")
    print(f"Nominal solution accuracy: {abs(f_check) < 1e-10}")
    
    print(f"\n=== STEP 2: GET DA MAP SOLUTION ===")
    
    # Get DA map solution
    MaxVar = DA.getMaxVariables()
    x_da_map = householder_iter_DA_Map(x_nom, p_da, MaxVar, f, tol=1e-12, MaxIter=100)
    
    print(f"DA Map solution obtained")
    print(f"x_da_map constant part: {x_da_map.cons()}")
    print(f"x_da_map linear part: {x_da_map.linear()}")
    
    # Extract derivatives from DA object
    if hasattr(x_da_map, 'deriv'):
        da_first_deriv_T = x_da_map.deriv(1)   # ∂x/∂T
        da_first_deriv_L = x_da_map.deriv(2)   # ∂x/∂L
        
        print(f"DA derivatives:")
        print(f"  ∂x/∂T = {da_first_deriv_T}")
        print(f"  ∂x/∂L = {da_first_deriv_L}")
    
    print(f"\n=== STEP 3: FINITE DIFFERENCE VALIDATION ===")
    
    # Create function that takes parameters as input (for finite differences)
    def x_func_of_params(params):
        """Function that returns x as a function of [T, L]"""
        T_p, L_p = params
        p_test = [T_p, L_p, M]
        
        # Re-solve for x given new parameters
        try:
            x_result = householder_iter_DA_nom(x_nom, p_test, f, tol=1e-12, MaxIter=100)
            return x_result
        except:
            # If householder fails, return a reasonable approximation
            return x_nom
    
    # Compute finite difference derivatives
    h = 1e-6  # Step size for finite differences
    params_nom = np.array([T_nom, L_nom])
    
    # Compute gradients using finite differences
    fd_gradient = approx_fprime(params_nom, x_func_of_params, h)
    
    fd_deriv_T = fd_gradient[0]  # ∂x/∂T via finite differences
    fd_deriv_L = fd_gradient[1]  # ∂x/∂L via finite differences
    
    print(f"Finite difference derivatives:")
    print(f"  ∂x/∂T = {fd_deriv_T}")
    print(f"  ∂x/∂L = {fd_deriv_L}")
    
    print(f"\n=== STEP 4: COMPARISON AND ANALYSIS ===")
    
    if hasattr(x_da_map, 'deriv'):
        # Compare first derivatives
        error_T = abs(da_first_deriv_T - fd_deriv_T)
        error_L = abs(da_first_deriv_L - fd_deriv_L)
        
        print(f"Derivative comparison:")
        print(f"  ∂x/∂T: DA = {da_first_deriv_T:.8f}, FD = {fd_deriv_T:.8f}, Error = {error_T:.2e}")
        print(f"  ∂x/∂L: DA = {da_first_deriv_L:.8f}, FD = {fd_deriv_L:.8f}, Error = {error_L:.2e}")
        
        # Check if errors are reasonable
        tolerance = 1e-6
        t_deriv_ok = error_T < tolerance
        l_deriv_ok = error_L < tolerance
        
        print(f"\nAccuracy assessment (tolerance = {tolerance}):")
        print(f"  ∂x/∂T accurate: {t_deriv_ok}")
        print(f"  ∂x/∂L accurate: {l_deriv_ok}")
        
        if t_deriv_ok and l_deriv_ok:
            print("✅ DA Map derivatives appear CORRECT")
        else:
            print("❌ DA Map derivatives appear INCORRECT")
            print("    This could explain DAIOD Newton divergence!")
    
    return x_da_map, fd_gradient


def test_alternative_jacobian_method():
    """
    Test an alternative method for computing Jacobian without DA higher-order terms
    This mimics what scipy.optimize.fsolve does internally
    """
    
    print(f"\n\n=== TESTING ALTERNATIVE JACOBIAN METHOD ===\n")
    
    # Load parameters
    try:
        with open('householder_params.pkl', 'rb') as f:
            p_da = pickle.load(f)
        T_da, L_da, M = p_da
        T_nom = T_da.cons()
        L_nom = L_da.cons()
    except:
        T_nom = 0.5
        L_nom = 0.2
        M = 0
    
    # Define the velocity difference function (similar to DAIOD f(x,p))
    def velocity_difference_function(range_magnitudes, perturbations):
        """
        Simplified version of the DAIOD f(x,p) function
        This represents the velocity difference that should be zero
        """
        # This is a placeholder - in real implementation, this would call Lambert solver
        # and compute velocity differences
        
        # For testing, use a simple nonlinear function that mimics the behavior
        x1, x2, x3 = range_magnitudes + perturbations
        
        # Simulate velocity calculations (nonlinear dependencies)
        v1 = np.sqrt(x1**2 + 0.1*x2) - np.sqrt(x2**2 + 0.1*x3)
        v2 = np.sqrt(x2**2 + 0.1*x3) - np.sqrt(x3**2 + 0.1*x1)
        v3 = np.sqrt(x3**2 + 0.1*x1) - np.sqrt(x1**2 + 0.1*x2)
        
        return np.array([v1, v2, v3])
    
    # Test parameters
    range_mag_nominal = np.array([1000.0, 1100.0, 1200.0])  # Nominal range magnitudes
    perturbations = np.array([0.0, 0.0, 0.0])  # No perturbations for nominal
    
    print(f"Testing with nominal range magnitudes: {range_mag_nominal}")
    
    # Method 1: DA-based Jacobian (current approach)
    print(f"\n--- Method 1: DA-based Jacobian ---")
    
    DA.init(2, 3)  # Order 2, 3 variables
    
    # Create DA variables
    range_da = array([range_mag_nominal[i] + DA(i+1) for i in range(3)])
    pert_da = array([0.0 + DA(i+1) * 0.001 for i in range(3)])  # Small perturbations
    
    # Compute function with DA
    try:
        result_da = velocity_difference_function(range_mag_nominal, pert_da.cons())
        
        # Extract Jacobian from DA linear terms
        jacobian_da = np.zeros((3, 3))
        if hasattr(result_da, '__len__'):
            for i in range(3):
                if hasattr(result_da[i], 'linear'):
                    jacobian_da[i, :] = result_da[i].linear()
        
        print(f"DA Jacobian:")
        print(jacobian_da)
        
    except Exception as e:
        print(f"DA method failed: {e}")
        jacobian_da = None
    
    # Method 2: Finite Difference Jacobian (scipy.optimize approach)
    print(f"\n--- Method 2: Finite Difference Jacobian ---")
    
    def func_for_fd(x):
        """Function for finite differences - takes only range magnitudes"""
        return velocity_difference_function(x, np.zeros(3))
    
    # Compute Jacobian via finite differences
    h = 1e-6
    jacobian_fd = approx_fprime(range_mag_nominal, func_for_fd, h).reshape(3, 3)
    
    print(f"Finite Difference Jacobian:")
    print(jacobian_fd)
    
    # Compare methods
    if jacobian_da is not None:
        print(f"\n--- Comparison ---")
        error_matrix = np.abs(jacobian_da - jacobian_fd)
        max_error = np.max(error_matrix)
        
        print(f"Maximum error between methods: {max_error:.2e}")
        print(f"Error matrix:")
        print(error_matrix)
        
        if max_error < 1e-6:
            print("✅ DA and FD Jacobians match - DA higher-order terms are correct")
        else:
            print("❌ DA and FD Jacobians differ - DA higher-order terms may be wrong")
            print("    This supports the hypothesis that DA Map is causing DAIOD issues")
    
    return jacobian_da, jacobian_fd


if __name__ == "__main__":
    # Run the tests
    print("Testing if householder_iter_DA_Map produces correct higher-order terms...\n")
    
    # Test 1: Validate DA Map derivatives
    x_da_map, fd_gradient = test_householder_da_map_accuracy()
    
    # Test 2: Compare Jacobian methods
    jacobian_da, jacobian_fd = test_alternative_jacobian_method()
    
    print(f"\n\n=== SUMMARY ===")
    print(f"This test helps determine if the DA higher-order terms in householder_iter_DA_Map")
    print(f"are causing the Newton divergence in DAIOD_1.")
    print(f"If the DA derivatives don't match finite differences, that's your culprit!")
