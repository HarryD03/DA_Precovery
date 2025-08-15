#!/usr/bin/env python3
"""
Simplified test to validate the hypothesis that householder_iter_DA_Map
produces incorrect higher-order DA terms, causing Newton iteration divergence in DAIOD.
"""

import numpy as np
import pickle
from daceypy import DA, array
import daceypy.op as op
from scipy.optimize import approx_fprime
from utils.lambert_izzo import householder_iter_DA_nom, householder_iter_DA_Map, x2tof

def householder_iter_DA_Map_debug(x_nom: float, p: list, MaxVar: int, f: callable, tol=1e-12, MaxIter=100):
    """
    Debug version of householder_iter_DA_Map that captures and returns the derivative values
    computed during the iteration for comparison with finite differences.
    """
    iter = 1
    x_Var = MaxVar
    xp = x_nom + DA(x_Var)       # temp variable for Automatic Derivative
    flag = True
    
    # Store derivative values for analysis
    derivatives_history = []
    
    while flag:
        # Collect Derivatives (same as original function)
        F = f(xp, p)
        
        dFdx = F.deriv(x_Var)
        dFFdxx = F.deriv(x_Var).deriv(x_Var)
        dFFFdxxx = F.deriv(x_Var).deriv(x_Var).deriv(x_Var)
        
        # Store the derivative values at current x
        current_x = xp.cons()
        derivatives_info = {
            'iteration': iter,
            'x': current_x,
            'F': F.cons(),
            'dFdx': dFdx.cons(),
            'dFFdxx': dFFdxx.cons(), 
            'dFFFdxxx': dFFFdxxx.cons()
        }
        derivatives_history.append(derivatives_info)
        
        print(f"DA Debug - Iter {iter}: x={current_x:.8f}")
        print(f"  F = {F.cons():.6e}")
        print(f"  dF/dx = {dFdx.cons():.6e}")
        print(f"  d²F/dx² = {dFFdxx.cons():.6e}")
        print(f"  d³F/dx³ = {dFFFdxxx.cons():.6e}")
        
        # Remove DA(x) as an independent DA variable once automatic derivation has taken place
        F = F.plug(x_Var, 0)
        dFdx = dFdx.plug(x_Var, 0)
        dFFdxx = dFFdxx.plug(x_Var, 0)
        dFFFdxxx = dFFFdxxx.plug(x_Var, 0)
        
        # Householder iteration Equation
        num = dFdx**2 - F * dFFdxx/2
        denom = (dFdx * (dFdx**2 - F*dFFdxx)) + (dFFFdxxx* F**2)/6
        
        assert denom != 0, "Cannot divide by 0!"
        x = xp - F*(num/denom)
        
        if iter > MaxIter:
            print("Max iterations reached")
            flag = False
            break
        iter += 1
        xp = x
    
    x = x.plug(x_Var, 0)        # Assert the DA(x) are eliminated in final expansion
    return x, derivatives_history

def test_householder_da_derivatives():
    """
    Test if householder_iter_DA_Map produces correct derivatives by comparing
    DA automatic differentiation with finite difference approximation.
    """
    DA.init(4, 3)
    
    print("=== TESTING HOUSEHOLDER DA MAP DERIVATIVES ===\n")
    
    # Load saved parameters from a DAIOD run
    try:
        with open('householder_params.pkl', 'rb') as P:
            p_da = pickle.load(P)
        T_da, L_da, M = p_da

        T_nom = T_da.cons()
        L_nom = L_da.cons()
        print(f"Loaded parameters: T_nom={T_nom:.6f}, L_nom={L_nom:.6f}, M={M}")
    
    except FileNotFoundError:
        print("No saved parameters found. Using default values.")
        T_nom = 0.5
        L_nom = 0.2
        M = 0
        DA.init(4, 2)
        T_da = T_nom + DA(1)
        L_da = L_nom + DA(2)
        p_da = [T_da, L_da, M]
    
    # Define the function for root finding
    def f(x, p):
        """Lambert time-of-flight equation: f(x) = T(x) - T* = 0"""
        T, L, M = p
        return x2tof(x, M, L) - T
    
    # Get nominal solution
    x0 = 0.7852417534231431  # Initial guess
    p0 = [T_nom, L_nom, M]  # Float parameters
    
    print(f"Computing nominal solution with x0 = {x0}")
    x_nom = householder_iter_DA_nom(x0, p0, f, tol=1e-12, MaxIter=10)
    print(f"Nominal solution: x_nom = {x_nom:.8f}")
    
    # Verify nominal solution is correct
    f_check = f(x_nom, p_da)
    print(f"Function value at x_nom: {f_check.cons()} (should be ~0)")
    
    # Get DA map solution (with higher-order terms) and capture derivatives
    print(f"\nComputing DA map solution with derivative debugging...")
    MaxVar = DA.getMaxVariables()
    x_da_map, derivatives_history = householder_iter_DA_Map_debug(x_nom, p_da, MaxVar, f, tol=1e-12, MaxIter=100)
    
    print(f"DA map constant term: {x_da_map.cons():.8f}")
    print(f"Should match x_nom: {abs(x_da_map.cons() - x_nom) < 1e-10}")
    
    # Extract DA derivatives for the final solution
    dx_dT = x_da_map.deriv(1)  # ∂x/∂T
    dx_dL = x_da_map.deriv(2)  # ∂x/∂L
    
    print(f"\nFinal DA derivatives:")
    print(f"  ∂x/∂T = {dx_dT.cons()}")
    print(f"  ∂x/∂L = {dx_dL.cons()}")
    
    # Now compare the DA derivatives with finite differences at each iteration point
    print(f"\n=== DERIVATIVE VALIDATION FOR EACH ITERATION ===")
    
    for i, deriv_info in enumerate(derivatives_history):
        x_point = deriv_info['x']
        da_f = deriv_info['F']
        da_df_dx = deriv_info['dFdx']
        da_d2f_dx2 = deriv_info['dFFdxx']
        da_d3f_dx3 = deriv_info['dFFFdxxx']
        
        print(f"\n--- Iteration {deriv_info['iteration']}: x = {x_point} ---")
        
        # Compute finite difference derivatives at this x point
        def f_at_x(x_val):
            """Evaluate f at a specific x value"""
            return f(x_val, p0)  # Use float parameters for FD
        
        # First derivative (central difference)
        h1 = 1e-8
        fd_df_dx = (f_at_x(x_point + h1) - f_at_x(x_point - h1)) / (2 * h1)
        
        # Second derivative (central difference)
        h2 = 1e-6
        fd_d2f_dx2 = (f_at_x(x_point + h2) - 2*f_at_x(x_point) + f_at_x(x_point - h2)) / (h2**2)
        
        # Third derivative (forward difference - more stable)
        h3 = 1e-4
        fd_d3f_dx3 = (f_at_x(x_point + 3*h3) - 3*f_at_x(x_point + 2*h3) + 
                      3*f_at_x(x_point + h3) - f_at_x(x_point)) / (h3**3)
        
        # Compare derivatives
        print(f"Function value:")
        print(f"  DA: {da_f}, FD: {f_at_x(x_point)}")
        
        print(f"First derivative (dF/dx):")
        print(f"  DA: {da_df_dx}, FD: {fd_df_dx}")
        error_1st = abs(da_df_dx - fd_df_dx)
        print(f"  Error: {error_1st}")
        
        print(f"Second derivative (d²F/dx²):")
        print(f"  DA: {da_d2f_dx2}, FD: {fd_d2f_dx2}")
        error_2nd = abs(da_d2f_dx2 - fd_d2f_dx2)
        print(f"  Error: {error_2nd}")
        
        print(f"Third derivative (d³F/dx³):")
        print(f"  DA: {da_d3f_dx3}, FD: {fd_d3f_dx3}")
        error_3rd = abs(da_d3f_dx3 - fd_d3f_dx3)
        print(f"  Error: {error_3rd}")
        
        # Check which derivatives are accurate
        tol_1st = 1e-6
        tol_2nd = 1e-4  # Second derivatives are typically less accurate
        tol_3rd = 1e-2  # Third derivatives are even less accurate
        
        print(f"Accuracy check:")
        print(f"  1st derivative OK: {error_1st < tol_1st}")
        print(f"  2nd derivative OK: {error_2nd < tol_2nd}")
        print(f"  3rd derivative OK: {error_3rd < tol_3rd}")
        
        # Store the worst error found
        if i == 0:  # First iteration
            worst_errors = [error_1st, error_2nd, error_3rd]
        else:
            worst_errors[0] = max(worst_errors[0], error_1st)
            worst_errors[1] = max(worst_errors[1], error_2nd)
            worst_errors[2] = max(worst_errors[2], error_3rd)
    
    # Overall assessment based on derivative accuracy
    print(f"\n=== DERIVATIVE ACCURACY ASSESSMENT ===")
    print(f"Worst errors across all iterations:")
    tol_1st = 1e-6
    tol_2nd = 1e-4  # Second derivatives are typically less accurate
    tol_3rd = 1e-2  # Third derivatives are even less accurate
    
    print(f"  1st derivative: {worst_errors[0]:.2e} (tolerance: {tol_1st:.1e})")
    print(f"  2nd derivative: {worst_errors[1]:.2e} (tolerance: {tol_2nd:.1e})")
    print(f"  3rd derivative: {worst_errors[2]:.2e} (tolerance: {tol_3rd:.1e})")
    
    # Determine which derivatives are problematic
    derivatives_accurate = [
        worst_errors[0] < tol_1st,
        worst_errors[1] < tol_2nd,
        worst_errors[2] < tol_3rd
    ]
    
    print(f"\nDerivative accuracy:")
    print(f"  1st derivative accurate: {derivatives_accurate[0]}")
    print(f"  2nd derivative accurate: {derivatives_accurate[1]}")
    print(f"  3rd derivative accurate: {derivatives_accurate[2]}")
    
    if all(derivatives_accurate):
        print("✅ ALL DA derivatives match finite differences")
        print("   householder_iter_DA_Map is computing derivatives correctly")
        print("   The Newton divergence issue must be elsewhere")
        overall_accurate = True
    elif derivatives_accurate[0] and not derivatives_accurate[1]:
        print("❌ 1st derivative OK, but 2nd/3rd derivatives are WRONG")
        print("   This explains Newton divergence - Householder needs accurate 2nd/3rd derivatives!")
        overall_accurate = False
    elif not derivatives_accurate[0]:
        print("❌ Even 1st derivative is WRONG")
        print("   Major issue with DA automatic differentiation in Lambert solver")
        overall_accurate = False
    else:
        print("⚠️  Mixed results - some derivatives accurate, others not")
        overall_accurate = False
    
    return {
        'x_nom': x_nom,
        'da_derivatives': [dx_dT, dx_dL],
        'derivatives_history': derivatives_history,
        'worst_errors': worst_errors,
        'accurate': overall_accurate
    }


def test_daiod_methods_comparison():
    """
    Compare DAIOD_1 (Newton DA) vs DAIOD_1Scipy (scipy.fsolve) on the same problem
    """
    
    print(f"\n\n=== DAIOD METHODS COMPARISON ===\n")
    
    # Import the test data function
    try:
        from tests.test_iod import sample_data_Earth
        from utils.iod import DAIOD_1, DAIOD_1Scipy
        
        # Get test data
        range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth()
        
        print(f"Test data:")
        print(f"  Initial range magnitudes: {range_mag}")
        print(f"  Times: {t}")
        
        # Test parameters
        order = 4
        mu = 3.986e5
        tol = 1e-3
        
        # Test Method 1: scipy.fsolve (should work)
        print(f"\n--- Testing DAIOD_1Scipy (scipy.fsolve) ---")
        try:
            result_scipy = DAIOD_1Scipy(range_mag, obs_dir, t, order, pos_obs, mu=mu, tol=tol)
            print(f"✅ DAIOD_1Scipy succeeded")
            print(f"   Result shape: {result_scipy.shape}")
            print(f"   terms: {result_scipy}")
            scipy_success = True
        except Exception as e:
            print(f"❌ DAIOD_1Scipy failed: {e}")
            result_scipy = None
            scipy_success = False
        
        # Test Method 2: Newton DA (should fail)
        print(f"\n--- Testing DAIOD_1 (Newton DA) ---")
        try:
            result_newton = DAIOD_1(range_mag, obs_dir, t, order, pos_obs, mu=mu, tol=tol)
            print(f"✅ DAIOD_1 (Newton DA) succeeded")
            print(f"   Result shape: {result_newton.shape}")
            print(f"   Constant terms: {result_newton.cons()}")
            newton_success = True
        except Exception as e:
            print(f"❌ DAIOD_1 (Newton DA) failed: {e}")
            print(f"   Error details: {str(e)}")
            result_newton = None
            newton_success = False
        
        # Analysis
        print(f"\n=== COMPARISON ANALYSIS ===")
        
        if scipy_success and not newton_success:
            print("❌ scipy.fsolve works but Newton DA fails")
            print("   This strongly suggests the DA Jacobian is incorrect")
            print("   Root cause: householder_iter_DA_Map produces wrong higher-order terms")
            
        elif scipy_success and newton_success:
            print("✅ Both methods work")
            print("   The issue may have been fixed or is problem-specific")
            
            # Compare results
            diff = np.linalg.norm(result_scipy.cons() - result_newton.cons())
            print(f"   Solution difference: {diff:.2e}")
            
        elif not scipy_success and not newton_success:
            print("❌ Both methods fail")
            print("   The issue may be with the test data or setup")
            
        else:  # newton works but scipy fails (unexpected)
            print("🤔 Newton DA works but scipy fails (unexpected)")
        
        return result_scipy, result_newton
        
    except ImportError as e:
        print(f"Cannot import required functions: {e}")
        print("Make sure the test_iod.py file contains sample_data_Earth function")
        return None, None


if __name__ == "__main__":
    print("Testing DA Map accuracy hypothesis...\n")
    
    # Test 1: Check if DA derivatives are correct
    derivative_results = test_householder_da_derivatives()
    
    # Test 2: Compare DAIOD methods directly
    scipy_result, newton_result = test_daiod_methods_comparison()
    
    print(f"\n\n=== FINAL SUMMARY ===")
    print(f"This test validates the hypothesis that householder_iter_DA_Map")
    print(f"produces incorrect higher-order DA terms, causing Newton divergence.")
    
    if derivative_results['accurate']:
        print(f"🤔 ALL DA derivatives (1st, 2nd, 3rd order) are accurate")
        print(f"   The issue is likely in Newton iteration logic, not DA derivatives")
    else:
        print(f"❌ DA derivatives contain errors - this is the root cause!")
        print(f"   Worst errors: 1st={derivative_results['worst_errors'][0]:.2e}, ")
        print(f"                 2nd={derivative_results['worst_errors'][1]:.2e}, ")
        print(f"                 3rd={derivative_results['worst_errors'][2]:.2e}")
        print(f"   Householder iteration relies on accurate 2nd and 3rd derivatives!")
    
    print(f"\nNext steps:")
    print(f"1. If 2nd/3rd derivatives are wrong: Debug DA automatic differentiation in Lambert solver")
    print(f"2. If all derivatives are right: Check Newton iteration implementation")
    print(f"3. Consider using scipy.fsolve as workaround (it uses robust finite differences)")
    print(f"4. Check if DA truncation order affects derivative accuracy")
