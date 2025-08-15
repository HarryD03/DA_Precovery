#!/usr/bin/env python3
"""
Compare Newton iteration vs Householder iteration on arbitrary functions
to isolate whether the issue is with DA 2nd/3rd derivatives or the algorithm itself.
"""

import numpy as np
import matplotlib.pyplot as plt
from daceypy import DA, array
import daceypy.op as op
from scipy.optimize import fsolve
import sympy as sp

def newton_iteration(f, df, x0, tol=1e-12, max_iter=100):
    """
    Standard Newton iteration: x_{n+1} = x_n - f(x_n)/f'(x_n)
    Returns the iteration history for analysis
    """
    x = x0
    history = [x0]
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivative too small at iteration {i}")
        
        x_new = x - fx / dfx
        history.append(x_new)
        
        print(f"Newton iter {i+1}: x = {x_new:.10f}, f(x) = {f(x_new):.2e}")
        
        if abs(fx) < tol:
            return x_new, history
        
        x = x_new
    
    raise ValueError("Newton iteration failed to converge")

def householder_iteration(f, df, d2f, d3f, x0, tol=1e-12, max_iter=100):
    """
    Householder iteration (3rd order): Uses f, f', f'', f'''
    x_{n+1} = x_n - f(x_n) * (f'(x_n)^2 - f(x_n)*f''(x_n)/2) / 
                     (f'(x_n) * (f'(x_n)^2 - f(x_n)*f''(x_n)) + f'''(x_n)*f(x_n)^2/6)
    """
    x = x0
    history = [x0]
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        d3fx = d3f(x)
        
        print(f"Householder iter {i+1}: x = {x:.10f}")
        print(f"  f(x) = {fx:.6e}")
        print(f"  f'(x) = {dfx:.6e}")
        print(f"  f''(x) = {d2fx:.6e}")
        print(f"  f'''(x) = {d3fx:.6e}")
        
        # Householder iteration formula
        num = dfx**2 - fx * d2fx / 2
        denom = dfx * (dfx**2 - fx * d2fx) + d3fx * fx**2 / 6
        
        if abs(denom) < 1e-15:
            raise ValueError(f"Denominator too small at iteration {i}")
        
        x_new = x - fx * (num / denom)
        history.append(x_new)
        
        print(f"  -> x_new = {x_new:.10f}, f(x_new) = {f(x_new):.2e}")
        
        if abs(fx) < tol:
            return x_new, history
        
        x = x_new
    
    raise ValueError("Householder iteration failed to converge")

def householder_iteration_da(f_da, x0, p_da, tol=1e-12, max_iter=100):
    """
    Householder iteration using DA automatic differentiation
    This mimics what happens in householder_iter_DA_Map
    """
    DA.init(4, len(p_da))  # 4th order, number of parameters
    
    x = x0 + DA(len(p_da) + 1)  # Last variable for x differentiation
    history = [x0]
    
    for i in range(max_iter):
        # Evaluate function with DA
        fx_da = f_da(x, p_da)
        
        # Get derivatives using DA
        dfx = fx_da.deriv(len(p_da) + 1).cons()
        d2fx = fx_da.deriv(len(p_da) + 1).deriv(len(p_da) + 1).cons()
        d3fx = fx_da.deriv(len(p_da) + 1).deriv(len(p_da) + 1).deriv(len(p_da) + 1).cons()
        fx = fx_da.cons()
        
        print(f"DA Householder iter {i+1}: x = {x.cons():.10f}")
        print(f"  f(x) = {fx:.6e}")
        print(f"  f'(x) = {dfx:.6e}")
        print(f"  f''(x) = {d2fx:.6e}")
        print(f"  f'''(x) = {d3fx:.6e}")
        
        # Householder iteration formula
        num = dfx**2 - fx * d2fx / 2
        denom = dfx * (dfx**2 - fx * d2fx) + d3fx * fx**2 / 6
        
        if abs(denom) < 1e-15:
            raise ValueError(f"Denominator too small at iteration {i}")
        
        step = fx * (num / denom)
        x_new_val = x.cons() - step
        history.append(x_new_val)
        
        # Update x for next iteration
        x = x_new_val + DA(len(p_da) + 1)
        
        print(f"  -> x_new = {x_new_val:.10f}, f(x_new) = {f_da(x_new_val, [p.cons() for p in p_da]):.2e}")
        
        if abs(fx) < tol:
            return x_new_val, history
    
    raise ValueError("DA Householder iteration failed to converge")

def create_test_functions():
    """
    Create a set of test functions with known analytical derivatives
    """
    functions = {}
    
    # Test 1: Simple polynomial f(x) = x^3 - 2x - 5, root ≈ 2.094551
    def f1(x):
        return x**3 - 2*x - 5
    
    def df1(x):
        return 3*x**2 - 2
    
    def d2f1(x):
        return 6*x
    
    def d3f1(x):
        return 6
    
    def f1_da(x, p):
        return x**3 - 2*x - 5
    
    functions['polynomial'] = {
        'f': f1, 'df': df1, 'd2f': d2f1, 'd3f': d3f1, 'f_da': f1_da,
        'x0': 2.0, 'root_exact': 2.094551481542327,
        'description': 'x³ - 2x - 5 = 0'
    }
    
    # Test 2: Exponential function f(x) = e^x - 3x, root ≈ 1.512
    def f2(x):
        return np.exp(x) - 3*x
    
    def df2(x):
        return np.exp(x) - 3
    
    def d2f2(x):
        return np.exp(x)
    
    def d3f2(x):
        return np.exp(x)
    
    def f2_da(x, p):
        return op.exp(x) - 3*x
    
    functions['exponential'] = {
        'f': f2, 'df': df2, 'd2f': d2f2, 'd3f': d3f2, 'f_da': f2_da,
        'x0': 1.0, 'root_exact': 1.512134551657842,
        'description': 'eˣ - 3x = 0'
    }
    
    # Test 3: Trigonometric function f(x) = sin(x) - x/2, root ≈ 1.896
    def f3(x):
        return np.sin(x) - x/2
    
    def df3(x):
        return np.cos(x) - 0.5
    
    def d2f3(x):
        return -np.sin(x)
    
    def d3f3(x):
        return -np.cos(x)
    
    def f3_da(x, p):
        return op.sin(x) - x/2
    
    functions['trigonometric'] = {
        'f': f3, 'df': df3, 'd2f': d2f3, 'd3f': d3f3, 'f_da': f3_da,
        'x0': 2.0, 'root_exact': 1.8955494267033981,
        'description': 'sin(x) - x/2 = 0'
    }
    
    # Test 4: Function similar to Lambert TOF equation structure
    def f4(x):
        return x - 0.5 * np.log(1 + x**2) - 1.0
    
    def df4(x):
        return 1 - x / (1 + x**2)
    
    def d2f4(x):
        return (x**2 - 1) / (1 + x**2)**2
    
    def d3f4(x):
        return 2*x*(x**2 - 3) / (1 + x**2)**3
    
    def f4_da(x, p):
        return x - 0.5 * op.log(1 + x**2) - 1.0
    
    functions['lambert_like'] = {
        'f': f4, 'df': df4, 'd2f': d2f4, 'd3f': d3f4, 'f_da': f4_da,
        'x0': 1.5, 'root_exact': None,  # Will compute numerically
        'description': 'x - 0.5*ln(1+x²) - 1 = 0'
    }
    
    return functions

def test_derivative_accuracy(func_data):
    """
    Test if DA automatic differentiation gives the same derivatives as analytical ones
    """
    print(f"\n=== TESTING DERIVATIVE ACCURACY ===")
    
    f = func_data['f']
    df = func_data['df']
    d2f = func_data['d2f']
    d3f = func_data['d3f']
    f_da = func_data['f_da']
    x0 = func_data['x0']
    
    # Test derivatives at the initial point
    x_test = x0
    
    # Analytical derivatives
    f_val = f(x_test)
    df_val = df(x_test)
    d2f_val = d2f(x_test)
    d3f_val = d3f(x_test)
    
    print(f"At x = {x_test}:")
    print(f"Analytical derivatives:")
    print(f"  f(x) = {f_val:.8e}")
    print(f"  f'(x) = {df_val:.8e}")
    print(f"  f''(x) = {d2f_val:.8e}")
    print(f"  f'''(x) = {d3f_val:.8e}")
    
    # DA derivatives
    DA.init(4, 1)
    x_da = x_test + DA(1)
    p_da = [0.0]  # Dummy parameter
    
    f_da_result = f_da(x_da, p_da)
    da_f = f_da_result.cons()
    da_df = f_da_result.deriv(1).cons()
    da_d2f = f_da_result.deriv(1).deriv(1).cons()
    da_d3f = f_da_result.deriv(1).deriv(1).deriv(1).cons()
    
    print(f"DA automatic differentiation:")
    print(f"  f(x) = {da_f:.8e}")
    print(f"  f'(x) = {da_df:.8e}")
    print(f"  f''(x) = {da_d2f:.8e}")
    print(f"  f'''(x) = {da_d3f:.8e}")
    
    # Compare errors
    errors = [
        abs(f_val - da_f),
        abs(df_val - da_df),
        abs(d2f_val - da_d2f),
        abs(d3f_val - da_d3f)
    ]
    
    print(f"Errors (analytical - DA):")
    print(f"  f: {errors[0]:.2e}")
    print(f"  f': {errors[1]:.2e}")
    print(f"  f'': {errors[2]:.2e}")
    print(f"  f''': {errors[3]:.2e}")
    
    # Check accuracy
    tolerances = [1e-12, 1e-10, 1e-8, 1e-6]
    accurate = [errors[i] < tolerances[i] for i in range(4)]
    
    print(f"Accuracy check:")
    for i, name in enumerate(['f', "f'", "f''", "f'''"]):
        print(f"  {name}: {'✅' if accurate[i] else '❌'} (error < {tolerances[i]:.0e})")
    
    return all(accurate)

def compare_convergence_rates(func_data):
    """
    Compare convergence rates between Newton and Householder methods
    """
    print(f"\n=== COMPARING CONVERGENCE RATES ===")
    print(f"Function: {func_data['description']}")
    
    x0 = func_data['x0']
    
    try:
        # Newton iteration
        print(f"\n--- Newton Iteration ---")
        x_newton, history_newton = newton_iteration(
            func_data['f'], func_data['df'], x0, max_iter=20
        )
        
    except Exception as e:
        print(f"Newton failed: {e}")
        x_newton, history_newton = None, []
    
    try:
        # Householder iteration (analytical derivatives)
        print(f"\n--- Householder Iteration (Analytical) ---")
        x_householder, history_householder = householder_iteration(
            func_data['f'], func_data['df'], func_data['d2f'], func_data['d3f'], 
            x0, max_iter=20
        )
        
    except Exception as e:
        print(f"Householder (analytical) failed: {e}")
        x_householder, history_householder = None, []
    
    try:
        # Householder iteration (DA derivatives)
        print(f"\n--- Householder Iteration (DA) ---")
        p_da = [0.0]  # Dummy parameter
        x_householder_da, history_householder_da = householder_iteration_da(
            func_data['f_da'], x0, p_da, max_iter=20
        )
        
    except Exception as e:
        print(f"Householder (DA) failed: {e}")
        x_householder_da, history_householder_da = None, []
    
    # Compare results
    print(f"\n=== CONVERGENCE COMPARISON ===")
    
    if x_newton is not None:
        print(f"Newton converged to: {x_newton:.12f} in {len(history_newton)-1} iterations")
    
    if x_householder is not None:
        print(f"Householder (analytical) converged to: {x_householder:.12f} in {len(history_householder)-1} iterations")
    
    if x_householder_da is not None:
        print(f"Householder (DA) converged to: {x_householder_da:.12f} in {len(history_householder_da)-1} iterations")
    
    # Check if they converge to the same root
    if all(x is not None for x in [x_newton, x_householder, x_householder_da]):
        diff_analytical = abs(x_newton - x_householder)
        diff_da = abs(x_newton - x_householder_da)
        diff_householder = abs(x_householder - x_householder_da)
        
        print(f"\nSolution differences:")
        print(f"  Newton vs Householder (analytical): {diff_analytical:.2e}")
        print(f"  Newton vs Householder (DA): {diff_da:.2e}")
        print(f"  Householder analytical vs DA: {diff_householder:.2e}")
        
        if diff_householder > 1e-10:
            print(f"❌ Householder methods give different results!")
            print(f"   This indicates DA derivatives are wrong")
        else:
            print(f"✅ All methods converge to same solution")
    
    return {
        'newton': (x_newton, history_newton),
        'householder_analytical': (x_householder, history_householder),
        'householder_da': (x_householder_da, history_householder_da)
    }

def main():
    """
    Main test function - runs all comparisons
    """
    print("=== NEWTON vs HOUSEHOLDER ITERATION COMPARISON ===")
    print("Testing to isolate if DA 2nd/3rd derivatives cause convergence issues\n")
    
    # Create test functions
    functions = create_test_functions()
    
    # Test each function
    for name, func_data in functions.items():
        print(f"\n{'='*60}")
        print(f"TESTING FUNCTION: {name.upper()}")
        print(f"Description: {func_data['description']}")
        print(f"Initial guess: x0 = {func_data['x0']}")
        print(f"{'='*60}")
        
        # Test 1: Check if DA derivatives are accurate
        derivatives_accurate = test_derivative_accuracy(func_data)
        
        # Test 2: Compare convergence
        results = compare_convergence_rates(func_data)
        
        # Analysis
        if not derivatives_accurate:
            print(f"\n❌ DA DERIVATIVES ARE INACCURATE for {name}")
            print(f"   This explains why Householder DA fails!")
        else:
            print(f"\n✅ DA derivatives are accurate for {name}")
        
        print(f"\n" + "-"*60)
    
    print(f"\n\n=== FINAL CONCLUSION ===")
    print(f"If DA derivatives are inaccurate for any function, especially")
    print(f"the 2nd and 3rd derivatives, this confirms your hypothesis that")
    print(f"the DA automatic differentiation is causing the Newton divergence")
    print(f"in the DAIOD algorithm.")

if __name__ == "__main__":
    main()
