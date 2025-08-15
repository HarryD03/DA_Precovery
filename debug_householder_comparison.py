#!/usr/bin/env python3
"""
Debug script to compare Float vs DA Householder iterations in Lambert solver.
This will help identify where DA derivatives diverge from correct float derivatives.
"""

import numpy as np
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from tests.test_iod import sample_data_Earth

# ===== COPIED FUNCTIONS FROM lambert_izzo.py =====

def _compute_y(x, ll):
    """Computes y."""
    return np.sqrt(1 - ll**2 * (1 - x**2))

def _tof_equation_p(x, y, T, ll):
    return (3 * T * x - 2 + 2 * ll**3 * x / y) / (1 - x**2)

def _tof_equation_p2(x, y, T, dT, ll):
    return (3 * T + 5 * x * dT + 2 * (1 - ll**2) * ll**3 / y**3) / (1 - x**2)

def _tof_equation_p3(x, y, _, dT, ddT, ll):
    return (7 * x * ddT + 8 * dT - 6 * (1 - ll**2) * ll**5 * x / y**5) / (1 - x**2)

def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computated y."""
    if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4 / 3 * hyp2f1b(S_1)
        T_ = (eta**3 * Q + 4 * ll * eta) * 0.5
    else:
        psi = _compute_psi_float(x, y, ll)
        T_ = np.divide(
            np.divide(psi + M * np.pi, np.sqrt(np.abs(1 - x**2))) - x + ll * y,
            (1 - x**2),
        )
    return T_ - T0

def _compute_psi_float(x, y, ll):
    """Computes psi for float version."""
    if -1 <= x < 1:
        return np.arccos(x * y + ll * (1 - x**2))
    elif x > 1:
        return np.arcsinh((y - x * ll) * np.sqrt(x**2 - 1))
    else:
        return 0.0

def hyp2f1b(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin]."""
    if x >= 1.0:
        return np.inf
    else:
        res = 1.0
        term = 1.0
        ii = 0
        while True:
            term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
            res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1

def householder_float_debug(p0, T0, ll, M, tol=1e-9, maxiter=100):
    """Float Householder with detailed debugging output."""
    print(f"\n=== FLOAT HOUSEHOLDER DEBUG ===")
    print(f"Initial: p0={p0}, T0={T0}, ll={ll}, M={M}")
    
    results = []
    
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder = _tof_equation_p(p0, y, T, ll)
        fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)

        # Householder step (quartic)
        num = fder**2 - fval * fder2 / 2
        denom = fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6
        
        step = fval * (num / denom)
        p = p0 - step

        # Store iteration data
        iteration_data = {
            'iter': ii + 1,
            'x': p0,
            'y': y,
            'F': fval,
            'dF_dx': fder,
            'd2F_dx2': fder2,
            'd3F_dx3': fder3,
            'num': num,
            'denom': denom,
            'step': step,
            'x_new': p
        }
        results.append(iteration_data)
        
        print(f"Float Iter {ii+1:2d}:")
        print(f"  x = {p0:.10f}")
        print(f"  F = {fval:.10e}")
        print(f"  dF/dx = {fder:.10e}")
        print(f"  d²F/dx² = {fder2:.10e}")
        print(f"  d³F/dx³ = {fder3:.10e}")
        print(f"  num = {num:.10e}")
        print(f"  denom = {denom:.10e}")
        print(f"  step = {step:.10e}")
        print(f"  x_new = {p:.10f}")

        if abs(p - p0) < tol:
            print(f"Float converged in {ii+1} iterations")
            return p, results
        p0 = p

    raise RuntimeError("Float Householder failed to converge")

# ===== DA VERSIONS =====

def x2tof_debug(x: Union[DA,float], M: float, L: Union[float, DA]):
    """DA version of x2tof with debugging."""
    K = L**2
    E = x**2 - 1
    y = op.sqrt(1 - L**2 * (1 - x**2))
    
    if isinstance(x, DA):
        x_cons = x.cons()
    else:
        x_cons = x

    # Use the general case (Lancaster formulation)
    psi = compute_psi_debug(x, y, L)
    
    if -1 < x_cons < 1:  # Elliptical case
        sqrt_term = op.sqrt(1 - x**2)
    else:  # Hyperbolic case
        sqrt_term = op.sqrt(x**2 - 1)
        
    T = ((psi + M * np.pi) / sqrt_term - x + L*y) / (1 - x**2)
    return T

def compute_psi_debug(x, y, L):
    """DA version of compute_psi with debugging."""
    if isinstance(x, float):
        x_cons = x
    elif isinstance(x, DA):
        x_cons = x.cons()

    if -1 <= x_cons < 1:
        return op.acos(x*y + L * (1 - x**2))
    elif x_cons > 1:
        return op.asinh((y - x * L) * op.sqrt(x**2 - 1))
    else:
        return 0.0

def householder_DA_debug(x0, p0, tol=1e-12, MaxIter=100):
    """DA Householder with detailed debugging output."""
    print(f"\n=== DA HOUSEHOLDER DEBUG ===")
    print(f"Initial: x0={x0}, p0={p0}")
    
    def f(x, p):
        """Function to obtain f(x) = T(x) - T* for Householder Iteration scheme"""
        T, L, M = p
        return x2tof_debug(x, M, L) - T
    
    MaxVar = DA.getMaxVariables()
    x = x0 + DA(MaxVar)
    flag = True
    iter = 1
    results = []
    
    DA.pushTO(4)
    
    while flag:
        F = f(x, p0)
        dFdx = F.deriv(MaxVar)
        
        if abs(dFdx.cons()) < 1e-15:
            raise ValueError("Derivative is zero - cannot continue")
        
        dFFdxx = dFdx.deriv(MaxVar)
        dFFFdxxx = dFFdxx.deriv(MaxVar)

        # Householder calculation
        num = dFdx.cons()**2 - (F.cons()*dFFdxx.cons()/2)
        denom = (dFdx.cons() * (dFdx.cons()**2 - F.cons()*dFFdxx.cons()) + (dFFFdxxx.cons() * F.cons()**2 / 6))
        
        step = F.cons()*(num/denom)
        x_new = x - step

        # Store iteration data
        iteration_data = {
            'iter': iter,
            'x': x.cons(),
            'F': F.cons(),
            'dF_dx': dFdx.cons(),
            'd2F_dx2': dFFdxx.cons(),
            'd3F_dx3': dFFFdxxx.cons(),
            'num': num,
            'denom': denom,
            'step': step,
            'x_new': x_new.cons()
        }
        results.append(iteration_data)

        print(f"DA Iter {iter:2d}:")
        print(f"  x = {x.cons():.10f}")
        print(f"  F = {F.cons():.10e}")
        print(f"  dF/dx = {dFdx.cons():.10e}")
        print(f"  d²F/dx² = {dFFdxx.cons():.10e}")
        print(f"  d³F/dx³ = {dFFFdxxx.cons():.10e}")
        print(f"  num = {num:.10e}")
        print(f"  denom = {denom:.10e}")
        print(f"  step = {step:.10e}")
        print(f"  x_new = {x_new.cons():.10f}")

        # Convergence check
        if abs(F.cons()) < tol or iter > MaxIter:
            flag = False
            if iter > MaxIter:
                print(f"DA reached max iterations")
            else:
                print(f"DA converged in {iter} iterations")
            break
            
        # Constraint check
        if x_new.cons() >= 0.999999:
            raise ValueError(f"DA: x_new = {x_new.cons()} exceeds 0.999999")
        elif x_new.cons() <= -0.999999:
            raise ValueError(f"DA: x_new = {x_new.cons()} exceeds -0.999999")

        iter += 1
        x = x_new
    
    DA.popTO()
    return x.cons(), results

def compare_derivatives(float_results, da_results):
    """Compare float and DA derivative calculations."""
    print(f"\n=== DERIVATIVE COMPARISON ===")
    print(f"{'Iter':<4} {'Type':<5} {'F':<12} {'dF/dx':<12} {'d²F/dx²':<12} {'d³F/dx³':<12} {'Step':<12}")
    print("-" * 80)
    
    max_iters = min(len(float_results), len(da_results))
    
    for i in range(max_iters):
        f_data = float_results[i]
        d_data = da_results[i]
        
        print(f"{i+1:<4} {'Float':<5} {f_data['F']:<12.5e} {f_data['dF_dx']:<12.5e} {f_data['d2F_dx2']:<12.5e} {f_data['d3F_dx3']:<12.5e} {f_data['step']:<12.5e}")
        print(f"{i+1:<4} {'DA':<5} {d_data['F']:<12.5e} {d_data['dF_dx']:<12.5e} {d_data['d2F_dx2']:<12.5e} {d_data['d3F_dx3']:<12.5e} {d_data['step']:<12.5e}")
        
        # Calculate relative errors
        f_err = abs(f_data['F'] - d_data['F']) / max(abs(f_data['F']), 1e-15)
        df_err = abs(f_data['dF_dx'] - d_data['dF_dx']) / max(abs(f_data['dF_dx']), 1e-15)
        d2f_err = abs(f_data['d2F_dx2'] - d_data['d2F_dx2']) / max(abs(f_data['d2F_dx2']), 1e-15)
        d3f_err = abs(f_data['d3F_dx3'] - d_data['d3F_dx3']) / max(abs(f_data['d3F_dx3']), 1e-15)
        step_err = abs(f_data['step'] - d_data['step']) / max(abs(f_data['step']), 1e-15)
        
        print(f"{'':4} {'Error':<5} {f_err:<12.2e} {df_err:<12.2e} {d2f_err:<12.2e} {d3f_err:<12.2e} {step_err:<12.2e}")
        print()

def run_comparison_test():
    """Run the main comparison test."""
    print("=== HOUSEHOLDER FLOAT vs DA COMPARISON ===")
    
    # Get test data
    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth()
    import pickle
    DA.init(4,3)
    with open('householder_params.pkl', 'rb') as f:
        p = pickle.load(f)
    # Example Lambert problem parameters (you'll need to extract these from actual DAIOD run)
    # These are placeholders - you should capture the actual values during DAIOD execution
    T = p[0] # Example lambda
    L = p[1]  # Example non-dimensional time
    M = p[2]  # Number of revolutions
    x0 = 0.7852417534231431 # Example initial guess
    # Initialize DA system


    
    print(f"Test parameters: L={L}, T={T}, M={M}, x0={x0}")
    
    try:
        # Run float Householder
        print("\n" + "="*50)
        print("RUNNING FLOAT HOUSEHOLDER")
        print("="*50)
        
        float_result, float_results = householder_float_debug(x0, T.cons(), L.cons(), M)
        
        # Run DA Householder  
        print("\n" + "="*50)
        print("RUNNING DA HOUSEHOLDER")
        print("="*50)
        
        p0 = [T, L, M]  # Parameters for DA version
        da_result, da_results = householder_DA_debug(x0, p0)
        
        # Compare results
        print(f"\n=== FINAL RESULTS ===")
        print(f"Float result: {float_result:.10f}")
        print(f"DA result:    {da_result:.10f}")
        print(f"Difference:   {abs(float_result - da_result):.2e}")
        
        # Compare derivatives iteration by iteration
        compare_derivatives(float_results, da_results)
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

def capture_real_lambert_parameters():
    """
    Capture actual Lambert parameters from a DAIOD run.
    You can modify the DAIOD code to call this function and capture real values.
    """
    print("=== CAPTURING REAL LAMBERT PARAMETERS ===")
    
    # This would be called from within the Lambert solver during DAIOD execution
    # to capture the exact L, T, M, x0 values that are causing convergence issues
    pass

if __name__ == "__main__":
    run_comparison_test()
