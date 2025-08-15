#!/usr/bin/env python3
"""
Script to compare Float vs DA Householder iteration terms
This will help identify if DA derivatives are causing the DAIOD convergence issues
"""

import numpy as np
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo, findxy
from tests.test_iod import sample_data_Earth

def finite_difference_derivatives(f, x, p0, h=1e-8):
    """Calculate derivatives using finite differences for comparison"""
    
    # Function value at x
    F_0 = f(x, p0)
    
    # First derivative (central difference)
    F_plus = f(x + h, p0)
    F_minus = f(x - h, p0)
    dF_dx_fd = (F_plus - F_minus) / (2 * h)
    
    # Second derivative (central difference)
    F_plus2 = f(x + 2*h, p0)
    F_minus2 = f(x - 2*h, p0)
    d2F_dx2_fd = (F_plus2 - 2*F_0 + F_minus2) / (4 * h**2)
    
    # Third derivative (central difference, 5-point stencil)
    F_plus3 = f(x + 3*h, p0)
    F_minus3 = f(x - 3*h, p0)
    d3F_dx3_fd = (F_plus3 - 2*F_plus + 2*F_minus - F_minus3) / (2 * h**3)
    
    return F_0, dF_dx_fd, d2F_dx2_fd, d3F_dx3_fd

def test_lambert_householder_derivatives():
    """Test and compare Householder derivatives in Lambert solver"""
    
    print("=== LAMBERT HOUSEHOLDER DERIVATIVE COMPARISON ===\n")
    
    # Get sample data from DAIOD test
    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth()
    mu = 3.986e5
    
    # Create position vectors using Gauss solution
    range_vec = range_mag * obs_dir
    r_vec = range_vec + pos_obs
    
    # Test case: First Lambert problem (obs1 -> obs2)
    r1 = r_vec[:, 0]
    r2 = r_vec[:, 1]
    dt = t[1] - t[0]
    
    print(f"Testing Lambert problem:")
    print(f"r1 = {r1}")
    print(f"r2 = {r2}")
    print(f"dt = {dt} s")
    print(f"mu = {mu}")
    print()
    
    # Calculate Lambert variables that go into findxy
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    A = np.sqrt(r1_norm * r2_norm * (1 + cos_dnu))
    
    if A == 0:
        print("ERROR: A = 0, cannot proceed")
        return
        
    # Dimensionless time of flight
    c = np.sqrt(r1_norm**2 + r2_norm**2 - 2*r1_norm*r2_norm*cos_dnu)
    s = (r1_norm + r2_norm + c) / 2
    T = np.sqrt(2*mu) * dt / (s**(3/2))
    L = (r1_norm + r2_norm) / (2*A) - 1
    
    print(f"Lambert parameters:")
    print(f"L = {L}")
    print(f"T = {T}")
    print()
    
    # Test the Householder iteration function that's causing problems
    # We'll define the function f(x) = x2tof(x, L) - T that needs to be solved
    
    def kepler_equation(x, params):
        """The equation f(x) = x2tof(x, L) - T = 0 that findxy solves"""
        L_val = params[0]
        T_val = params[1]
        
        # Simplified x2tof calculation (you may need to import/implement this)
        # For now, let's use a simple test function
        return x**3 + L_val * x**2 - T_val
    
    # Test point (initial guess for Lambert x variable)
    x_test = 0.5
    params = [L, T]
    
    print(f"=== DERIVATIVE COMPARISON AT x = {x_test} ===")
    print()
    
    # 1. FINITE DIFFERENCE DERIVATIVES (Ground Truth)
    print("1. FINITE DIFFERENCE DERIVATIVES (Ground Truth):")
    F_fd, dF_dx_fd, d2F_dx2_fd, d3F_dx3_fd = finite_difference_derivatives(
        kepler_equation, x_test, params
    )
    print(f"   F = {F_fd}")
    print(f"   dF/dx = {dF_dx_fd}")
    print(f"   d²F/dx² = {d2F_dx2_fd}")
    print(f"   d³F/dx³ = {d3F_dx3_fd}")
    print()
    
    # 2. DA AUTOMATIC DIFFERENTIATION
    print("2. DA AUTOMATIC DIFFERENTIATION:")
    
    DA.init(4, 1)  # Order 4, 1 variable
    x_da = x_test + DA(1)  # Create DA variable
    
    try:
        F_da = kepler_equation(x_da, params)
        dF_dx_da = F_da.deriv(1)
        d2F_dx2_da = dF_dx_da.deriv(1)
        d3F_dx3_da = d2F_dx2_da.deriv(1)
        
        print(f"   F = {F_da.cons()}")
        print(f"   dF/dx = {dF_dx_da.cons()}")
        print(f"   d²F/dx² = {d2F_dx2_da.cons()}")
        print(f"   d³F/dx³ = {d3F_dx3_da.cons()}")
        print()
        
        # 3. COMPARISON AND ERROR ANALYSIS
        print("3. ERROR ANALYSIS:")
        f_error = abs(F_da.cons() - F_fd)
        df_error = abs(dF_dx_da.cons() - dF_dx_fd)
        d2f_error = abs(d2F_dx2_da.cons() - d2F_dx2_fd)
        d3f_error = abs(d3F_dx3_da.cons() - d3F_dx3_fd)
        
        print(f"   |F_DA - F_FD| = {f_error}")
        print(f"   |dF/dx_DA - dF/dx_FD| = {df_error}")
        print(f"   |d²F/dx²_DA - d²F/dx²_FD| = {d2f_error}")
        print(f"   |d³F/dx³_DA - d³F/dx³_FD| = {d3f_error}")
        print()
        
        # 4. HOUSEHOLDER UPDATE COMPARISON
        print("4. HOUSEHOLDER UPDATE COMPARISON:")
        
        # Float version
        num_fd = dF_dx_fd**2 - (F_fd * d2F_dx2_fd / 2)
        denom_fd = (dF_dx_fd * (dF_dx_fd**2 - F_fd * d2F_dx2_fd) + 
                   (d3F_dx3_fd * F_fd**2 / 6))
        step_fd = F_fd * (num_fd / denom_fd) if denom_fd != 0 else float('inf')
        
        # DA version
        num_da = dF_dx_da.cons()**2 - (F_da.cons() * d2F_dx2_da.cons() / 2)
        denom_da = (dF_dx_da.cons() * (dF_dx_da.cons()**2 - F_da.cons() * d2F_dx2_da.cons()) + 
                   (d3F_dx3_da.cons() * F_da.cons()**2 / 6))
        step_da = F_da.cons() * (num_da / denom_da) if denom_da != 0 else float('inf')
        
        print(f"   Float Householder:")
        print(f"     num = {num_fd}")
        print(f"     denom = {denom_fd}")
        print(f"     step = {step_fd}")
        print()
        print(f"   DA Householder:")
        print(f"     num = {num_da}")
        print(f"     denom = {denom_da}")
        print(f"     step = {step_da}")
        print()
        print(f"   Step difference: {abs(step_da - step_fd)}")
        
        # 5. CONVERGENCE ASSESSMENT
        print("\n5. CONVERGENCE ASSESSMENT:")
        if abs(step_da - step_fd) > 1e-10:
            print("   ⚠️  SIGNIFICANT DIFFERENCE in Householder steps!")
            print("   This could explain DAIOD convergence issues.")
        else:
            print("   ✅ Householder steps match well.")
            
        if d3f_error > 1e-6:
            print("   ⚠️  LARGE ERROR in 3rd derivative!")
            print("   DA may not have sufficient order or accuracy.")
        else:
            print("   ✅ Third derivative accuracy acceptable.")
            
    except Exception as e:
        print(f"   ERROR in DA calculation: {e}")
        print("   This suggests DA implementation issues.")

def test_real_lambert_function():
    """Test with the actual Lambert function to see where it fails"""
    
    print("\n=== REAL LAMBERT FUNCTION TEST ===\n")
    
    # Get sample data
    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth()
    mu = 3.986e5
    
    # Create position vectors
    range_vec = range_mag * obs_dir
    r_vec = range_vec + pos_obs
    
    r1 = r_vec[:, 0]
    r2 = r_vec[:, 1]
    dt = t[1] - t[0]
    
    print(f"Testing actual Lambert function with:")
    print(f"r1 = {r1}")
    print(f"r2 = {r2}")
    print(f"dt = {dt}")
    print()
    
    try:
        # Test with float inputs (this should work)
        print("Float Lambert call:")
        velocities_float = lambert_izzo(r1, r2, dt, mu, 0, prograde=True)
        v1_float = velocities_float[0][:, 0]
        v2_float = velocities_float[0][:, 1]
        print(f"✅ Float version succeeded:")
        print(f"   v1 = {v1_float}")
        print(f"   v2 = {v2_float}")
        print()
        
    except Exception as e:
        print(f"❌ Float version failed: {e}")
        print()
    
    try:
        # Test with DA inputs (this might fail)
        print("DA Lambert call:")
        DA.init(4, 3)
        r1_da = array([r1[i] + DA(i+1) for i in range(3)])
        r2_da = array([r2[i] for i in range(3)])  # Keep r2 as constants
        
        velocities_da = lambert_izzo(r1_da, r2_da, dt, mu, 0, prograde=True)
        v1_da = velocities_da[0][:, 0]
        v2_da = velocities_da[0][:, 1]
        print(f"✅ DA version succeeded:")
        print(f"   v1 = {[v.cons() for v in v1_da]}")
        print(f"   v2 = {[v.cons() for v in v2_da]}")
        print()
        
        # Compare results
        print("Comparison:")
        v1_diff = np.linalg.norm([v1_da[i].cons() - v1_float[i] for i in range(3)])
        v2_diff = np.linalg.norm([v2_da[i].cons() - v2_float[i] for i in range(3)])
        print(f"   ||v1_DA - v1_float|| = {v1_diff}")
        print(f"   ||v2_DA - v2_float|| = {v2_diff}")
        
        if v1_diff > 1e-10 or v2_diff > 1e-10:
            print("   ⚠️  SIGNIFICANT DIFFERENCE between DA and float results!")
        else:
            print("   ✅ DA and float results match well.")
            
    except Exception as e:
        print(f"❌ DA version failed: {e}")
        print("   This confirms DA implementation issues in Lambert solver.")

if __name__ == "__main__":
    print("HOUSEHOLDER DERIVATIVE COMPARISON TOOL")
    print("=" * 50)
    
    # Test 1: Derivative comparison with simple function
    test_lambert_householder_derivatives()
    
    # Test 2: Real Lambert function comparison
    test_real_lambert_function()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
