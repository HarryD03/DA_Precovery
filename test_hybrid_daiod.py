#!/usr/bin/env python3
"""
Test hybrid DAIOD approach:
1. Use scipy.fsolve to find nominal solution (numerically stable)
2. Compute numerical Jacobian at nominal point
3. Use Implicit_solver_DAVec with DAIOD=True to get DA expansion

This should bypass the Lambert solver conditioning issues we identified.
"""

import numpy as np
from daceypy import DA
from daceypy.array import array
import scipy.optimize as opt

# Import our modules
from utils.iod import DAIOD, Implicit_solver_DAVec
from utils.observation import create_los_vectors
from utils.data import load_observation_data
from utils.dynamics import compute_velocity_difference

def test_hybrid_daiod():
    """Test hybrid approach: scipy nominal + DA expansion"""
    
    print("=== Testing Hybrid DAIOD Approach ===")
    print("1. Use scipy for nominal solution")
    print("2. Use DA implicit solver for derivatives")
    print()
    
    # Load the problematic DAIOD data
    obs_data = load_observation_data('apophis_data.csv')
    obs1, obs2, obs3 = obs_data[0], obs_data[1], obs_data[2]
    
    # Create LOS vectors
    rho_hat1, rho_hat2, rho_hat3 = create_los_vectors(obs1, obs2, obs3)
    
    # Define the DAIOD function that we want to solve
    def daiod_function(ranges):
        """DAIOD function: F(rho1, rho2, rho3) = 0"""
        rho1, rho2, rho3 = ranges
        try:
            result = DAIOD(obs1, obs2, obs3, rho1, rho2, rho3)
            return result
        except Exception as e:
            print(f"DAIOD failed for ranges {ranges}: {e}")
            return np.array([1e6, 1e6, 1e6])  # Large residual for failed cases
    
    # Step 1: Use scipy to find nominal solution
    print("Step 1: Finding nominal solution with scipy.fsolve...")
    initial_guess = [20000.0, 20000.0, 20000.0]  # km
    
    # Solve numerically first
    solution = opt.fsolve(daiod_function, initial_guess, xtol=1e-12)
    residual = daiod_function(solution)
    
    print(f"Scipy solution: {solution}")
    print(f"Residual norm: {np.linalg.norm(residual):.2e}")
    
    if np.linalg.norm(residual) > 1e-6:
        print("⚠️  Scipy failed to converge to acceptable tolerance")
        return False
    
    # Step 2: Compute numerical Jacobian at nominal point
    print("\nStep 2: Computing numerical Jacobian...")
    
    def compute_numerical_jacobian(func, x0, h=1e-6):
        """Compute numerical Jacobian using finite differences"""
        n = len(x0)
        f0 = func(x0)
        m = len(f0)
        jac = np.zeros((m, n))
        
        for j in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[j] += h
            x_minus[j] -= h
            
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            
            jac[:, j] = (f_plus - f_minus) / (2 * h)
        
        return jac
    
    jacobian = compute_numerical_jacobian(daiod_function, solution)
    print(f"Jacobian condition number: {np.linalg.cond(jacobian):.2e}")
    print(f"Jacobian determinant: {np.linalg.det(jacobian):.2e}")
    
    # Step 3: Set up DA expansion around nominal solution
    print("\nStep 3: Setting up DA expansion...")
    
    DA.init(3, 3)  # 3rd order, 3 variables
    
    # Create DA variables representing deviations from nominal
    p_da = array([solution[i] + DA(i+1) for i in range(3)])
    
    # Define DAIOD function for DA expansion
    def daiod_function_da(x, p):
        """DAIOD function for DA expansion"""
        return daiod_function(x)
    
    # Step 4: Use Implicit_solver_DAVec with DAIOD=True
    print("\nStep 4: Computing DA expansion...")
    
    try:
        # Use the hybrid approach
        x_da = Implicit_solver_DAVec(
            x0=solution,                    # Nominal solution from scipy
            p=p_da,                        # DA variables (not used in DAIOD mode)
            f=daiod_function_da,           # Function to solve
            NumVariables=3,                # Number of variables
            x0DA=False,                    # Return x0 + M(p) form
            DAIOD=True,                    # Use DAIOD-specific algorithm
            JacDAIOD=jacobian              # Pre-computed numerical Jacobian
        )
        
        print("✅ DA expansion computed successfully!")
        print(f"DA solution type: {type(x_da)}")
        print(f"Solution length: {len(x_da)}")
        
        # Check the nominal values
        nominal_values = [x_da[i].cons() for i in range(3)]
        print(f"\nNominal values from DA: {nominal_values}")
        print(f"Scipy values:           {solution}")
        print(f"Difference:             {np.array(nominal_values) - solution}")
        
        # Check first-order derivatives (sensitivity matrix)
        print("\nFirst-order sensitivities:")
        for i in range(3):
            derivs = [x_da[i].getCoefficient([j+1 if k==j else 0 for k in range(3)]) for j in range(3)]
            print(f"∂rho{i+1}/∂[rho1,rho2,rho3] = {derivs}")
        
        return True
        
    except Exception as e:
        print(f"❌ DA expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_direct_da():
    """Compare hybrid approach with direct DA approach"""
    
    print("\n=== Comparison with Direct DA Approach ===")
    
    # Load data
    obs_data = load_observation_data('apophis_data.csv')
    obs1, obs2, obs3 = obs_data[0], obs_data[1], obs_data[2]
    
    # Try direct DA approach (this should fail due to conditioning)
    print("Attempting direct DA approach...")
    
    DA.init(3, 3)
    initial_guess = [20000.0, 20000.0, 20000.0]
    
    try:
        # Create DA variables
        rho_da = array([initial_guess[i] + DA(i+1) for i in range(3)])
        
        # Try direct DA DAIOD
        result_da = DAIOD(obs1, obs2, obs3, rho_da[0], rho_da[1], rho_da[2])
        
        print("✅ Direct DA approach succeeded (unexpected)")
        return result_da
        
    except Exception as e:
        print(f"❌ Direct DA approach failed as expected: {e}")
        return None

if __name__ == "__main__":
    # Test the hybrid approach
    success = test_hybrid_daiod()
    
    if success:
        print("\n🎉 Hybrid approach successful!")
        print("This demonstrates that using scipy + implicit DA can solve DAIOD")
        
        # Compare with direct approach
        compare_with_direct_da()
        
    else:
        print("\n💥 Hybrid approach failed")
        print("Further investigation needed")
