import numpy as np
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def simple_lambert_derivative_test():
    """
    Simple test to check if Lambert solver derivatives are correct
    """
    
    # Test data
    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          
        [3430.2, 3460.1, 3490.1],          
        [4078.5, 4078.5, 4078.5]           
    ])

    obs_dir = np.array([            
        [0.71643, 0.56897, 0.41841],            
        [0.68074, 0.79531, 0.87007],            
        [-0.15270,-0.20917,-0.26059]            
    ])

    t = np.array([0.0, 118.10, 237.58])
    mu = 3.986e5
    
    from utils.iod import Guass_8th_seed
    _, _, range_mag_guass, _ = Guass_8th_seed(pos_obs, obs_dir, t, mu=mu)
    
    print("=== SIMPLE LAMBERT DERIVATIVE TEST ===")
    
    # Fixed positions for Lambert arc
    r1 = (range_mag_guass[1] * obs_dir[:,1]) + pos_obs[:,1]  
    r2_base = (range_mag_guass[2] * obs_dir[:,2]) + pos_obs[:,2]
    dt = t[2] - t[1]
    
    print(f"Testing Lambert arc with:")
    print(f"r1 = {r1}")
    print(f"r2_base = {r2_base}")
    print(f"dt = {dt}")
    
    # Test: vary r2[2] (z-component) and see effect on v1[2] (z-component)
    def lambert_v1z_from_r2z_perturbation(delta_r2z):
        """Return z-component of v1 when r2[2] is perturbed by delta_r2z"""
        r2_perturbed = r2_base.copy()
        r2_perturbed[2] += delta_r2z
        
        velocities = lambert_izzo(r1, r2_perturbed, dt, mu, 0, prograde=True)
        v1 = velocities[0][:,0]
        return v1[2].cons() if hasattr(v1[2], 'cons') else v1[2]
    
    # Finite difference derivative
    fd_deriv = approx_fprime([0.0], lambert_v1z_from_r2z_perturbation, 1e-6)[0]
    
    # DA derivative 
    DA.init(4, 1)
    delta_r2z_da = DA(1)  # Small perturbation as DA variable
    
    r2_da = array([r2_base[0], r2_base[1], r2_base[2] + delta_r2z_da])
    
    velocities_da = lambert_izzo(array(r1), r2_da, dt, mu, 0, prograde=True)
    v1_da = velocities_da[0][:,0]
    
    da_deriv = v1_da[2].deriv(1).cons()
    
    print(f"\nResults:")
    print(f"FD ∂v1z/∂r2z: {fd_deriv}")
    print(f"DA ∂v1z/∂r2z: {da_deriv}")
    
    error = abs(da_deriv - fd_deriv)
    rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
    
    print(f"Absolute error: {error}")
    print(f"Relative error: {rel_error:.6f}")
    print(f"Relative error %: {rel_error*100:.3f}%")
    
    if rel_error < 0.01:  # 1% tolerance
        print("✅ Lambert derivatives are accurate")
        return True
    else:
        print("❌ Lambert derivatives have significant errors!")
        print("This confirms the issue is in Lambert solver DA implementation,")
        print("not in the Householder iteration.")
        return False

def test_da_basic_operations():
    """Test if basic DA operations work correctly"""
    print("\n=== BASIC DA OPERATIONS TEST ===")
    
    DA.init(4, 2)
    x = 3.0 + DA(1)
    y = 2.0 + DA(2)
    
    # Test basic operations
    z1 = x + y      # Should have derivatives [1, 1]
    z2 = x * y      # Should have derivatives [y.cons(), x.cons()] = [2, 3]
    z3 = x**2       # Should have derivative [2*x.cons()] = [6]
    
    print("Testing basic DA operations:")
    print(f"z1 = x + y: value={z1.cons()}, ∂/∂x={z1.deriv(1).cons()}, ∂/∂y={z1.deriv(2).cons()}")
    print(f"Expected: value=5, ∂/∂x=1, ∂/∂y=1")
    
    print(f"z2 = x * y: value={z2.cons()}, ∂/∂x={z2.deriv(1).cons()}, ∂/∂y={z2.deriv(2).cons()}")
    print(f"Expected: value=6, ∂/∂x=2, ∂/∂y=3")
    
    print(f"z3 = x^2: value={z3.cons()}, ∂/∂x={z3.deriv(1).cons()}")
    print(f"Expected: value=9, ∂/∂x=6")
    
    # Check accuracy
    errors = [
        abs(z1.cons() - 5.0),
        abs(z1.deriv(1).cons() - 1.0), 
        abs(z1.deriv(2).cons() - 1.0),
        abs(z2.cons() - 6.0),
        abs(z2.deriv(1).cons() - 2.0),
        abs(z2.deriv(2).cons() - 3.0),
        abs(z3.cons() - 9.0),
        abs(z3.deriv(1).cons() - 6.0)
    ]
    
    max_error = max(errors)
    if max_error < 1e-12:
        print("✅ Basic DA operations are correct")
        return True
    else:
        print(f"❌ Basic DA operations have errors (max: {max_error})")
        return False

if __name__ == "__main__":
    print("Testing basic DA operations first...")
    da_basic_ok = test_da_basic_operations()
    
    print("\nTesting Lambert derivatives...")
    lambert_ok = simple_lambert_derivative_test()
    
    print(f"\n=== CONCLUSION ===")
    if da_basic_ok and not lambert_ok:
        print("🎯 CONFIRMED: Issue is in Lambert solver DA derivatives")
        print("Basic DA works fine, but Lambert solver has derivative errors.")
        print("Since you confirmed Householder iteration is correct,")
        print("the issue must be in other parts of Lambert solver:")
        print("- Vector norm computations")
        print("- Trigonometric functions") 
        print("- Function composition in Lambert algorithm")
    elif not da_basic_ok:
        print("🎯 Issue is in fundamental DA operations")
    else:
        print("🎯 Both basic DA and Lambert derivatives work - unexpected!")
