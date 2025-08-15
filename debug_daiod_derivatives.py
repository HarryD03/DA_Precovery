import numpy as np
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def debug_daiod_derivative_chain():
    """
    Debug the derivative chain in DAIOD to identify where DA derivatives fail
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
    
    print("=== DEBUGGING DAIOD DERIVATIVE CHAIN ===")
    print(f"Initial range_mag: {range_mag_guass}")
    
    # Test 1: Individual Lambert arc derivatives
    def test_single_lambert_arc():
        print("\n1. Testing Individual Lambert Arc Derivatives")
        
        # Pick first arc: obs1 -> obs2
        r1 = range_mag_guass[0] * obs_dir[:,0] + pos_obs[:,0]
        r2 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
        dt = t[1] - t[0]
        
        print(f"Arc 1: r1={r1}, r2={r2}, dt={dt}")
        
        # Test derivative of v2 w.r.t. range_mag[1] (affects r2)
        def v2_from_range1(range_vals):
            r2_test = range_vals[1] * obs_dir[:,1] + pos_obs[:,1]
            velocities = lambert_izzo(r1, r2_test, dt, mu, 0, prograde=True)
            return velocities[0][:,1].cons() if hasattr(velocities[0][:,1], 'cons') else velocities[0][:,1]
        
        # Finite difference
        v2_fd_jac = approx_fprime(range_mag_guass[:2], 
                                  lambda x: v2_from_range1(x)[2], 1e-6)  # z-component
        
        # DA version
        DA.init(4, 2)
        range_da = array([range_mag_guass[i] + DA(i+1) for i in range(2)])
        r2_da = range_da[1] * obs_dir[2,1] + pos_obs[2,1]  # z-component of r2
        
        # This is where we might see issues with the Lambert DA derivatives
        print(f"FD ∂v2z/∂[range0,range1]: {v2_fd_jac}")
        
    # Test 2: Composition of derivatives  
    def test_velocity_difference_derivatives():
        print("\n2. Testing Velocity Difference Derivatives")
        
        # Focus on the third range variable where we see large errors
        def dv_component_z(range_mag):
            """Return z-component of velocity difference"""
            i_rho = obs_dir
            
            range_vec = np.zeros_like(i_rho)
            for i in range(len(range_mag)):
                range_vec[:,i] = range_mag[i] * i_rho[:,i]

            r_vec = np.zeros_like(range_vec)
            for i in range(len(range_vec[0])):                  
                r_vec[:,i] = range_vec[:,i] + pos_obs[:,i]

            # Arc 1: obs1 -> obs2
            velocities_12 = lambert_izzo(r_vec[:,0], r_vec[:,1], t[1] - t[0], mu, 0, prograde=True)
            v2_from_arc1 = velocities_12[0][:,1]
            
            # Arc 2: obs2 -> obs3  
            velocities_23 = lambert_izzo(r_vec[:,1], r_vec[:,2], t[2] - t[1], mu, 0, prograde=True)
            v2_from_arc2 = velocities_23[0][:,0]
            
            # Velocity difference at obs2
            dv = v2_from_arc2 - v2_from_arc1
            return dv[2].cons() if hasattr(dv[2], 'cons') else dv[2]  # z-component
        
        # Finite difference derivative w.r.t. range3
        def dv_z_single_var(range3_val):
            range_test = range_mag_guass.copy()
            range_test[2] = range3_val
            return dv_component_z(range_test)
        
        fd_deriv = approx_fprime([range_mag_guass[2]], dv_z_single_var, 1e-6)[0]
        
        # DA derivative
        DA.init(4, 1)
        range3_da = range_mag_guass[2] + DA(1)
        
        # Create DA-compatible version of dv_component_z
        def dv_component_z_da(range3_da_val):
            i_rho = obs_dir
            range_mag = [range_mag_guass[0], range_mag_guass[1], range3_da_val]
            
            range_vec = array(np.zeros_like(i_rho))
            for i in range(len(range_mag)):
                for j in range(3):
                    range_vec[j,i] = range_mag[i] * i_rho[j,i]

            r_vec = array(np.zeros_like(range_vec.cons()))
            for i in range(3):                  
                for j in range(3):
                    r_vec[j,i] = range_vec[j,i] + pos_obs[j,i]

            # Arc 1: obs1 -> obs2
            velocities_12 = lambert_izzo(r_vec[:,0], r_vec[:,1], t[1] - t[0], mu, 0, prograde=True)
            v2_from_arc1 = velocities_12[0][:,1]
            
            # Arc 2: obs2 -> obs3  
            velocities_23 = lambert_izzo(r_vec[:,1], r_vec[:,2], t[2] - t[1], mu, 0, prograde=True)
            v2_from_arc2 = velocities_23[0][:,0]
            
            # Velocity difference at obs2
            dv = v2_from_arc2 - v2_from_arc1
            return dv[2]  # z-component
        
        # This will exercise the full derivative chain
        dv_z_da = dv_component_z_da(range3_da)
        da_deriv = dv_z_da.deriv(1).cons() if hasattr(dv_z_da, 'deriv') else 0
        
        print(f"FD ∂(dvz)/∂range3: {fd_deriv}")
        print(f"DA ∂(dvz)/∂range3: {da_deriv}")
        print(f"Error: {abs(da_deriv - fd_deriv)}")
        print(f"Relative error: {abs(da_deriv - fd_deriv) / abs(fd_deriv) if fd_deriv != 0 else float('inf')}")
        
        if abs(da_deriv - fd_deriv) > 1e-6:
            print("❌ Large error in individual derivative component!")
            return False
        else:
            print("✅ Individual derivative component matches")
            return True
    
    # Test 3: Check if it's a higher-order derivative issue
    def test_higher_order_terms():
        print("\n3. Testing Higher-Order Derivative Terms")
        
        # Use smaller perturbations to see if it's a higher-order truncation issue
        step_sizes = [1e-4, 1e-6, 1e-8, 1e-10]
        
        def dv_z_for_step(range_mag):
            return dv_component_z(range_mag)
        
        for h in step_sizes:
            fd_deriv_h = approx_fprime(range_mag_guass, 
                                       lambda x: dv_z_for_step(x), h)[2]  # ∂/∂range3
            print(f"FD derivative (h={h:1.0e}): {fd_deriv_h}")
    
    # Run all tests
    test_single_lambert_arc()
    matches = test_velocity_difference_derivatives()
    test_higher_order_terms()
    
    return matches

if __name__ == "__main__":
    debug_daiod_derivative_chain()
