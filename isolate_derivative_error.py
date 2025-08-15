import numpy as np
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def isolate_derivative_error():
    """
    Since Householder iteration is correct, isolate where the derivative error occurs
    in the DAIOD chain: range_mag → positions → Lambert → velocities → difference
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
    
    print("=== ISOLATING DERIVATIVE ERROR LOCATION ===")
    print(f"Testing with range_mag: {range_mag_guass}")
    
    # Step 1: Test simple position vector derivatives
    def test_position_derivatives():
        print("\n1. Testing Position Vector Derivatives")
        
        def position_z_from_range3(range3):
            """z-component of position 3 from range 3"""
            return range3 * obs_dir[2,2] + pos_obs[2,2]
        
        # Analytical derivative (should be exactly obs_dir[2,2])
        analytical = obs_dir[2,2]
        
        # DA derivative
        DA.init(4, 1)
        range3_da = range_mag_guass[2] + DA(1)
        pos_z_da = position_z_from_range3(range3_da)
        da_deriv = pos_z_da.deriv(1).cons()
        
        # Finite difference
        fd_deriv = approx_fprime([range_mag_guass[2]], 
                                lambda x: position_z_from_range3(x[0]), 1e-6)[0]
        
        print(f"Analytical ∂pos_z/∂range3: {analytical}")
        print(f"DA ∂pos_z/∂range3: {da_deriv}")
        print(f"FD ∂pos_z/∂range3: {fd_deriv}")
        
        error = abs(da_deriv - analytical)
        print(f"DA vs Analytical error: {error}")
        
        if error < 1e-12:
            print("✅ Position derivatives are correct")
            return True
        else:
            print("❌ Position derivatives have errors!")
            return False
    
    # Step 2: Test single Lambert arc derivatives  
    def test_single_lambert_derivatives():
        print("\n2. Testing Single Lambert Arc Derivatives")
        
        # Fix first two positions, vary third position z-component
        r1 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]  # Fixed
        
        def lambert_v1_z_from_r2z(r2z):
            """z-component of v1 from Lambert arc, varying r2 z-component"""
            r2 = np.array([
                range_mag_guass[2] * obs_dir[0,2] + pos_obs[0,2],  # Fixed x,y
                range_mag_guass[2] * obs_dir[1,2] + pos_obs[1,2],
                r2z  # Variable z
            ])
            velocities = lambert_izzo(r1, r2, t[2] - t[1], mu, 0, prograde=True)
            v1 = velocities[0][:,0]
            return v1[2].cons() if hasattr(v1[2], 'cons') else v1[2]
        
        # Current r2 z-coordinate
        r2z_nominal = range_mag_guass[2] * obs_dir[2,2] + pos_obs[2,2]
        
        # Finite difference
        fd_deriv = approx_fprime([r2z_nominal], lambert_v1_z_from_r2z, 1e-6)[0]
        
        # DA version
        DA.init(4, 1)
        r2z_da = r2z_nominal + DA(1)
        
        def lambert_v1_z_da(r2z_da_val):
            r2_da = array([
                range_mag_guass[2] * obs_dir[0,2] + pos_obs[0,2],
                range_mag_guass[2] * obs_dir[1,2] + pos_obs[1,2], 
                r2z_da_val
            ])
            velocities = lambert_izzo(array(r1), r2_da, t[2] - t[1], mu, 0, prograde=True)
            v1 = velocities[0][:,0]
            return v1[2]
        
        v1z_da = lambert_v1_z_da(r2z_da)
        da_deriv = v1z_da.deriv(1).cons()
        
        print(f"FD ∂(Lambert_v1z)/∂r2z: {fd_deriv}")
        print(f"DA ∂(Lambert_v1z)/∂r2z: {da_deriv}")
        
        error = abs(da_deriv - fd_deriv)
        rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
        
        print(f"Absolute error: {error}")
        print(f"Relative error: {rel_error}")
        
        if rel_error < 0.01:  # 1% tolerance
            print("✅ Single Lambert derivatives are acceptable")
            return True
        else:
            print("❌ Single Lambert derivatives have significant errors!")
            return False
    
    # Step 3: Test velocity difference operation
    def test_velocity_difference():
        print("\n3. Testing Velocity Difference Operation")
        
        # Test: (v_a - v_b) derivative = v_a_derivative - v_b_derivative
        DA.init(4, 1)
        x = 5.0 + DA(1)
        
        # Simple test function
        v_a = x**2 + 3*x  # Derivative = 2x + 3 = 13 at x=5
        v_b = x + 1       # Derivative = 1
        diff = v_a - v_b  # Derivative should be 12
        
        analytical_deriv = 12.0
        da_deriv = diff.deriv(1).cons()
        
        print(f"Analytical ∂(v_a - v_b)/∂x: {analytical_deriv}")
        print(f"DA ∂(v_a - v_b)/∂x: {da_deriv}")
        
        error = abs(da_deriv - analytical_deriv)
        if error < 1e-12:
            print("✅ Velocity difference operations are correct")
            return True
        else:
            print("❌ Velocity difference operations have errors!")
            return False
    
    # Step 4: Test the full chain with known problematic derivative
    def test_full_chain_step_by_step():
        print("\n4. Testing Full Chain Step by Step")
        
        # This is the derivative that showed 63% error
        def dv_z_component(range3):
            """Full DAIOD chain focusing on range3 → dv_z"""
            
            # Step 4a: range3 → position 3
            r3 = range3 * obs_dir[:,2] + pos_obs[:,2]
            
            # Step 4b: positions → velocities via Lambert arcs
            r1 = range_mag_guass[0] * obs_dir[:,0] + pos_obs[:,0]
            r2 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
            
            # Arc 1: r1 → r2
            velocities_12 = lambert_izzo(r1, r2, t[1] - t[0], mu, 0, prograde=True)
            v2_from_arc1 = velocities_12[0][:,1]
            
            # Arc 2: r2 → r3 (this involves range3!)
            velocities_23 = lambert_izzo(r2, r3, t[2] - t[1], mu, 0, prograde=True)
            v2_from_arc2 = velocities_23[0][:,0]
            
            # Step 4c: velocity difference
            dv = v2_from_arc2 - v2_from_arc1
            return dv[2].cons() if hasattr(dv[2], 'cons') else dv[2]
        
        # Test each sub-step
        range3_test = range_mag_guass[2]
        
        # Sub-step derivatives
        print("\n4a. Testing range3 → r3 derivative:")
        dr3_drange3 = obs_dir[2,2]  # Should be exact
        print(f"∂r3z/∂range3 = {dr3_drange3}")
        
        print("\n4b. Testing r3 → Lambert v2 derivative:")
        def v2_from_r3z(r3z):
            r2 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
            r3 = np.array([
                range_mag_guass[2] * obs_dir[0,2] + pos_obs[0,2],
                range_mag_guass[2] * obs_dir[1,2] + pos_obs[1,2],
                r3z
            ])
            velocities = lambert_izzo(r2, r3, t[2] - t[1], mu, 0, prograde=True)
            return velocities[0][:,0][2].cons() if hasattr(velocities[0][:,0][2], 'cons') else velocities[0][:,0][2]
        
        r3z_nominal = range3_test * obs_dir[2,2] + pos_obs[2,2]
        dv2_dr3z = approx_fprime([r3z_nominal], v2_from_r3z, 1e-6)[0]
        print(f"∂v2z/∂r3z = {dv2_dr3z}")
        
        print("\n4c. Chain rule prediction:")
        chain_rule_pred = dr3_drange3 * dv2_dr3z
        print(f"Chain rule: ∂v2z/∂range3 = {chain_rule_pred}")
        
        # Full function derivative
        fd_full = approx_fprime([range3_test], dv_z_component, 1e-6)[0]
        print(f"Full FD: ∂(dv_z)/∂range3 = {fd_full}")
        
        print(f"Chain rule vs FD error: {abs(chain_rule_pred - fd_full)}")
        
        return abs(chain_rule_pred - fd_full) < 1e-6
    
    # Run all tests
    results = []
    results.append(("Position derivatives", test_position_derivatives()))
    results.append(("Single Lambert derivatives", test_single_lambert_derivatives())) 
    results.append(("Velocity difference", test_velocity_difference()))
    results.append(("Full chain analysis", test_full_chain_step_by_step()))
    
    print(f"\n=== SUMMARY ===")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Determine where the error is
    if not results[1][1]:  # Single Lambert derivatives failed
        print("\n🎯 ERROR LOCATION: Lambert solver derivatives (not Householder)")
        print("The issue is in the derivative computation within Lambert solver,")
        print("possibly in the vector operations or function composition.")
    elif not results[0][1]:
        print("\n🎯 ERROR LOCATION: Basic DA vector operations")
    elif not results[2][1]:
        print("\n🎯 ERROR LOCATION: DA arithmetic operations")
    else:
        print("\n🎯 ERROR LOCATION: Complex function composition")
        print("Individual components work, but composition amplifies errors")

if __name__ == "__main__":
    isolate_derivative_error()
