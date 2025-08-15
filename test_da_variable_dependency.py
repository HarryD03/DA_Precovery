import numpy as np
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def test_da_variable_dependency():
    """
    Test if DA errors are specific to certain DA variable indices
    """
    print("=== TESTING DA VARIABLE DEPENDENCY ===")
    
    # Test data (same as DAIOD)
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
    
    # Test 1: Individual DA variables (one at a time)
    def test_individual_da_variables():
        print("\n1. Testing Individual DA Variables")
        
        for var_idx in range(3):
            print(f"\n--- Testing DA variable {var_idx+1} (range{var_idx+1}) ---")
            
            # Initialize DA with only one variable
            DA.init(4, 1)
            
            # Create range array with one DA variable
            range_da = np.array([
                range_mag_guass[0] + (DA(1) if var_idx == 0 else 0),
                range_mag_guass[1] + (DA(1) if var_idx == 1 else 0), 
                range_mag_guass[2] + (DA(1) if var_idx == 2 else 0)
            ])
            
            # Simple test: position vector computation
            def simple_position_test():
                pos3_z = range_da[2] * obs_dir[2,2] + pos_obs[2,2]
                if var_idx == 2:  # Should have derivative
                    expected_deriv = obs_dir[2,2]
                    actual_deriv = pos3_z.deriv(1).cons() if hasattr(pos3_z, 'deriv') else 0
                    error = abs(actual_deriv - expected_deriv)
                    print(f"  Position derivative error: {error}")
                    return error < 1e-12
                else:  # Should have zero derivative
                    actual_deriv = pos3_z.deriv(1).cons() if hasattr(pos3_z, 'deriv') else 0
                    error = abs(actual_deriv)
                    print(f"  Position derivative (should be 0): {actual_deriv}")
                    return error < 1e-12
                    
            # Lambert test: single arc with one DA variable
            def lambert_single_variable_test():
                # Fixed positions
                r1 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
                
                # r2 with potential DA dependency
                r2 = array([
                    range_da[2] * obs_dir[0,2] + pos_obs[0,2],
                    range_da[2] * obs_dir[1,2] + pos_obs[1,2],
                    range_da[2] * obs_dir[2,2] + pos_obs[2,2]
                ])
                
                velocities = lambert_izzo(array(r1), r2, t[2] - t[1], mu, 0, prograde=True)
                v1 = velocities[0][:,0]
                
                if var_idx == 2:  # Should have derivative  
                    da_deriv = v1[2].deriv(1).cons()
                    
                    # Compare with finite difference
                    def fd_func(delta):
                        r2_fd = np.array([
                            (range_mag_guass[2] + delta) * obs_dir[0,2] + pos_obs[0,2],
                            (range_mag_guass[2] + delta) * obs_dir[1,2] + pos_obs[1,2],
                            (range_mag_guass[2] + delta) * obs_dir[2,2] + pos_obs[2,2]
                        ])
                        velocities_fd = lambert_izzo(r1, r2_fd, t[2] - t[1], mu, 0, prograde=True)
                        return velocities_fd[0][:,0][2].cons() if hasattr(velocities_fd[0][:,0][2], 'cons') else velocities_fd[0][:,0][2]
                    
                    fd_deriv = approx_fprime([0.0], fd_func, 1e-6)[0]
                    error = abs(da_deriv - fd_deriv)
                    rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
                    
                    print(f"  Lambert DA derivative: {da_deriv}")
                    print(f"  Lambert FD derivative: {fd_deriv}")
                    print(f"  Relative error: {rel_error:.6f} ({rel_error*100:.2f}%)")
                    
                    return rel_error < 0.01  # 1% tolerance
                else:  # Should have zero derivative
                    da_deriv = v1[2].deriv(1).cons() if hasattr(v1[2], 'deriv') else 0
                    print(f"  Lambert derivative (should be 0): {da_deriv}")
                    return abs(da_deriv) < 1e-12
            
            pos_ok = simple_position_test()
            lambert_ok = lambert_single_variable_test()
            
            print(f"  Position test: {'✅' if pos_ok else '❌'}")
            print(f"  Lambert test: {'✅' if lambert_ok else '❌'}")
            
            if not lambert_ok and var_idx == 2:
                print(f"  ⚠️  DA variable {var_idx+1} shows Lambert derivative errors!")
            elif not lambert_ok:
                print(f"  ⚠️  DA variable {var_idx+1} shows unexpected derivative behavior!")
                
    # Test 2: DA variable ordering effect
    def test_da_variable_ordering():
        print(f"\n2. Testing DA Variable Ordering Effects")
        
        # Test same computation with different DA variable assignments
        test_cases = [
            ("Standard order", [1, 2, 3]),  # range1=DA(1), range2=DA(2), range3=DA(3)
            ("Reversed order", [3, 2, 1]),  # range1=DA(3), range2=DA(2), range3=DA(1) 
            ("Rotated order", [2, 3, 1])   # range1=DA(2), range2=DA(3), range3=DA(1)
        ]
        
        results = []
        
        for case_name, da_assignment in test_cases:
            print(f"\n--- {case_name}: range_mag → DA({da_assignment}) ---")
            
            DA.init(4, 3)
            range_da = array([
                range_mag_guass[0] + DA(da_assignment[0]),
                range_mag_guass[1] + DA(da_assignment[1]),
                range_mag_guass[2] + DA(da_assignment[2])
            ])
            
            # Simple Lambert test
            r2 = array([
                range_da[2] * obs_dir[0,2] + pos_obs[0,2],
                range_da[2] * obs_dir[1,2] + pos_obs[1,2], 
                range_da[2] * obs_dir[2,2] + pos_obs[2,2]
            ])
            
            r1 = array([
                range_da[1] * obs_dir[0,1] + pos_obs[0,1],
                range_da[1] * obs_dir[1,1] + pos_obs[1,1],
                range_da[1] * obs_dir[2,1] + pos_obs[2,1]
            ])
            
            velocities = lambert_izzo(r1, r2, t[2] - t[1], mu, 0, prograde=True)
            v1 = velocities[0][:,0]
            
            # Extract derivative w.r.t. range3 (which is DA(da_assignment[2]))
            da_var_for_range3 = da_assignment[2] 
            da_deriv = v1[2].deriv(da_var_for_range3).cons()
            
            print(f"  DA derivative ∂v1z/∂range3: {da_deriv}")
            results.append((case_name, da_deriv, da_var_for_range3))
        
        # Compare results
        print(f"\n--- Comparison ---")
        baseline = results[0][1]  # Standard order result
        for case_name, deriv, da_var in results:
            diff = abs(deriv - baseline)
            print(f"  {case_name} (using DA({da_var})): {deriv}, diff from baseline: {diff}")
            
        # Check if higher-numbered DA variables have more errors
        high_da_vars = [result for result in results if result[2] == 3]  # DA(3)
        low_da_vars = [result for result in results if result[2] == 1]   # DA(1)
        
        if high_da_vars and low_da_vars:
            high_avg = np.mean([r[1] for r in high_da_vars])  
            low_avg = np.mean([r[1] for r in low_da_vars])
            print(f"\n  Average for DA(3): {high_avg}")
            print(f"  Average for DA(1): {low_avg}")
            print(f"  Difference: {abs(high_avg - low_avg)}")
            
            if abs(high_avg - low_avg) > 1e-6:
                print("  ⚠️  Higher-numbered DA variables show different behavior!")
                return False
            else:
                print("  ✅ DA variable numbering doesn't affect results")
                return True
    
    # Run tests
    test_individual_da_variables()
    ordering_ok = test_da_variable_ordering()
    
    return ordering_ok

if __name__ == "__main__":
    test_da_variable_dependency()
