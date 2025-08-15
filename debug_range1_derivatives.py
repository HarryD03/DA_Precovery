import numpy as np
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def debug_daiod_range1_derivatives():
    """
    Investigate why range1 derivatives work well in complete DAIOD 
    but individual Lambert arcs show errors
    """
    print("=== DEBUGGING RANGE1 DERIVATIVES IN DAIOD CONTEXT ===")
    
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
    
    print(f"range_mag_guass: {range_mag_guass}")
    
    # Test 1: Complete DAIOD derivative w.r.t. range1
    def complete_daiod_range1_derivative():
        print("\n1. Complete DAIOD ∂DV/∂range1")
        
        def daiod_dv_z_from_range1(range1_val):
            """Complete DAIOD function: DV[2] as function of range1"""
            range_mag = np.array([range1_val, range_mag_guass[1], range_mag_guass[2]])
            
            # Convert to positions
            r1 = range_mag[0] * obs_dir[:,0] + pos_obs[:,0]
            r2 = range_mag[1] * obs_dir[:,1] + pos_obs[:,1]
            r3 = range_mag[2] * obs_dir[:,2] + pos_obs[:,2]
            
            # Arc 1→2
            velocities_12 = lambert_izzo(r1, r2, t[1] - t[0], mu, 0, prograde=True)
            v2_from_arc1 = velocities_12[0][:,1]
            
            # Arc 2→3  
            velocities_23 = lambert_izzo(r2, r3, t[2] - t[1], mu, 0, prograde=True)
            v2_from_arc2 = velocities_23[0][:,0]
            
            # Velocity difference
            dv = v2_from_arc2 - v2_from_arc1
            return dv[2].cons() if hasattr(dv[2], 'cons') else dv[2]
        
        # Finite difference
        fd_deriv = approx_fprime([range_mag_guass[0]], daiod_dv_z_from_range1, 1e-6)[0]
        
        # DA version (mimic complete DAIOD setup)
        DA.init(4, 3)
        range_da = array([
            range_mag_guass[0] + DA(1),  # range1 as DA(1)
            range_mag_guass[1] + DA(2),  # range2 as DA(2)
            range_mag_guass[2] + DA(3)   # range3 as DA(3)
        ])
        
        # Convert to positions
        r1_da = array([range_da[0] * obs_dir[i,0] + pos_obs[i,0] for i in range(3)])
        r2_da = array([range_da[1] * obs_dir[i,1] + pos_obs[i,1] for i in range(3)])
        r3_da = array([range_da[2] * obs_dir[i,2] + pos_obs[i,2] for i in range(3)])
        
        # Arc 1→2
        velocities_12_da = lambert_izzo(r1_da, r2_da, t[1] - t[0], mu, 0, prograde=True)
        v2_from_arc1_da = velocities_12_da[0][:,1]
        
        # Arc 2→3
        velocities_23_da = lambert_izzo(r2_da, r3_da, t[2] - t[1], mu, 0, prograde=True)
        v2_from_arc2_da = velocities_23_da[0][:,0]
        
        # Velocity difference
        dv_da = v2_from_arc2_da - v2_from_arc1_da
        da_deriv = dv_da[2].deriv(1).cons()  # ∂/∂range1 = ∂/∂DA(1)
        
        error = abs(da_deriv - fd_deriv)
        rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
        
        print(f"Complete DAIOD ∂DV_z/∂range1:")
        print(f"  FD: {fd_deriv}")
        print(f"  DA: {da_deriv}")
        print(f"  Error: {error}")
        print(f"  Rel error: {rel_error:.6f} ({rel_error*100:.3f}%)")
        
        return rel_error < 0.01

    # Test 2: Individual arc contribution analysis
    def analyze_individual_contributions():
        print("\n2. Individual Arc Contributions to ∂DV_z/∂range1")
        
        # range1 only affects arc 1→2, specifically the r1 input
        # DV = v2_from_arc2 - v2_from_arc1
        # ∂DV/∂range1 = 0 - ∂v2_from_arc1/∂range1
        
        def arc1_v2_contribution(range1_val):
            """v2 from arc 1→2 as function of range1"""
            r1 = range1_val * obs_dir[:,0] + pos_obs[:,0]
            r2 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
            
            velocities = lambert_izzo(r1, r2, t[1] - t[0], mu, 0, prograde=True)
            v2 = velocities[0][:,1]
            return v2[2].cons() if hasattr(v2[2], 'cons') else v2[2]
        
        # This should match our earlier "arc 1→2" test but with r1 varying instead of r2
        fd_deriv_arc1 = approx_fprime([range_mag_guass[0]], arc1_v2_contribution, 1e-6)[0]
        
        # The DAIOD derivative should be: ∂DV/∂range1 = -∂v2_arc1/∂range1
        expected_daiod_deriv = -fd_deriv_arc1
        
        print(f"Arc 1→2 contribution ∂v2/∂range1: {fd_deriv_arc1}")
        print(f"Expected DAIOD ∂DV/∂range1: {expected_daiod_deriv}")
        
        return expected_daiod_deriv

    # Test 3: Check if the difference is in how r1 vs r2 perturbations affect Lambert
    def compare_r1_vs_r2_sensitivity():
        print("\n3. Comparing r1 vs r2 Sensitivity in Lambert Arc 1→2")
        
        r1_base = range_mag_guass[0] * obs_dir[:,0] + pos_obs[:,0]
        r2_base = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
        dt = t[1] - t[0]
        
        # Test: perturb r1, measure effect on v2
        def v2_from_r1_perturbation(delta_r1z):
            r1_pert = r1_base + np.array([0, 0, delta_r1z])
            velocities = lambert_izzo(r1_pert, r2_base, dt, mu, 0, prograde=True)
            return velocities[0][:,1][2].cons() if hasattr(velocities[0][:,1][2], 'cons') else velocities[0][:,1][2]
        
        # Test: perturb r2, measure effect on v2
        def v2_from_r2_perturbation(delta_r2z):
            r2_pert = r2_base + np.array([0, 0, delta_r2z])
            velocities = lambert_izzo(r1_base, r2_pert, dt, mu, 0, prograde=True)
            return velocities[0][:,1][2].cons() if hasattr(velocities[0][:,1][2], 'cons') else velocities[0][:,1][2]
        
        # Finite difference derivatives
        fd_r1_sens = approx_fprime([0.0], v2_from_r1_perturbation, 1e-6)[0]
        fd_r2_sens = approx_fprime([0.0], v2_from_r2_perturbation, 1e-6)[0]
        
        print(f"Lambert arc 1→2 sensitivities:")
        print(f"  ∂v2z/∂r1z: {fd_r1_sens}")
        print(f"  ∂v2z/∂r2z: {fd_r2_sens}")
        print(f"  Ratio |∂v2z/∂r1z| / |∂v2z/∂r2z|: {abs(fd_r1_sens/fd_r2_sens) if fd_r2_sens != 0 else 'inf'}")
        
        # Now test DA accuracy for each
        DA.init(4, 1)
        
        # DA test for r1 perturbation
        delta_r1z_da = DA(1)
        r1_da = array([r1_base[0], r1_base[1], r1_base[2] + delta_r1z_da])
        velocities_da_r1 = lambert_izzo(r1_da, array(r2_base), dt, mu, 0, prograde=True)
        v2_da_r1 = velocities_da_r1[0][:,1][2]
        da_r1_sens = v2_da_r1.deriv(1).cons()
        
        # DA test for r2 perturbation  
        DA.init(4, 1)
        delta_r2z_da = DA(1)
        r2_da = array([r2_base[0], r2_base[1], r2_base[2] + delta_r2z_da])
        velocities_da_r2 = lambert_izzo(array(r1_base), r2_da, dt, mu, 0, prograde=True)
        v2_da_r2 = velocities_da_r2[0][:,1][2]
        da_r2_sens = v2_da_r2.deriv(1).cons()
        
        r1_error = abs(da_r1_sens - fd_r1_sens) / abs(fd_r1_sens) if fd_r1_sens != 0 else float('inf')
        r2_error = abs(da_r2_sens - fd_r2_sens) / abs(fd_r2_sens) if fd_r2_sens != 0 else float('inf')
        
        print(f"DA accuracy:")
        print(f"  r1 perturbation error: {r1_error:.6f} ({r1_error*100:.3f}%)")
        print(f"  r2 perturbation error: {r2_error:.6f} ({r2_error*100:.3f}%)")
        
        if r1_error < 0.01 and r2_error > 0.05:
            print("  🎯 FOUND IT! r1 perturbations have better DA accuracy than r2")
            return True
        elif r2_error < 0.01 and r1_error > 0.05:
            print("  🎯 FOUND IT! r2 perturbations have better DA accuracy than r1")
            return True
        else:
            print("  ❓ Both r1 and r2 perturbations have similar DA accuracy")
            return False

    # Run tests
    complete_ok = complete_daiod_range1_derivative()
    expected_daiod = analyze_individual_contributions()
    sensitivity_diff = compare_r1_vs_r2_sensitivity()
    
    print(f"\n=== SUMMARY ===")
    print(f"Complete DAIOD range1 derivative accurate: {'✅' if complete_ok else '❌'}")
    print(f"Sensitivity difference found: {'✅' if sensitivity_diff else '❌'}")
    
    if complete_ok:
        print("🎯 EXPLANATION: In complete DAIOD, range1 affects r1 of arc 1→2")
        print("   The DA accuracy depends on which position (r1 vs r2) is perturbed")
        print("   in the Lambert solver, not just which arc is used.")
    
    return complete_ok, sensitivity_diff

if __name__ == "__main__":
    debug_daiod_range1_derivatives()
