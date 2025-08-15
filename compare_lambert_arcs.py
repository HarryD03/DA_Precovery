import numpy as np
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def compare_lambert_arcs():
    """
    Compare the two Lambert arcs in DAIOD to see why one works and the other fails
    """
    print("=== COMPARING LAMBERT ARCS IN DAIOD ===")
    
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
    
    # Calculate positions
    r1 = range_mag_guass[0] * obs_dir[:,0] + pos_obs[:,0]
    r2 = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
    r3 = range_mag_guass[2] * obs_dir[:,2] + pos_obs[:,2]
    
    dt1 = t[1] - t[0]  # Time for arc 1→2
    dt2 = t[2] - t[1]  # Time for arc 2→3
    
    print(f"Arc 1→2: dt = {dt1:.3f} s")
    print(f"Arc 2→3: dt = {dt2:.3f} s") 
    print(f"r1 = {r1}")
    print(f"r2 = {r2}")
    print(f"r3 = {r3}")
    
    # Test Arc 1→2: vary r2, see effect on v2
    def test_arc_1_to_2():
        print(f"\n=== ARC 1→2 (r1 fixed, r2 varies) ===")
        
        def v2_from_r2_perturbation(delta_r2):
            """v2 from arc 1→2 when r2 is perturbed"""
            r2_pert = r2 + delta_r2
            velocities = lambert_izzo(r1, r2_pert, dt1, mu, 0, prograde=True)
            v2 = velocities[0][:,1]
            return v2[2].cons() if hasattr(v2[2], 'cons') else v2[2]  # z-component
        
        # Finite difference derivative
        fd_deriv = approx_fprime([0.0], lambda x: v2_from_r2_perturbation(np.array([0, 0, x[0]])), 1e-6)[0]
        
        # DA derivative
        DA.init(4, 1)
        delta_r2z = DA(1)
        r2_da = array([r2[0], r2[1], r2[2] + delta_r2z])
        
        velocities_da = lambert_izzo(array(r1), r2_da, dt1, mu, 0, prograde=True)
        v2_da = velocities_da[0][:,1]
        da_deriv = v2_da[2].deriv(1).cons()
        
        error = abs(da_deriv - fd_deriv)
        rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
        
        print(f"∂v2z/∂r2z (FD): {fd_deriv}")
        print(f"∂v2z/∂r2z (DA): {da_deriv}")
        print(f"Relative error: {rel_error:.6f} ({rel_error*100:.3f}%)")
        
        return rel_error < 0.05  # 5% tolerance

    # Test Arc 2→3: vary r3, see effect on v2 (from arc 2→3)
    def test_arc_2_to_3():
        print(f"\n=== ARC 2→3 (r2 fixed, r3 varies) ===")
        
        def v2_from_r3_perturbation(delta_r3):
            """v2 from arc 2→3 when r3 is perturbed"""
            r3_pert = r3 + delta_r3
            velocities = lambert_izzo(r2, r3_pert, dt2, mu, 0, prograde=True)
            v2 = velocities[0][:,0]  # v2 is the first velocity of this arc
            return v2[2].cons() if hasattr(v2[2], 'cons') else v2[2]  # z-component
        
        # Finite difference derivative
        fd_deriv = approx_fprime([0.0], lambda x: v2_from_r3_perturbation(np.array([0, 0, x[0]])), 1e-6)[0]
        
        # DA derivative
        DA.init(4, 1)
        delta_r3z = DA(1)
        r3_da = array([r3[0], r3[1], r3[2] + delta_r3z])
        
        velocities_da = lambert_izzo(array(r2), r3_da, dt2, mu, 0, prograde=True)
        v2_da = velocities_da[0][:,0]
        da_deriv = v2_da[2].deriv(1).cons()
        
        error = abs(da_deriv - fd_deriv)
        rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
        
        print(f"∂v2z/∂r3z (FD): {fd_deriv}")
        print(f"∂v2z/∂r3z (DA): {da_deriv}")
        print(f"Relative error: {rel_error:.6f} ({rel_error*100:.3f}%)")
        
        return rel_error < 0.05  # 5% tolerance

    # Test arc geometry differences
    def analyze_arc_geometry():
        print(f"\n=== ARC GEOMETRY ANALYSIS ===")
        
        # Arc 1→2
        c1 = np.linalg.norm(r2 - r1)
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        s1 = 0.5 * (r1_mag + r2_mag + c1)
        
        print(f"Arc 1→2:")
        print(f"  |r1| = {r1_mag:.3f} km")
        print(f"  |r2| = {r2_mag:.3f} km")
        print(f"  |r2-r1| = {c1:.3f} km")
        print(f"  s = {s1:.3f} km")
        print(f"  dt = {dt1:.3f} s")
        
        # Arc 2→3  
        c2 = np.linalg.norm(r3 - r2)
        r3_mag = np.linalg.norm(r3)
        s2 = 0.5 * (r2_mag + r3_mag + c2)
        
        print(f"Arc 2→3:")
        print(f"  |r2| = {r2_mag:.3f} km")
        print(f"  |r3| = {r3_mag:.3f} km")
        print(f"  |r3-r2| = {c2:.3f} km")
        print(f"  s = {s2:.3f} km")
        print(f"  dt = {dt2:.3f} s")
        
        # Compute Lambert parameters
        L1 = np.sqrt(1 - c1/s1)
        L2 = np.sqrt(1 - c2/s2)
        T1 = np.sqrt(2*mu/(s1**3)) * dt1
        T2 = np.sqrt(2*mu/(s2**3)) * dt2
        
        print(f"\nLambert parameters:")
        print(f"Arc 1→2: L = {L1:.6f}, T = {T1:.6f}")
        print(f"Arc 2→3: L = {L2:.6f}, T = {T2:.6f}")
        
        # Check if one arc is near a singularity or special case
        print(f"\nSpecial case analysis:")
        print(f"Arc 1→2: L close to 0? {L1 < 0.1} (L={L1:.6f})")
        print(f"Arc 1→2: L close to 1? {L1 > 0.9} (L={L1:.6f})")
        print(f"Arc 2→3: L close to 0? {L2 < 0.1} (L={L2:.6f})")
        print(f"Arc 2→3: L close to 1? {L2 > 0.9} (L={L2:.6f})")
        
        # Check transfer angles
        cos_dnu1 = np.dot(r1, r2) / (r1_mag * r2_mag)
        cos_dnu2 = np.dot(r2, r3) / (r2_mag * r3_mag)
        dnu1 = np.arccos(np.clip(cos_dnu1, -1, 1))
        dnu2 = np.arccos(np.clip(cos_dnu2, -1, 1))
        
        print(f"Transfer angles:")
        print(f"Arc 1→2: Δν = {np.degrees(dnu1):.3f}°")
        print(f"Arc 2→3: Δν = {np.degrees(dnu2):.3f}°")
        
        return L1, L2, T1, T2, dnu1, dnu2

    # Test if the issue is in specific Lambert algorithm branches
    def test_lambert_algorithm_branches():
        print(f"\n=== LAMBERT ALGORITHM BRANCH ANALYSIS ===")
        
        # Check which algorithm path each arc takes
        print("This would require looking at the internal Lambert solver logic...")
        print("Key things to check:")
        print("- Does one arc use different convergence criteria?")
        print("- Does one arc hit different algorithmic branches?")
        print("- Are there different numerical conditioning issues?")
        
    # Run all tests
    arc1_ok = test_arc_1_to_2()
    arc2_ok = test_arc_2_to_3() 
    
    L1, L2, T1, T2, dnu1, dnu2 = analyze_arc_geometry()
    test_lambert_algorithm_branches()
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Arc 1→2 DA accuracy: {'✅ Good' if arc1_ok else '❌ Poor'}")
    print(f"Arc 2→3 DA accuracy: {'✅ Good' if arc2_ok else '❌ Poor'}")
    
    if arc1_ok and not arc2_ok:
        print("\n🎯 CONFIRMED: Arc 2→3 has DA derivative issues, Arc 1→2 is fine")
        print("Possible causes:")
        print(f"- Different Lambert parameters: L1={L1:.4f} vs L2={L2:.4f}")
        print(f"- Different transfer angles: {np.degrees(dnu1):.1f}° vs {np.degrees(dnu2):.1f}°")
        print(f"- Different time scales: {dt1:.1f}s vs {dt2:.1f}s")
        print("- Different algorithmic branches in Lambert solver")
        print("- Numerical conditioning differences")
    elif not arc1_ok and arc2_ok:
        print("\n🎯 SURPRISING: Arc 1→2 has issues, Arc 2→3 is fine")
    elif not arc1_ok and not arc2_ok:
        print("\n🎯 Both arcs have DA derivative issues")
    else:
        print("\n🎯 Both arcs work fine - issue must be elsewhere")
    
    return arc1_ok, arc2_ok

if __name__ == "__main__":
    compare_lambert_arcs()
