import numpy as np
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def compare_lambert_parameters():
    """
    Compare Lambert parameters between the working Vallado example 
    and the problematic DAIOD geometry
    """
    print("=== COMPARING LAMBERT PARAMETERS ===")
    
    # Vallado example (working perfectly)
    print("\n1. VALLADO EXAMPLE (working):")
    R1_vallado = np.array([15945.34, 0.0, 0.0])
    R2_vallado = np.array([12214.83399, 10249.46731, 0.0])
    DT_vallado = 76 * 60
    MU = 3.986e5
    
    c_vallado = np.linalg.norm(R2_vallado - R1_vallado)
    r1_mag_vallado = np.linalg.norm(R1_vallado)
    r2_mag_vallado = np.linalg.norm(R2_vallado)
    s_vallado = 0.5 * (r1_mag_vallado + r2_mag_vallado + c_vallado)
    L_vallado = np.sqrt(1 - c_vallado / s_vallado)
    T_vallado = np.sqrt((2 * MU) / (s_vallado ** 3)) * DT_vallado
    
    print(f"  |r1| = {r1_mag_vallado:.3f} km")
    print(f"  |r2| = {r2_mag_vallado:.3f} km") 
    print(f"  |r2-r1| = {c_vallado:.3f} km")
    print(f"  s = {s_vallado:.3f} km")
    print(f"  dt = {DT_vallado:.3f} s")
    print(f"  L = {L_vallado:.6f}")
    print(f"  T = {T_vallado:.6f}")
    
    # Transfer angle
    cos_dnu_vallado = np.dot(R1_vallado, R2_vallado) / (r1_mag_vallado * r2_mag_vallado)
    dnu_vallado = np.arccos(np.clip(cos_dnu_vallado, -1, 1))
    print(f"  Transfer angle = {np.degrees(dnu_vallado):.3f}°")
    
    # DAIOD example (problematic)
    print("\n2. DAIOD EXAMPLE (problematic):")
    
    # Get DAIOD data
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
    
    from utils.iod import Guass_8th_seed
    _, _, range_mag_guass, _ = Guass_8th_seed(pos_obs, obs_dir, t, mu=MU)
    
    r1_daiod = range_mag_guass[0] * obs_dir[:,0] + pos_obs[:,0]
    r2_daiod = range_mag_guass[1] * obs_dir[:,1] + pos_obs[:,1]
    r3_daiod = range_mag_guass[2] * obs_dir[:,2] + pos_obs[:,2]
    
    # Arc 1→2
    dt1_daiod = t[1] - t[0]
    c1_daiod = np.linalg.norm(r2_daiod - r1_daiod)
    r1_mag_daiod = np.linalg.norm(r1_daiod)
    r2_mag_daiod = np.linalg.norm(r2_daiod)
    s1_daiod = 0.5 * (r1_mag_daiod + r2_mag_daiod + c1_daiod)
    L1_daiod = np.sqrt(1 - c1_daiod / s1_daiod)
    T1_daiod = np.sqrt((2 * MU) / (s1_daiod ** 3)) * dt1_daiod
    
    cos_dnu1_daiod = np.dot(r1_daiod, r2_daiod) / (r1_mag_daiod * r2_mag_daiod)
    dnu1_daiod = np.arccos(np.clip(cos_dnu1_daiod, -1, 1))
    
    print(f"  Arc 1→2:")
    print(f"    |r1| = {r1_mag_daiod:.3f} km")
    print(f"    |r2| = {r2_mag_daiod:.3f} km")
    print(f"    |r2-r1| = {c1_daiod:.3f} km")
    print(f"    s = {s1_daiod:.3f} km")
    print(f"    dt = {dt1_daiod:.3f} s")
    print(f"    L = {L1_daiod:.6f}")
    print(f"    T = {T1_daiod:.6f}")
    print(f"    Transfer angle = {np.degrees(dnu1_daiod):.3f}°")
    
    # Arc 2→3
    dt2_daiod = t[2] - t[1]
    c2_daiod = np.linalg.norm(r3_daiod - r2_daiod)
    r3_mag_daiod = np.linalg.norm(r3_daiod)
    s2_daiod = 0.5 * (r2_mag_daiod + r3_mag_daiod + c2_daiod)
    L2_daiod = np.sqrt(1 - c2_daiod / s2_daiod)
    T2_daiod = np.sqrt((2 * MU) / (s2_daiod ** 3)) * dt2_daiod
    
    cos_dnu2_daiod = np.dot(r2_daiod, r3_daiod) / (r2_mag_daiod * r3_mag_daiod)
    dnu2_daiod = np.arccos(np.clip(cos_dnu2_daiod, -1, 1))
    
    print(f"  Arc 2→3:")
    print(f"    |r2| = {r2_mag_daiod:.3f} km")
    print(f"    |r3| = {r3_mag_daiod:.3f} km")
    print(f"    |r3-r2| = {c2_daiod:.3f} km")
    print(f"    s = {s2_daiod:.3f} km")
    print(f"    dt = {dt2_daiod:.3f} s")
    print(f"    L = {L2_daiod:.6f}")
    print(f"    T = {T2_daiod:.6f}")
    print(f"    Transfer angle = {np.degrees(dnu2_daiod):.3f}°")
    
    # Analysis
    print(f"\n3. PARAMETER COMPARISON:")
    print(f"Lambert L parameter:")
    print(f"  Vallado:    L = {L_vallado:.6f}")
    print(f"  DAIOD Arc1: L = {L1_daiod:.6f}")
    print(f"  DAIOD Arc2: L = {L2_daiod:.6f}")
    
    print(f"Lambert T parameter:")
    print(f"  Vallado:    T = {T_vallado:.6f}")
    print(f"  DAIOD Arc1: T = {T1_daiod:.6f}")
    print(f"  DAIOD Arc2: T = {T2_daiod:.6f}")
    
    print(f"Transfer angles:")
    print(f"  Vallado:    {np.degrees(dnu_vallado):.3f}°")
    print(f"  DAIOD Arc1: {np.degrees(dnu1_daiod):.3f}°")
    print(f"  DAIOD Arc2: {np.degrees(dnu2_daiod):.3f}°")
    
    # Check for problematic regimes
    print(f"\n4. PROBLEMATIC REGIME ANALYSIS:")
    
    def check_regime(L, T, name):
        print(f"  {name}:")
        if L > 0.99:
            print(f"    ⚠️  L = {L:.6f} > 0.99 (near-parabolic, numerically challenging)")
        elif L < 0.01:
            print(f"    ⚠️  L = {L:.6f} < 0.01 (very elliptical, numerically challenging)")
        else:
            print(f"    ✅ L = {L:.6f} (normal regime)")
            
        if T > 2*np.pi:
            print(f"    ⚠️  T = {T:.6f} > 2π (multi-revolution)")
        elif T < 0.1:
            print(f"    ⚠️  T = {T:.6f} < 0.1 (very short time)")
        else:
            print(f"    ✅ T = {T:.6f} (normal regime)")
    
    check_regime(L_vallado, T_vallado, "Vallado")
    check_regime(L1_daiod, T1_daiod, "DAIOD Arc1")
    check_regime(L2_daiod, T2_daiod, "DAIOD Arc2")
    
    # Test a DAIOD-like geometry with Lambert test
    print(f"\n5. TESTING DAIOD GEOMETRY WITH LAMBERT TEST:")
    
    def test_lambert_accuracy(r1, r2, dt, name):
        print(f"  Testing {name}...")
        
        # Test DA vs FD accuracy
        def v2_z_from_r2z_perturbation(delta_r2z):
            r2_pert = r2 + np.array([0, 0, delta_r2z])
            velocities = lambert_izzo(r1, r2_pert, dt, MU, 0, prograde=True)
            v2 = velocities[0][:,1]
            return v2[2].cons() if hasattr(v2[2], 'cons') else v2[2]
        
        from scipy.optimize import approx_fprime
        fd_deriv = approx_fprime([0.0], v2_z_from_r2z_perturbation, 1e-6)[0]
        
        # DA derivative
        DA.init(4, 1)
        delta_r2z_da = DA(1)
        r2_da = array([r2[0], r2[1], r2[2] + delta_r2z_da])
        
        velocities_da = lambert_izzo(array(r1), r2_da, dt, MU, 0, prograde=True)
        v2_da = velocities_da[0][:,1]
        da_deriv = v2_da[2].deriv(1).cons()
        
        error = abs(da_deriv - fd_deriv)
        rel_error = error / abs(fd_deriv) if fd_deriv != 0 else float('inf')
        
        print(f"    DA derivative: {da_deriv}")
        print(f"    FD derivative: {fd_deriv}")
        print(f"    Relative error: {rel_error:.6f} ({rel_error*100:.3f}%)")
        
        if rel_error > 0.01:
            print(f"    ❌ Poor accuracy in {name}")
        else:
            print(f"    ✅ Good accuracy in {name}")
            
        return rel_error < 0.01
    
    vallado_ok = test_lambert_accuracy(R1_vallado, R2_vallado, DT_vallado, "Vallado geometry")
    daiod1_ok = test_lambert_accuracy(r1_daiod, r2_daiod, dt1_daiod, "DAIOD Arc 1→2")
    daiod2_ok = test_lambert_accuracy(r2_daiod, r3_daiod, dt2_daiod, "DAIOD Arc 2→3")
    
    print(f"\n=== CONCLUSION ===")
    if vallado_ok and not (daiod1_ok and daiod2_ok):
        print("🎯 CONFIRMED: Lambert solver works perfectly for Vallado geometry")
        print("   but has issues with DAIOD orbital geometry.")
        print("   This suggests numerical conditioning problems in specific")
        print("   parameter regimes rather than fundamental DA algorithm issues.")
    
    return vallado_ok, daiod1_ok, daiod2_ok

if __name__ == "__main__":
    compare_lambert_parameters()
