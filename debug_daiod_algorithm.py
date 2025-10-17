"""
Debug where the scaling amplification occurs in the DAIOD algorithm.
Since DA construction is correct, the problem must be in the algorithm steps.
"""

import numpy as np
from daceypy import DA, array
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import utils.iod as iod
import utils.time_reference as time_ref
from astropy.time import Time
from astropy import units as u
import astropy.constants as ac
import astropy.coordinates as acoords

def debug_daiod_steps():
    """
    Step through DAIOD algorithm to find where scaling amplification occurs.
    """
    print("="*70)
    print("DAIOD ALGORITHM STEP-BY-STEP DEBUG")
    print("="*70)
    
    # Use realistic Apophis-like parameters
    obs_time0 = Time('2005-04-13 21:59:00', scale='utc')
    obs_times = Time([obs_time0 - TimeDelta(1, format='jd'), obs_time0, obs_time0 + TimeDelta(1, format='jd')])
    
    mu = ac.G * ac.M_sun
    mu = mu.to('km**3 / s**2').value
    epochs = Time(obs_times, scale='tdb')
    acoords.solar_system_ephemeris.set("builtin")
    
    pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]
    pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
    pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]
    
    for i, (p_helio, v_helio) in enumerate(pv_earth_helio):
        pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))

    pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T
    
    # Nominal observations
    RA_rad = np.array([0.759857367, 0.949802813, 1.12254676])
    DEC_rad = np.array([-0.15329974, -0.210726107, -0.263633267])
    RA_sigma = 1e-5
    DEC_sigma = 1e-5
    time_sec = (obs_times.mjd * u.day).to(u.s).value
    
    print(f"Input parameters:")
    print(f"  RA_rad: {RA_rad}")
    print(f"  DEC_rad: {DEC_rad}")
    print(f"  RA_sigma: {RA_sigma:.2e} rad")
    print(f"  DEC_sigma: {DEC_sigma:.2e} rad")
    print(f"  Expected 3σ angular pert: {3*RA_sigma:.2e} rad")
    
    # Step 1: Gauss initial estimate
    print(f"\n" + "="*50)
    print("STEP 1: GAUSS INITIAL ESTIMATE")
    print("="*50)
    
    i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad)
    positions, ranges, range_mags, v_2 = iod.Guass_8th_seed(pos_obs, i_rho, time_sec, mu)
    
    print(f"Gauss results:")
    print(f"  range_mags: {range_mags}")
    print(f"  Expected range scale: {np.mean(range_mags):.2e} km")
    
    # Step 2: DAIOD_1 - Range refinement
    print(f"\n" + "="*50)
    print("STEP 2: DAIOD_1 - RANGE REFINEMENT")  
    print("="*50)
    
    DA.init(6, 3)  # 3 range variables
    try:
        range_mag_L1, Jacobian_dv = iod.DAIOD_1Scipy_invert(
            range_mags, i_rho, time_sec, 6, pos_obs, mu, tol=1e-9, prograde_bool=True
        )
        print(f"DAIOD_1 successful:")
        print(f"  range_mag_L1 constants: {range_mag_L1.cons()}")
        print(f"  Jacobian shape: {Jacobian_dv.shape}")
        
        # Check range perturbations from Step 1
        DA_range_vars = np.array([1, 0, 0])  # Perturb only first range
        range_eval = range_mag_L1.eval(DA_range_vars)
        range_pert = np.abs(range_eval - range_mag_L1.cons())
        print(f"  Range perturbation from DA(1)=1: {range_pert} km")
        
    except Exception as e:
        print(f"DAIOD_1 failed: {e}")
        return
    
    # Step 3: DAIOD_2 - Angular DA expansion  
    print(f"\n" + "="*50)
    print("STEP 3: DAIOD_2 - ANGULAR DA EXPANSION")
    print("="*50)
    
    DA.init(6, 6)  # 6 angular variables
    
    # Construct angular DA variables (this is where scaling should be correct)
    RA_DA = array([RA_rad[i] + 3*RA_sigma*DA(i + 1) for i in range(3)])
    DEC_DA = array([DEC_rad[i] + 3*DEC_sigma*DA(i + 4) for i in range(3)])
    
    print(f"Angular DA construction:")
    print(f"  RA_DA constants: {[ra.cons() for ra in RA_DA]}")
    print(f"  DEC_DA constants: {[dec.cons() for dec in DEC_DA]}")
    
    # Test angular perturbations before DAIOD_2
    test_angular_pert = np.array([1, 0, 0, 0, 0, 0])  # Only RA[0] perturbed
    RA_test = [RA_DA[i].eval(test_angular_pert) for i in range(3)]
    DEC_test = [DEC_DA[i].eval(test_angular_pert) for i in range(3)]
    
    RA_angular_pert = np.abs(np.array(RA_test) - np.array([ra.cons() for ra in RA_DA]))
    DEC_angular_pert = np.abs(np.array(DEC_test) - np.array([dec.cons() for dec in DEC_DA]))
    
    print(f"  Pre-DAIOD_2 angular perturbations:")
    print(f"    RA perturbation from DA(1)=1: {RA_angular_pert}")
    print(f"    DEC perturbation from DA(1)=1: {DEC_angular_pert}")
    print(f"    Expected: [3e-05, 0, 0] for RA, [0, 0, 0] for DEC")
    
    try:
        # This is where the problem might be!
        range_mag_DAangles = iod.DAIOD_2(
            range_mag_L1, RA_DA, DEC_DA, time_sec, 6, 
            r_obs_heliocentric=pos_obs, mu=mu, J=Jacobian_dv, prograde_bool=True
        )
        
        print(f"DAIOD_2 successful:")
        print(f"  range_mag_DAangles constants: {range_mag_DAangles.cons()}")
        
        # Test range response to angular perturbations
        range_angular_test = range_mag_DAangles.eval(test_angular_pert)
        range_angular_response = np.abs(range_angular_test - range_mag_DAangles.cons())
        
        print(f"  Range response to angular DA(1)=1: {range_angular_response}")
        print(f"  Expected scale (rough): ~few km per observation")
        
        # This is the critical test - check if DAIOD_2 is amplifying correctly
        expected_range_response = np.mean(range_mags) * 3 * RA_sigma  # Linear approximation
        amplification_factor = np.max(range_angular_response) / expected_range_response
        
        print(f"  Expected range response: {expected_range_response:.2e} km")
        print(f"  Amplification factor: {amplification_factor:.1f}×")
        
        if amplification_factor > 10:
            print(f"  ⚠️  LARGE AMPLIFICATION DETECTED IN DAIOD_2!")
            print(f"      This is likely where the over-scaling occurs")
        
    except Exception as e:
        print(f"DAIOD_2 failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Final state construction
    print(f"\n" + "="*50)
    print("STEP 4: FINAL STATE CONSTRUCTION")
    print("="*50)
    
    try:
        # Line-of-sight vectors with DA
        i_rho_DA = array(time_ref.create_da_los_vectors(RA_DA, DEC_DA))
        range_vec = i_rho_DA * range_mag_DAangles
        pos_vec = range_vec + pos_obs
        
        print(f"Final position construction:")
        print(f"  pos_vec constants: {pos_vec.cons()}")
        
        # Test final position response to angular perturbations
        pos_final_test = np.array([[pos_vec[i,j].eval(test_angular_pert) for j in range(3)] for i in range(3)])
        pos_final_response = np.abs(pos_final_test - pos_vec.cons())
        pos_final_max_response = np.max(pos_final_response)
        
        print(f"  Position response to angular DA(1)=1: max = {pos_final_max_response:.2e} km")
        
        # Expected positional response
        expected_pos_response = np.mean(range_mags) * 3 * RA_sigma
        final_amplification = pos_final_max_response / expected_pos_response
        
        print(f"  Expected position response: {expected_pos_response:.2e} km")
        print(f"  Final amplification factor: {final_amplification:.1f}×")
        
        if final_amplification > 50:
            print(f"  ⚠️  EXCESSIVE FINAL AMPLIFICATION!")
            print(f"      Your trigonometric analysis was correct - this is the source of over-scaling")
        
    except Exception as e:
        print(f"Final state construction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from astropy.time import TimeDelta
    debug_daiod_steps()
