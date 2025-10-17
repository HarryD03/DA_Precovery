#!/usr/bin/env python3
"""
Debug DA polynomial evaluation to understand why we get huge range spans.
"""

import sys
sys.path.append('.')

import numpy as np
import utils.iod as iod
import utils.post_process as utils_post
import astropy.constants as ac
import astropy.coordinates as acoords
import astropy.units as u
from astropy.time import Time, TimeDelta

def debug_da_evaluation():
    """Debug DA polynomial evaluation scales"""
    print("="*70)
    print("DEBUG: DA POLYNOMIAL EVALUATION")
    print("="*70)
    
    # Same setup as robust test
    obs_time0 = Time('2005-04-13 21:59:00', scale='utc')
    obs_times = Time([obs_time0 - TimeDelta(1, format='jd'), obs_time0, obs_time0 + TimeDelta(1, format='jd')])
    
    mu = ac.G * ac.M_sun
    mu = mu.to('km**3 / s**2').value
    
    # Observer positions
    pv_earth_helio = {}
    for i, obs_time in enumerate(obs_times):
        earth = acoords.get_body_barycentric_posvel('earth', obs_time)
        p_helio = earth[0]
        v_helio = earth[1] 
        pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))
    
    pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T
    
    # Observations
    eph, _ = utils_post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')
    eph2, _ = utils_post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')
    
    ra_nom = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad).value
    dec_nom = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad).value
    
    ra_sigma = 1e-5   # ~2 arcsec
    dec_sigma = 1e-5  # ~2 arcsec
    
    time_sec = (obs_times.mjd * u.day).to(u.s).value
    
    print(f"Input parameters:")
    print(f"  RA sigma: {ra_sigma} rad = {ra_sigma * 180/np.pi * 3600:.2f} arcsec")
    print(f"  DEC sigma: {dec_sigma} rad = {dec_sigma * 180/np.pi * 3600:.2f} arcsec")
    print(f"  3*sigma perturbation: {3*ra_sigma * 180/np.pi * 3600:.2f} arcsec")
    
    # Get DA result
    da_result = iod.DAIOD_full(ra_nom, dec_nom, ra_sigma, dec_sigma, 
                              pos_obs, time_sec, mu, order=6, prograde=True)
    
    if da_result is None:
        print("DA result is None!")
        return
    
    print(f"\nDA polynomial analysis:")
    
    # Evaluate at nominal (should be close to zero perturbation)
    nominal_eval = da_result.eval(np.zeros(6))
    print(f"  Nominal evaluation (zeros): {nominal_eval}")
    
    # Evaluate at small perturbations
    small_pert = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    small_eval = da_result.eval(small_pert)
    print(f"  Small perturbation (0.1): {small_eval}")
    
    # Evaluate at corners (±1)
    corner_1 = np.array([1, 1, 1, 1, 1, 1])
    corner_eval_1 = da_result.eval(corner_1)
    print(f"  Corner (+1,+1,...): {corner_eval_1}")
    
    corner_2 = np.array([-1, -1, -1, -1, -1, -1])
    corner_eval_2 = da_result.eval(corner_2)
    print(f"  Corner (-1,-1,...): {corner_eval_2}")
    
    # Check range and range-rate scales
    print(f"\nRange analysis:")
    ranges = [np.linalg.norm(nominal_eval[:3]), 
              np.linalg.norm(small_eval[:3]), 
              np.linalg.norm(corner_eval_1[:3]), 
              np.linalg.norm(corner_eval_2[:3])]
    print(f"  Nominal range: {ranges[0]:.2e} km")
    print(f"  Small pert range: {ranges[1]:.2e} km")
    print(f"  Corner +1 range: {ranges[2]:.2e} km")
    print(f"  Corner -1 range: {ranges[3]:.2e} km")
    
    print(f"\nRange-rate analysis:")
    range_rates = [np.linalg.norm(nominal_eval[3:]), 
                   np.linalg.norm(small_eval[3:]), 
                   np.linalg.norm(corner_eval_1[3:]), 
                   np.linalg.norm(corner_eval_2[3:])]
    print(f"  Nominal range-rate: {range_rates[0]:.2e} km/s")
    print(f"  Small pert range-rate: {range_rates[1]:.2e} km/s")
    print(f"  Corner +1 range-rate: {range_rates[2]:.2e} km/s")
    print(f"  Corner -1 range-rate: {range_rates[3]:.2e} km/s")
    
    # Expected physical scales
    print(f"\nExpected physical scales:")
    # For Apophis-like asteroid at ~1 AU
    expected_range = 150e6  # ~1 AU in km
    expected_range_rate = 30  # ~30 km/s orbital velocity
    print(f"  Expected range: ~{expected_range:.0e} km")
    print(f"  Expected range-rate: ~{expected_range_rate:.0f} km/s")
    
    # Check if DA results are reasonable
    print(f"\nSanity check:")
    if ranges[0] > 1e10:  # > 10^10 km is unreasonable
        print("  WARNING: Nominal range is unreasonably large!")
    if range_rates[0] > 1e6:  # > 10^6 km/s is unreasonable
        print("  WARNING: Nominal range-rate is unreasonably large!")
        
    # Check coefficient magnitude
    print(f"\nDA polynomial coefficient analysis:")
    # The DA result should be a state vector [r1, r2, r3, v1, v2, v3]
    # Let's examine the first coefficient (position component)
    try:
        # Print some coefficient information if accessible
        print(f"  DA result type: {type(da_result)}")
        if hasattr(da_result, 'cons'):
            print(f"  Constant term: {da_result.cons()}")
        
        # Check derivative scaling
        unit_pert = np.zeros(6)
        unit_pert[0] = 1.0  # Perturb first variable by 1
        unit_eval = da_result.eval(unit_pert)
        
        print(f"  Unit perturbation in RA1:")
        print(f"    Position change: {np.linalg.norm(unit_eval[:3] - nominal_eval[:3]):.2e} km")
        print(f"    Velocity change: {np.linalg.norm(unit_eval[3:] - nominal_eval[3:]):.2e} km/s")
        
        # This should correspond to 3*sigma physical perturbation
        expected_angular_change = 3 * ra_sigma  # 3*sigma in radians
        print(f"    Expected angular perturbation: {expected_angular_change:.2e} rad = {expected_angular_change * 180/np.pi * 3600:.2f} arcsec")
        
    except Exception as e:
        print(f"  Error analyzing coefficients: {e}")

if __name__ == "__main__":
    debug_da_evaluation()
