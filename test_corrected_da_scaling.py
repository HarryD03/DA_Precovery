#!/usr/bin/env python3
"""
Test corrected DA evaluation with proper scaling to match 3σ perturbations.
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
import itertools

def test_corrected_da_evaluation():
    """Test DA evaluation with corrected scaling"""
    
    print("="*70)
    print("CORRECTED DA EVALUATION TEST")
    print("="*70)
    
    # Same setup as previous tests
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
    
    # Get DA result
    da_result = iod.DAIOD_full(ra_nom, dec_nom, ra_sigma, dec_sigma, 
                              pos_obs, time_sec, mu, order=6, prograde=True)
    
    if da_result is None:
        print("DA result is None!")
        return
    
    # Nominal evaluation
    nominal_eval = da_result.eval(np.zeros(6))
    nominal_range = np.linalg.norm(nominal_eval[:3])
    
    print(f"Nominal state: {nominal_eval}")
    print(f"Nominal range: {nominal_range:.2e} km")
    
    # Expected 3σ perturbation
    expected_3sigma_perturbation = nominal_range * 3 * ra_sigma
    print(f"Expected 3σ position perturbation: {expected_3sigma_perturbation:.2e} km")
    
    # Method 1: Original full domain evaluation (problematic)
    print(f"\n1. ORIGINAL FULL DOMAIN EVALUATION:")
    corner_full = da_result.eval(np.ones(6))
    residual_full = corner_full - nominal_eval
    perturbation_full = np.linalg.norm(residual_full[:3])
    print(f"   Full domain (+1,+1,...) perturbation: {perturbation_full:.2e} km")
    print(f"   Scaling factor vs expected: {perturbation_full/expected_3sigma_perturbation:.1f}×")
    
    # Method 2: Find correct scaling factor
    print(f"\n2. FINDING CORRECT SCALING FACTOR:")
    target_scaling = expected_3sigma_perturbation / perturbation_full
    print(f"   Required scaling factor: {target_scaling:.4f}")
    
    # Method 3: Use scaled domain for proper 3σ perturbations
    print(f"\n3. CORRECTED SCALED DOMAIN EVALUATION:")
    
    corrected_results = []
    scaling_factor = target_scaling
    
    # Generate scaled corners
    corner_values = [-scaling_factor, scaling_factor]
    
    for combination in itertools.product(corner_values, repeat=6):
        eval_point = np.array(combination)
        try:
            da_state = da_result.eval(eval_point)
            corrected_results.append(da_state)
        except Exception as e:
            print(f"DA evaluation failed at {combination}: {e}")
            continue
    
    corrected_results = np.array(corrected_results)
    
    print(f"   Generated {len(corrected_results)} corrected evaluations")
    
    # Analyze corrected results
    if len(corrected_results) > 0:
        positions = corrected_results[:, :3]
        velocities = corrected_results[:, 3:]
        
        ranges = np.linalg.norm(positions, axis=1)
        range_rates = np.sum(positions * velocities, axis=1) / ranges
        
        # Check perturbation magnitudes
        position_perturbations = [np.linalg.norm(pos - nominal_eval[:3]) for pos in positions]
        velocity_perturbations = [np.linalg.norm(vel - nominal_eval[3:]) for vel in velocities]
        
        print(f"\n4. CORRECTED RESULTS ANALYSIS:")
        print(f"   Position perturbations: {np.min(position_perturbations):.2e} to {np.max(position_perturbations):.2e} km")
        print(f"   Expected ~3σ: {expected_3sigma_perturbation:.2e} km")
        print(f"   Velocity perturbations: {np.min(velocity_perturbations):.2e} to {np.max(velocity_perturbations):.2e} km/s")
        
        print(f"\n   Range span: {ranges.max() - ranges.min():.2e} km")
        print(f"   Range-rate span: {range_rates.max() - range_rates.min():.2e} km/s")
        
        # Correlation analysis
        if len(ranges) > 1:
            correlation = np.corrcoef(ranges, range_rates)[0,1]
            print(f"   Range vs range-rate correlation: {correlation:+.4f}")
            
            # Check for box-like structure
            if abs(correlation) < 0.5:
                print(f"   ✓ Low correlation suggests box-like structure!")
            else:
                print(f"   ? Correlation still significant")
        
        # Compare with pointwise evaluation expected scale
        expected_range_span = 2 * expected_3sigma_perturbation  # roughly ±3σ
        actual_range_span = ranges.max() - ranges.min()
        
        print(f"\n5. BOX-LIKE STRUCTURE CHECK:")
        print(f"   Expected range span (6σ): {2*expected_3sigma_perturbation:.2e} km")
        print(f"   Actual range span: {actual_range_span:.2e} km")
        print(f"   Ratio: {actual_range_span/(2*expected_3sigma_perturbation):.2f}")
        
        if 0.5 < actual_range_span/(2*expected_3sigma_perturbation) < 2.0:
            print(f"   ✓ Range span is reasonable for 6σ box structure")
        else:
            print(f"   ? Range span differs from expected box structure")
    
    return corrected_results, scaling_factor

if __name__ == "__main__":
    results, scaling = test_corrected_da_evaluation()
    print(f"\nRecommended DA evaluation scaling factor: {scaling:.6f}")
    print(f"Use this factor instead of ±1 for realistic 3σ perturbations")
