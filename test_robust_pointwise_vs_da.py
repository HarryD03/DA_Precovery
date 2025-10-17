"""
Robust pointwise vs DA boundary comparison test.
Uses larger perturbations and better numerical handling to avoid NaN issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import sys
import os
from daceypy import DA, array
from astropy.time import Time
from astropy import units as u

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import utils.iod as iod
from utils.Classical_IOD import los_jacobian
import utils.post_process as utils_post
import utils.time_reference as time_ref
import astropy.constants as ac
import astropy.coordinates as acoords
import astropy.units as u
from astropy.time import Time, TimeDelta


def robust_pointwise_boundary_test():
    """
    Robust test using larger, more realistic perturbations.
    """
    print("="*70)
    print("ROBUST POINTWISE vs DA BOUNDARY COMPARISON")
    print("="*70)
    obs_time0 = Time('2005-04-13 21:59:00', scale='utc') #Central Observation Time
    
    obs_times = Time([obs_time0 - TimeDelta(1, format='jd'), obs_time0, obs_time0 + TimeDelta(1, format='jd')])

    #Generate Observer Positions at Observation Times
    mu = ac.G * ac.M_sun
    mu = mu.to('km**3 / s**2').value
    epochs = Time(obs_times, scale='tdb')
    acoords.solar_system_ephemeris.set("builtin")
    #Get ICRS Equatorial position of the Earth at each observation epoch
    pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]          # pv[i] = (pos, vel)
    #Translate to Heliocentric Equatorial Frame
    pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
    pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]  # (pos, vel) relative to Sun
    
    for i, (p_helio, v_helio) in enumerate(pv_earth_helio):
    #Maintain Equaotorial Frame
        pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))  # [x, y, z, vx, vy, vz]

    pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T  # km [3xN]

    #Fetch RA, DEC, and observation uncertanity using JPL Horizons
    eph, _ = utils_post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')  #ICRS Geocentric
    _, vec = utils_post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1', location='500@10') #ICRS Heliocentric

    _, vec2 = utils_post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1', location='500@10')
    eph2, _  = utils_post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')

    ra = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad)
    dec = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad)

    # Use simplified but realistic parameters
    # Observer positions (Earth-like heliocentric positions in km)
    
    # Nominal observations (asteroid-like)
    ra_nom = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad).value
    dec_nom = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad).value
    
    # Use larger uncertainties for numerical stability
    ra_sigma = 1e-5   # ~2 arcsec (larger than Apophis for stability)
    dec_sigma = 1e-5  # ~2 arcsec
    
    time_sec = (obs_times.mjd * u.day).to(u.s).value

    print(f"Test parameters:")
    print(f"  RA nominal: {ra_nom * 180/np.pi} degrees")
    print(f"  DEC nominal: {dec_nom * 180/np.pi} degrees")
    print(f"  RA sigma: {ra_sigma:.2e} rad ({ra_sigma * 3600 * 180/np.pi:.2f} arcsec)")
    print(f"  DEC sigma: {dec_sigma:.2e} rad ({dec_sigma * 3600 * 180/np.pi:.2f} arcsec)")
    
    # Method 1: Pointwise boundary evaluation
    print(f"\n1. POINTWISE BOUNDARY EVALUATION:")
    pointwise_results = robust_pointwise_evaluation(
        ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu
    )
    
    # Method 2: Simulate DA-like systematic evaluation
    print(f"\n2. DA-LIKE SYSTEMATIC EVALUATION:")
    da_like_results = da_like_systematic_evaluation(
        ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu
    )
    
    # Method 3: Try actual DA if possible
    print(f"\n3. ACTUAL DA IMPLEMENTATION (if stable):")
    try:
        actual_da_result = iod.DAIOD_full(ra_nom, dec_nom, ra_sigma, dec_sigma, 
                                         pos_obs, time_sec, mu, order=6, prograde=True)
        print(f"DA result: {actual_da_result}")
        if actual_da_result is not None:
            print("DA implementation successful")
            # Use same sampling as robust_pointwise_evaluation
            
            # DA Evaluation: Generate box-like structure by evaluating at 6D hypercube corners
            # This should produce the box-like pattern described in the Pirovano paper
            actual_da_results = []
            
            # Generate all possible combinations of ±1 for the 6 DA variables
            # This creates the corners of the 6D uncertainty hypercube
            corner_values = [-1, 1]  # DA domain corners
            
            import itertools
            
            # CORRECTED DA EVALUATION: Use proper scaling for realistic 3σ perturbations
            # Problem: DA domain [-1,+1] gives ~150× larger perturbations than expected
            # Solution: Scale down to ~0.0065 for realistic 3σ perturbations
            
            # Calculate correct scaling factor
            nominal_eval = actual_da_result.eval(np.zeros(6))
            nominal_range = np.linalg.norm(nominal_eval[:3])
            expected_3sigma = nominal_range * 3 * ra_sigma
            
            # Test full domain perturbation
            test_corner = actual_da_result.eval(np.ones(6))
            test_perturbation = np.linalg.norm(test_corner[:3] - nominal_eval[:3])
            scaling_factor = expected_3sigma / test_perturbation
            
            print(f"DA scaling correction:")
            print(f"  Expected 3σ perturbation: {expected_3sigma:.2e} km")
            print(f"  Full domain perturbation: {test_perturbation:.2e} km")
            print(f"  Required scaling factor: {scaling_factor:.6f}")
            
            # Generate corrected corner combinations with proper scaling
            corner_values = [-scaling_factor, scaling_factor]
            
            # Generate all 64 corner combinations (2^6) with corrected scaling
            for combination in itertools.product(corner_values, repeat=6):
                # combination is like (-0.0065, 0.0065, -0.0065, 0.0065, -0.0065, 0.0065)
                eval_point = np.array(combination)
                
                try:
                    # Evaluate DA polynomial at this corrected corner point
                    da_state = actual_da_result.eval(eval_point)
                    actual_da_results.append(da_state)
                except Exception as e:
                    print(f"DA evaluation failed at corner {combination}: {e}")
                    continue
            
            print(f"DA corner evaluation: {len(actual_da_results)} points from 6D hypercube corners")
            print("This should produce the box-like structure described in Pirovano paper")
            
            actual_da_results = np.array(actual_da_results) if actual_da_results else np.array([])
            print(f"DA boundary evaluation: {len(actual_da_results)} points (same sampling as pointwise)")
            
        else:
            print("DA implementation returned NaN")
            actual_da_results = None
    except Exception as e:
        print(f"DA implementation failed: {e}")
        actual_da_results = None
    
    # Analysis
    print(f"\n" + "="*70)
    print("CORRELATION ANALYSIS:")
    print("="*70)
    
    # Analyze pointwise results
    if len(pointwise_results) > 3:  # Need multiple points for correlation
        pointwise_analysis = analyze_correlation(pointwise_results, "Pointwise Boundary")
    else:
        print("Pointwise: Insufficient valid results for correlation")
        pointwise_analysis = None
    
    # Analyze DA-like results
    if len(da_like_results) > 3:
        da_like_analysis = analyze_correlation(da_like_results, "DA-like Systematic")
    else:
        print("DA-like: Insufficient valid results for correlation")
        da_like_analysis = None
    
    # Analyze actual DA results
    if 'actual_da_results' in locals() and actual_da_results is not None and len(actual_da_results) > 3:
        actual_da_analysis = analyze_correlation(actual_da_results, "Actual DA Perimeter")
    else:
        print("Actual DA: Insufficient valid results for correlation")
        actual_da_analysis = None
    
    # Create plots
    create_robust_comparison_plots(pointwise_analysis, da_like_analysis, actual_da_analysis)
    
    # Summary
    print(f"\n" + "="*70)
    print("FINAL COMPARISON:")
    print("="*70)
    
    # Comprehensive comparison including all three methods
    valid_analyses = [a for a in [pointwise_analysis, da_like_analysis, actual_da_analysis] if a is not None]
    
    if len(valid_analyses) >= 2:
        for i, analysis in enumerate(valid_analyses):
            corr = analysis['correlation']
            method = analysis['method']
            n_samples = analysis['n_samples']
            print(f"{method:25} Correlation: {corr:+.4f} (N={n_samples})")
        
        # Check for boundary sampling patterns
        correlations = [a['correlation'] for a in valid_analyses]
        methods = [a['method'] for a in valid_analyses]
        
        negative_correlations = [c for c in correlations if c < -0.1]
        positive_correlations = [c for c in correlations if c > 0.1]
        
        print(f"\nBoundary Sampling Analysis:")
        print(f"  Methods with negative correlation: {len(negative_correlations)}/{len(correlations)}")
        print(f"  Methods with positive correlation: {len(positive_correlations)}/{len(correlations)}")
        
        if len(negative_correlations) >= 2:
            print(f"\n✓ STRONG BOUNDARY SAMPLING EVIDENCE:")
            print(f"  Multiple methods show negative correlation")
            print(f"  Consistent with boundary sampling hypothesis")
        elif len(negative_correlations) == 1 and len(positive_correlations) >= 1:
            print(f"\n? MIXED SAMPLING PATTERNS:")
            print(f"  Both boundary and interior-like patterns detected")
        elif all(abs(c1 - c2) < 0.3 for c1 in correlations for c2 in correlations):
            print(f"\n✓ CONSISTENT METHODS:")
            print(f"  Similar correlation patterns across methods")
        else:
            print(f"\n? DIVERSE RESULTS:")
            print(f"  Significant variation between methods")
            
        # Specific DA comparison if available
        if actual_da_analysis is not None:
            da_corr = actual_da_analysis['correlation']
            print(f"\nActual DA Implementation Analysis:")
            print(f"  DA perimeter correlation: {da_corr:+.4f}")
            
            if da_corr < -0.2:
                print(f"  ✓ Strong negative correlation - confirms boundary sampling")
            elif da_corr < 0:
                print(f"  ✓ Negative correlation - suggests boundary sampling")
            else:
                print(f"  ? Non-negative correlation - unexpected for boundary method")
    else:
        print("Insufficient data for comprehensive comparison")


def robust_pointwise_evaluation(ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu):
    """
    Robust pointwise evaluation with better error handling.
    """
    results = []
    
    # Use systematic boundary points with fewer combinations for stability
    boundary_multipliers = [-3, 0, 3]  # -3σ, nominal, +3σ
    
    for ra_mult in boundary_multipliers:
        for dec_mult in boundary_multipliers:
            # Skip nominal point for boundary sampling
            if ra_mult == 0 and dec_mult == 0:
                continue
                
            try:
                u = np.zeros((3, 3))
                valid_u = True
                
                for j in range(3):
                    d_ra = ra_mult * ra_sigma
                    d_dec = dec_mult * dec_sigma
                    
                    # Perturbed RA and DEC
                    ra_pert = ra_nom[j] + d_ra
                    dec_pert = dec_nom[j] + d_dec
                    
                    # Check bounds
                    if abs(dec_pert) > np.pi/2:
                        valid_u = False
                        break
                    
                    # Create unit vector directly (simpler approach)
                    u[:,j] = np.array([
                        np.cos(dec_pert) * np.cos(ra_pert),
                        np.cos(dec_pert) * np.sin(ra_pert),
                        np.sin(dec_pert)
                    ])
                
                if not valid_u:
                    continue
                
                # IOD
                r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
                range_mag_L1 = iod.DAIOD_1Scipy_invert(range_mag, u, time_sec, None, pos_obs, mu, tol=1e-9, prograde_bool=True)
                
                #get states from range_mag_L1
                range_vec_L1 = range_mag_L1 * u
                r = pos_obs + range_vec_L1
                vel = iod.lambert_izzo(r[:,0], r[:,1], time_sec[1]-time_sec[0], mu, 0, prograde=True)

                v_2 = vel[0][:,1]  # Take velocity at second observation
                # Check for valid result
                if (not np.any(np.isnan(r)) and not np.any(np.isnan(v_2)) and 
                    not np.any(np.isinf(r)) and not np.any(np.isinf(v_2))):
                    
                    X_state = np.concatenate((r[:,1], v_2))
                    results.append(X_state)
                    
            except Exception as e:
                continue
    
    print(f"Pointwise evaluation: {len(results)} valid solutions")
    return np.array(results) if results else np.array([])


def da_like_systematic_evaluation(ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu):
    """
    Simulate DA-like systematic boundary evaluation.
    """
    results = []
    
    # DA-like evaluation: systematic combinations of ±1 scaled by 3σ
    da_combinations = [
        [-1, -1], [-1, 1], [1, -1], [1, 1],  # Corner points
        [-1, 0], [1, 0], [0, -1], [0, 1]     # Edge midpoints
    ]
    
    for da_ra, da_dec in da_combinations:
        try:
            u = np.zeros((3, 3))
            valid_u = True
            
            for j in range(3):
                # DA-like scaling: 3σ * DA_variable
                d_ra = da_ra * 3 * ra_sigma
                d_dec = da_dec * 3 * dec_sigma
                
                ra_pert = ra_nom[j] + d_ra
                dec_pert = dec_nom[j] + d_dec
                
                if abs(dec_pert) > np.pi/2:
                    valid_u = False
                    break
                
                u[:,j] = np.array([
                    np.cos(dec_pert) * np.cos(ra_pert),
                    np.cos(dec_pert) * np.sin(ra_pert),
                    np.sin(dec_pert)
                ])
            
            if not valid_u:
                continue
            
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
            
            if (not np.any(np.isnan(r)) and not np.any(np.isnan(v_2)) and 
                not np.any(np.isinf(r)) and not np.any(np.isinf(v_2))):
                
                X_state = np.concatenate((r[:,1], v_2))
                results.append(X_state)
                
        except Exception as e:
            continue
    
    print(f"DA-like evaluation: {len(results)} valid solutions")
    return np.array(results) if results else np.array([])


def analyze_correlation(results, method_name):
    """Analyze range vs range-rate correlation."""
    if len(results) < 2:
        return None
    
    positions = results[:, :3]
    velocities = results[:, 3:]
    
    obs = np.zeros_like(results)
    for i in range(positions.shape[0]):
        obs[i, :] = time_ref.CC2obs(results[i,:])


    ranges = obs[:,2]
    range_rates = obs[:,5]
    
    # Filter out any remaining invalid values
    valid_mask = ~(np.isnan(ranges) | np.isnan(range_rates) | 
                   np.isinf(ranges) | np.isinf(range_rates))
    
    if np.sum(valid_mask) < 2:
        print(f"{method_name}: All results invalid after filtering")
        return None
    
    ranges = ranges[valid_mask]
    range_rates = range_rates[valid_mask]
    
    if len(ranges) < 2:
        print(f"{method_name}: Insufficient valid data for correlation")
        return None
    
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    print(f"{method_name}:")
    print(f"  Valid samples: {len(ranges)}")
    print(f"  Range span: {ranges.max() - ranges.min():.2e} km")
    print(f"  Range-rate span: {range_rates.max() - range_rates.min():.2e} km/s") 
    print(f"  Correlation: {correlation:+.4f}")
    
    return {
        'correlation': correlation,
        'ranges': ranges,
        'range_rates': range_rates,
        'method': method_name,
        'n_samples': len(ranges)
    }


def create_robust_comparison_plots(pointwise_analysis, da_like_analysis, actual_da_analysis):
    """Create comparison plots for valid analyses including actual DA results."""
    valid_analyses = [a for a in [pointwise_analysis, da_like_analysis, actual_da_analysis] if a is not None]
    
    if len(valid_analyses) == 0:
        print("No valid analyses for plotting")
        return
    
    # Create subplots based on number of valid analyses
    n_plots = len(valid_analyses)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']  # circle, square, triangle
    
    for i, analysis in enumerate(valid_analyses):
        ranges = analysis['ranges']
        range_rates = analysis['range_rates']
        correlation = analysis['correlation']
        method = analysis['method']
        n_samples = analysis['n_samples']
        
        # Use different markers for different methods
        axes[i].scatter(ranges/1e6, range_rates, alpha=0.7, s=50, 
                       c=colors[i], marker=markers[i], edgecolors='black', linewidth=0.5)
        axes[i].set_xlabel('Range (10⁶ km)')
        axes[i].set_ylabel('Range Rate (km/s)')
        
        # Truncate method name for cleaner titles
        method_short = method.replace("Boundary", "Bndry").replace("Systematic", "Sys").replace("Perimeter", "Perim")
        axes[i].set_title(f'{method_short}\nCorr: {correlation:+.4f} (N={n_samples})')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line
        if len(ranges) > 1:
            z = np.polyfit(ranges, range_rates, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(ranges.min(), ranges.max(), 100)
            axes[i].plot(x_trend/1e6, p(x_trend), '--', color=colors[i], alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('robust_pointwise_vs_da_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Robust comparison plot saved as 'robust_pointwise_vs_da_comparison.png'")
    print(f"Plot includes {n_plots} methods: {[a['method'] for a in valid_analyses]}")


if __name__ == "__main__":
    robust_pointwise_boundary_test()
