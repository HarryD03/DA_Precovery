"""
Simplified boundary sampling hypothesis test using real Apophis parameters.
This creates a focused test with realistic IOD parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.typing import NDArray
import sys
import os
from astropy.time import Time
from astropy import units as u
import astropy.coordinates as acoords

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import utils.iod as iod
import utils.time_reference as time_ref
from utils.Classical_IOD import sample_los, los_jacobian
from utils.lambert_izzo import lambert_izzo
import utils.post_process as post

def boundary_sample_los_simple(RA, DEC, sigma_ra, sigma_dec, method='corners'):
    """
    Simplified boundary sampling for line-of-sight vectors.
    """
    if method == 'corners':
        # Sample from 4 corners of 3-sigma box
        corners = [
            [3*sigma_ra, 3*sigma_dec],
            [3*sigma_ra, -3*sigma_dec], 
            [-3*sigma_ra, 3*sigma_dec],
            [-3*sigma_ra, -3*sigma_dec]
        ]
        
        # Randomly select one corner
        rng = default_rng()
        corner_idx = rng.integers(0, len(corners))
        d_ra, d_dec = corners[corner_idx]
        
    elif method == 'boundary':
        # Sample from boundary of 3-sigma ellipse
        rng = default_rng()
        theta = rng.uniform(0, 2*np.pi)
        d_ra = 3*sigma_ra * np.cos(theta)
        d_dec = 3*sigma_dec * np.sin(theta)
    
    # Convert to unit vector
    u0 = np.array([np.cos(DEC)*np.cos(RA), np.cos(DEC)*np.sin(RA), np.sin(DEC)])
    J = los_jacobian(RA, DEC)
    
    u = u0 + J @ np.array([d_ra, d_dec])
    return u / np.linalg.norm(u)


def simplified_boundary_test():
    """
    Simplified test using real Apophis parameters but fewer simulations.
    """
    print("Simplified Boundary vs Interior Sampling Test")
    print("="*60)
    
    # Use actual Apophis observation parameters from your main script
    obs_time0 = Time('2005-04-13 21:59:00', scale='utc')
    dt = np.array([0, 1, 2])  # days  
    obs_times = obs_time0 + (dt * u.day)
    
    # Get Earth positions (simplified - using actual code structure)
    epochs = Time(obs_times, scale='tdb')
    acoords.solar_system_ephemeris.set("builtin")
    
    try:
        # Get Earth position relative to Sun
        pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]
        pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
        pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]
        
        pos_obs = np.array([np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))[:3] 
                           for p_helio, v_helio in pv_earth_helio]).T
        
        print(f"Observer positions loaded: {pos_obs.shape}")
        
    except Exception as e:
        print(f"Failed to load observer positions: {e}")
        print("Using simplified test positions...")
        # Fallback to simplified positions
        pos_obs = np.array([
            [-1.47e8, -1.46e8, -1.45e8],
            [2.89e7, 2.91e7, 2.93e7], 
            [1.25e7, 1.26e7, 1.27e7]
        ])
    
    # Try to load actual Apophis observations
    try:
        eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')
        eph2, _ = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')
        
        ra = np.concatenate([eph['RA'], eph2['RA'][1:]]) * np.pi/180  # Convert to radians
        dec = np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * np.pi/180
        
        # Get observation uncertainties (convert from arcsec to radians)
        ra_sigma = (30 * eph['RA_3sigma'][1] / 3600) * np.pi/180 / 3  # 1-sigma in radians
        dec_sigma = (30 * eph['DEC_3sigma'][1] / 3600) * np.pi/180 / 3
        
        print(f"Loaded Apophis observations: RA={ra[1]:.6f}, DEC={dec[1]:.6f}")
        print(f"Uncertainties: σ_RA={ra_sigma:.2e}, σ_DEC={dec_sigma:.2e}")
        
    except Exception as e:
        print(f"Failed to load Apophis data: {e}")
        print("Using simplified test observations...")
        ra = np.array([0.1, 0.11, 0.12])
        dec = np.array([0.05, 0.06, 0.07])
        ra_sigma = 1e-6  # About 0.2 arcsec
        dec_sigma = 1e-6
    
    # Test parameters
    num_simulations = 100  # Reduced for faster testing
    mu = 1.32712440042e11  # km^3/s^2 (heliocentric)
    time_sec = (obs_times.mjd * u.day).to(u.s).value
    
    print(f"\nRunning {num_simulations} simulations for each method...")
    print(f"Time intervals: {time_sec}")
    
    # Test different sampling methods
    methods = {
        'interior': test_interior_sampling,
        'boundary_corners': test_boundary_sampling,
        'boundary_ellipse': test_ellipse_sampling
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()} SAMPLING:")
        try:
            positions, velocities, correlation = method_func(
                num_simulations, pos_obs, ra, dec, time_sec, 
                ra_sigma, dec_sigma, mu
            )
            results[method_name] = {
                'positions': positions,
                'velocities': velocities, 
                'correlation': correlation
            }
            print(f"   Correlation: {correlation:.4f}")
            
        except Exception as e:
            print(f"   Failed: {e}")
            results[method_name] = {'correlation': np.nan}
    
    # Create comparison plot
    create_simple_plots(results)
    
    # Summary
    print("\n" + "="*60)
    print("HYPOTHESIS TEST SUMMARY:")
    print("="*60)
    for method_name, result in results.items():
        corr = result['correlation']
        print(f"{method_name:20s}: {corr:+.4f}")
    
    # Test hypothesis
    interior_corr = results.get('interior', {}).get('correlation', np.nan)
    boundary_corr = results.get('boundary_corners', {}).get('correlation', np.nan)
    
    if not np.isnan(interior_corr) and not np.isnan(boundary_corr):
        if interior_corr > 0.5 and boundary_corr < -0.5:
            print("\n✓ HYPOTHESIS CONFIRMED:")
            print("  Interior sampling → Positive correlation")
            print("  Boundary sampling → Negative correlation")
        elif abs(interior_corr - boundary_corr) < 0.1:
            print("\n✗ HYPOTHESIS REJECTED:")
            print("  No significant difference between methods")
        else:
            print("\n? MIXED RESULTS:")
            print("  Partial support for hypothesis")
    

def test_interior_sampling(num_sim, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu):
    """Test interior (standard Monte Carlo) sampling."""
    results = []
    rng = default_rng(42)
    
    successful = 0
    attempts = 0
    
    while successful < num_sim and attempts < num_sim * 3:
        try:
            # Standard interior sampling
            u = np.zeros((3, 3))
            for j in range(3):
                # Gaussian sampling (interior)
                d_ra = rng.normal(0, ra_sigma)
                d_dec = rng.normal(0, dec_sigma)
                
                # 3-sigma clipping
                d_ra = np.clip(d_ra, -3*ra_sigma, 3*ra_sigma)
                d_dec = np.clip(d_dec, -3*dec_sigma, 3*dec_sigma)
                
                # Convert to unit vector
                u0 = np.array([np.cos(dec[j])*np.cos(ra[j]), 
                              np.cos(dec[j])*np.sin(ra[j]), 
                              np.sin(dec[j])])
                J = los_jacobian(ra[j], dec[j])
                u[:,j] = u0 + J @ np.array([d_ra, d_dec])
                u[:,j] = u[:,j] / np.linalg.norm(u[:,j])
            
            # IOD
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
            X_state = np.concatenate((r[:,1], v_2))
            results.append(X_state)
            successful += 1
            
        except:
            pass
        attempts += 1
    
    if len(results) == 0:
        return None, None, np.nan
        
    results = np.array(results)
    positions = results[:, :3]
    velocities = results[:, 3:]
    
    # Calculate range and range-rate
    ranges = np.linalg.norm(positions, axis=1)
    range_rates = np.sum(positions * velocities, axis=1) / ranges
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    return positions, velocities, correlation


def test_boundary_sampling(num_sim, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu):
    """Test boundary sampling (corners)."""
    results = []
    
    successful = 0
    attempts = 0
    
    while successful < num_sim and attempts < num_sim * 3:
        try:
            # Boundary corner sampling
            u = np.zeros((3, 3))
            for j in range(3):
                u[:,j] = boundary_sample_los_simple(ra[j], dec[j], ra_sigma, dec_sigma, 'corners')
            
            # IOD
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
            X_state = np.concatenate((r[:,1], v_2))
            results.append(X_state)
            successful += 1
            
        except:
            pass
        attempts += 1
    
    if len(results) == 0:
        return None, None, np.nan
        
    results = np.array(results)
    positions = results[:, :3]
    velocities = results[:, 3:]
    
    ranges = np.linalg.norm(positions, axis=1)
    range_rates = np.sum(positions * velocities, axis=1) / ranges
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    return positions, velocities, correlation


def test_ellipse_sampling(num_sim, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu):
    """Test ellipse boundary sampling."""
    results = []
    
    successful = 0
    attempts = 0
    
    while successful < num_sim and attempts < num_sim * 3:
        try:
            u = np.zeros((3, 3))
            for j in range(3):
                u[:,j] = boundary_sample_los_simple(ra[j], dec[j], ra_sigma, dec_sigma, 'boundary')
            
            # IOD
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
            X_state = np.concatenate((r[:,1], v_2))
            results.append(X_state)
            successful += 1
            
        except:
            pass
        attempts += 1
    
    if len(results) == 0:
        return None, None, np.nan
        
    results = np.array(results)
    positions = results[:, :3]
    velocities = results[:, 3:]
    
    ranges = np.linalg.norm(positions, axis=1)
    range_rates = np.sum(positions * velocities, axis=1) / ranges
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    return positions, velocities, correlation


def create_simple_plots(results):
    """Create simple comparison plots."""
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['correlation'])}
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(1, len(valid_results), figsize=(5*len(valid_results), 5))
    if len(valid_results) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green']
    
    for i, (method_name, result) in enumerate(valid_results.items()):
        positions = result['positions']
        velocities = result['velocities']
        correlation = result['correlation']
        
        ranges = np.linalg.norm(positions, axis=1)
        range_rates = np.sum(positions * velocities, axis=1) / ranges
        
        axes[i].scatter(ranges, range_rates, alpha=0.6, s=10, c=colors[i])
        axes[i].set_title(f'{method_name.replace("_", " ").title()}\nCorr: {correlation:+.4f}')
        axes[i].set_xlabel('Range (km)')
        axes[i].set_ylabel('Range Rate (km/s)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simplified_boundary_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved as 'simplified_boundary_test.png'")


if __name__ == "__main__":
    simplified_boundary_test()
