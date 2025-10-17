"""
Accurate DA-like sampling test that replicates polynomial evaluation at domain boundaries.

The key insight: DA methods evaluate polynomials at specific grid points (±1),
not random sampling. This creates systematic boundary evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.typing import NDArray
import sys
import os
import itertools

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import utils.iod as iod
from utils.Classical_IOD import sample_los, los_jacobian

def da_like_sampling(RA, DEC, sigma_ra, sigma_dec, n_variables=6):
    """
    Replicate DA polynomial evaluation approach.
    
    DA methods evaluate polynomials at ±1 for each DA variable.
    This creates systematic boundary sampling, not random sampling.
    """
    # Create all combinations of ±1 for n_variables (like DA domain evaluation)
    # For RA/DEC we need 6 DA variables: 3 for RA, 3 for DEC (one per observation)
    domain_points = list(itertools.product([-1, 1], repeat=n_variables))
    
    results = []
    for domain_point in domain_points:
        # Map domain point to RA/DEC perturbations
        # First 3 variables for RA, next 3 for DEC
        d_ra = domain_point[0] * 3 * sigma_ra  # 3-sigma scaling like DA
        d_dec = domain_point[1] * 3 * sigma_dec
        
        # Convert to unit vector (same as other methods)
        u0 = np.array([np.cos(DEC)*np.cos(RA), np.cos(DEC)*np.sin(RA), np.sin(DEC)])
        J = los_jacobian(RA, DEC)
        
        u = u0 + J @ np.array([d_ra, d_dec])
        u = u / np.linalg.norm(u)
        
        results.append(u)
    
    return results


def systematic_boundary_test():
    """
    Test using systematic DA-like boundary evaluation vs random sampling.
    """
    print("Systematic DA-like Boundary Evaluation Test")
    print("="*60)
    
    # Simple test parameters
    ra = np.array([0.1, 0.11, 0.12])  # radians
    dec = np.array([0.05, 0.06, 0.07])  # radians
    ra_sigma = 1e-6  # about 0.2 arcsec
    dec_sigma = 1e-6
    
    # Observer positions (simplified but realistic scale)
    pos_obs = np.array([
        [-1.47e8, -1.46e8, -1.45e8],
        [2.89e7, 2.91e7, 2.93e7], 
        [1.25e7, 1.26e7, 1.27e7]
    ])
    
    time_sec = np.array([0, 86400, 172800])  # 0, 1, 2 days in seconds
    mu = 1.32712440042e11  # km^3/s^2
    
    print(f"Test parameters:")
    print(f"  RA: {ra}")
    print(f"  DEC: {dec}")
    print(f"  Uncertainties: σ_RA={ra_sigma:.2e}, σ_DEC={dec_sigma:.2e}")
    
    # Method 1: Random interior sampling (standard Monte Carlo)
    print(f"\n1. STANDARD MONTE CARLO (Random Interior):")
    mc_results = test_random_interior(100, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu)
    
    # Method 2: Random boundary sampling  
    print(f"\n2. RANDOM BOUNDARY SAMPLING:")
    boundary_results = test_random_boundary(100, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu)
    
    # Method 3: Systematic DA-like boundary evaluation
    print(f"\n3. SYSTEMATIC DA-LIKE BOUNDARY EVALUATION:")
    da_results = test_systematic_da_like(pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu)
    
    # Create comprehensive plots
    create_comprehensive_plots(mc_results, boundary_results, da_results)
    
    # Analysis
    print(f"\n" + "="*60)
    print("DETAILED ANALYSIS:")
    print("="*60)
    
    methods = ['Random Interior (MC)', 'Random Boundary', 'Systematic DA-like']
    results = [mc_results, boundary_results, da_results]
    
    for i, (method, result) in enumerate(zip(methods, results)):
        if result['correlation'] is not None:
            corr = result['correlation']
            n_samples = len(result['ranges'])
            range_std = np.std(result['ranges'])
            rate_std = np.std(result['range_rates'])
            
            print(f"{method:25s}: Corr={corr:+.4f}, N={n_samples:3d}, "
                  f"σ_range={range_std:.2e}, σ_rate={rate_std:.2e}")
        else:
            print(f"{method:25s}: FAILED")
    
    # Test the refined hypothesis
    mc_corr = mc_results['correlation']
    da_corr = da_results['correlation']
    
    if mc_corr is not None and da_corr is not None:
        if mc_corr > 0.7 and da_corr < -0.7:
            print(f"\n✓ STRONG HYPOTHESIS SUPPORT:")
            print(f"  Interior sampling (MC): {mc_corr:+.4f} (positive)")
            print(f"  Systematic boundary (DA): {da_corr:+.4f} (negative)")
        elif mc_corr > 0.5 and da_corr < -0.5:
            print(f"\n✓ MODERATE HYPOTHESIS SUPPORT:")
            print(f"  Interior sampling shows positive correlation: {mc_corr:+.4f}")
            print(f"  Systematic boundary shows negative correlation: {da_corr:+.4f}")
        elif abs(mc_corr - da_corr) > 0.5:
            print(f"\n? WEAK HYPOTHESIS SUPPORT:")
            print(f"  Different correlation signs but not as strong as expected")
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED:")
            print(f"  Similar correlations between methods")


def test_random_interior(num_sim, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu):
    """Standard Monte Carlo with Gaussian interior sampling."""
    results = []
    rng = default_rng(42)
    
    for _ in range(num_sim):
        try:
            u = np.zeros((3, 3))
            for j in range(3):
                # Gaussian sampling (interior)
                d_ra = rng.normal(0, ra_sigma)
                d_dec = rng.normal(0, dec_sigma)
                d_ra = np.clip(d_ra, -3*ra_sigma, 3*ra_sigma)
                d_dec = np.clip(d_dec, -3*dec_sigma, 3*dec_sigma)
                
                u0 = np.array([np.cos(dec[j])*np.cos(ra[j]), 
                              np.cos(dec[j])*np.sin(ra[j]), 
                              np.sin(dec[j])])
                J = los_jacobian(ra[j], dec[j])
                u[:,j] = u0 + J @ np.array([d_ra, d_dec])
                u[:,j] = u[:,j] / np.linalg.norm(u[:,j])
            
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
            X_state = np.concatenate((r[:,1], v_2))
            results.append(X_state)
            
        except:
            continue
    
    return analyze_results(results, "Random Interior (MC)")


def test_random_boundary(num_sim, pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu):
    """Random boundary sampling."""
    results = []
    rng = default_rng(43)
    
    for _ in range(num_sim):
        try:
            u = np.zeros((3, 3))
            for j in range(3):
                # Random angle on boundary circle
                theta = rng.uniform(0, 2*np.pi)
                d_ra = 3*ra_sigma * np.cos(theta)
                d_dec = 3*dec_sigma * np.sin(theta)
                
                u0 = np.array([np.cos(dec[j])*np.cos(ra[j]), 
                              np.cos(dec[j])*np.sin(ra[j]), 
                              np.sin(dec[j])])
                J = los_jacobian(ra[j], dec[j])
                u[:,j] = u0 + J @ np.array([d_ra, d_dec])
                u[:,j] = u[:,j] / np.linalg.norm(u[:,j])
            
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
            X_state = np.concatenate((r[:,1], v_2))
            results.append(X_state)
            
        except:
            continue
    
    return analyze_results(results, "Random Boundary")


def test_systematic_da_like(pos_obs, ra, dec, time_sec, ra_sigma, dec_sigma, mu):
    """Systematic DA-like evaluation at domain corners."""
    results = []
    
    # Create systematic evaluation points (like DA polynomial evaluation)
    # Use fewer variables for simplicity but maintain systematic approach
    domain_combinations = [
        [-1, -1], [-1, 1], [1, -1], [1, 1]  # 4 corners of 2D domain
    ]
    
    # Evaluate at multiple systematic points
    for i in range(3):  # For each observation
        for da_point in domain_combinations:
            try:
                u = np.zeros((3, 3))
                
                # For systematic evaluation, apply the same perturbation to all observations
                # (this mimics how DA polynomials are evaluated across the domain)
                d_ra = da_point[0] * 3 * ra_sigma  
                d_dec = da_point[1] * 3 * dec_sigma
                
                for j in range(3):
                    u0 = np.array([np.cos(dec[j])*np.cos(ra[j]), 
                                  np.cos(dec[j])*np.sin(ra[j]), 
                                  np.sin(dec[j])])
                    J = los_jacobian(ra[j], dec[j])
                    u[:,j] = u0 + J @ np.array([d_ra, d_dec])
                    u[:,j] = u[:,j] / np.linalg.norm(u[:,j])
                
                r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
                X_state = np.concatenate((r[:,1], v_2))
                results.append(X_state)
                
            except:
                continue
    
    return analyze_results(results, "Systematic DA-like")


def analyze_results(results, method_name):
    """Common analysis for all methods."""
    if len(results) == 0:
        print(f"   {method_name}: FAILED - No successful IOD solutions")
        return {'correlation': None, 'ranges': None, 'range_rates': None}
    
    results = np.array(results)
    positions = results[:, :3]
    velocities = results[:, 3:]
    
    ranges = np.linalg.norm(positions, axis=1)
    range_rates = np.sum(positions * velocities, axis=1) / ranges
    
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    print(f"   {method_name}: Corr={correlation:+.4f}, N={len(results)} samples")
    print(f"      Range: [{ranges.min():.2e}, {ranges.max():.2e}] km")
    print(f"      Range-rate: [{range_rates.min():.2e}, {range_rates.max():.2e}] km/s")
    
    return {
        'correlation': correlation,
        'ranges': ranges,
        'range_rates': range_rates,
        'positions': positions,
        'velocities': velocities
    }


def create_comprehensive_plots(mc_results, boundary_results, da_results):
    """Create detailed comparison plots."""
    valid_results = [(mc_results, 'Random Interior (MC)', 'blue'),
                     (boundary_results, 'Random Boundary', 'red'),
                     (da_results, 'Systematic DA-like', 'green')]
    
    valid_results = [(r, n, c) for r, n, c in valid_results if r['correlation'] is not None]
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, len(valid_results), figsize=(5*len(valid_results), 10))
    if len(valid_results) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (result, name, color) in enumerate(valid_results):
        ranges = result['ranges']
        range_rates = result['range_rates']
        correlation = result['correlation']
        
        # Range vs Range-rate scatter plot
        axes[0, i].scatter(ranges, range_rates, alpha=0.7, s=20, c=color)
        axes[0, i].set_title(f'{name}\nCorrelation: {correlation:+.4f}')
        axes[0, i].set_xlabel('Range (km)')
        axes[0, i].set_ylabel('Range Rate (km/s)')
        axes[0, i].grid(True, alpha=0.3)
        
        # Distribution histograms
        axes[1, i].hist(ranges, bins=20, alpha=0.5, label='Range', color=color, density=True)
        axes[1, i].hist(range_rates, bins=20, alpha=0.5, label='Range Rate', color='orange', density=True)
        axes[1, i].set_title(f'{name} - Distributions')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Density')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('systematic_da_boundary_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nDetailed plots saved as 'systematic_da_boundary_test.png'")


if __name__ == "__main__":
    systematic_boundary_test()
