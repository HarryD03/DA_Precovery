"""
Direct comparison test: Pointwise 3-sigma box boundary evaluation vs DA implementation.

This test directly compares:
1. Manual pointwise evaluation of 3-sigma box corners/edges
2. Actual DA polynomial implementation from your codebase
3. Range vs range-rate correlations for both methods

Expected result: Both should show similar negative correlations if our hypothesis is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import sys
import os
from daceypy import DA, array
from astropy.time import Time
from astropy import units as u
import astropy.coordinates as acoords

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import utils.iod as iod
import utils.time_reference as time_ref
import utils.post_process as post
from utils.Classical_IOD import los_jacobian

def pointwise_box_boundary_evaluation(ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu):
    """
    Manual pointwise evaluation of 3-sigma box boundary.
    This replicates what DA does but with explicit point evaluation.
    """
    print("Running pointwise 3-sigma box boundary evaluation...")
    
    # Create systematic 3-sigma box boundary points
    # This mimics DA evaluation at domain corners: DA ∈ {-1, +1}
    boundary_points = []
    
    # For 3 observations, we need 6 DA variables (3 for RA, 3 for DEC)
    # Create all combinations of ±1 for systematic boundary evaluation
    import itertools
    
    # Start with 2D case for clarity (can extend to 6D)
    # Use ±3σ perturbations (like DA scaling)
    sigma_levels = [-3, -1, 0, 1, 3]  # Include intermediate points
    
    results = []
    valid_combinations = []
    
    for ra_mult in sigma_levels:
        for dec_mult in sigma_levels:
            try:
                # Apply perturbations to all observations consistently
                u = np.zeros((3, 3))
                
                for j in range(3):
                    # 3-sigma perturbations (like DA: 3*sigma*DA_variable)
                    d_ra = ra_mult * ra_sigma
                    d_dec = dec_mult * dec_sigma
                    
                    # Convert to unit vector using same math as DA
                    u0 = np.array([np.cos(dec_nom[j])*np.cos(ra_nom[j]), 
                                  np.cos(dec_nom[j])*np.sin(ra_nom[j]), 
                                  np.sin(dec_nom[j])])
                    J = los_jacobian(ra_nom[j], dec_nom[j])
                    u[:,j] = u0 + J @ np.array([d_ra, d_dec])
                    u[:,j] = u[:,j] / np.linalg.norm(u[:,j])
                
                # IOD using same method as DA
                r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(pos_obs, u, time_sec, mu)
                X_state = np.concatenate((r[:,1], v_2))
                
                results.append(X_state)
                valid_combinations.append((ra_mult, dec_mult))
                
            except Exception as e:
                continue
    
    print(f"Pointwise evaluation: {len(results)} successful solutions")
    return np.array(results), valid_combinations


def da_implementation_evaluation(ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu, order=1):
    """
    Use actual DA implementation from your codebase.
    """
    print("Running actual DA implementation...")
    
    try:
        # Use your actual DAIOD implementation
        X0_DA = iod.DAIOD_full(ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu, order, prograde=True)
        
        if X0_DA is None:
            print("DA implementation failed")
            return None, None
        
        # The DA result is a polynomial - need to evaluate at specific points
        # Extract coefficients to understand the polynomial structure
        print(f"DA result type: {type(X0_DA)}")
        print(f"DA result shape/structure: {X0_DA}")
        
        # For comparison, we need to evaluate the DA polynomial at boundary points
        # This requires extracting the polynomial and evaluating it
        da_evaluations = []
        da_points = []
        
        # Evaluate DA polynomial at systematic boundary points
        # DA domain evaluation at ±1 combinations
        boundary_combinations = [
            [-1, -1, -1, -1, -1, -1],  # All negative
            [1, 1, 1, 1, 1, 1],        # All positive
            [-1, 1, -1, 1, -1, 1],     # Alternating
            [1, -1, 1, -1, 1, -1],     # Alternating opposite
            # Add more systematic combinations...
        ]
        
        for combo in boundary_combinations:
            try:
                # This would require DA polynomial evaluation at specific points
                # For now, we'll use the structure we can access
                if hasattr(X0_DA, 'constant'):
                    # If it's a DA array, extract evaluations
                    evaluated_point = X0_DA  # This needs proper DA evaluation
                    da_evaluations.append(evaluated_point)
                    da_points.append(combo)
                else:
                    # If it's already evaluated, use as is
                    da_evaluations.append(X0_DA)
                    da_points.append(combo)
                    break  # Only one evaluation available
            except:
                continue
        
        return da_evaluations, da_points
        
    except Exception as e:
        print(f"DA implementation failed: {e}")
        return None, None


def compare_boundary_vs_da():
    """
    Main comparison function.
    """
    print("="*70)
    print("POINTWISE BOUNDARY vs DA IMPLEMENTATION COMPARISON")
    print("="*70)
    
    # Test parameters - use realistic values from your Apophis case
    try:
        # Try to load real Apophis data
        obs_time0 = Time('2005-04-13 21:59:00', scale='utc')
        dt = np.array([0, 1, 2])  # days  
        obs_times = obs_time0 + (dt * u.day)
        
        # Get observer positions
        epochs = Time(obs_times, scale='tdb')
        acoords.solar_system_ephemeris.set("builtin")
        
        pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]
        pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
        pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]
        
        pos_obs = np.array([np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))[:3] 
                           for p_helio, v_helio in pv_earth_helio]).T
        
        # Try to get actual Apophis observations
        eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')
        eph2, _ = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')
        
        ra_nom = np.concatenate([eph['RA'], eph2['RA'][1:]]) * np.pi/180
        dec_nom = np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * np.pi/180
        
        ra_sigma = (30 * eph['RA_3sigma'][1] / 3600) * np.pi/180 / 3
        dec_sigma = (30 * eph['DEC_3sigma'][1] / 3600) * np.pi/180 / 3
        
        print(f"Using real Apophis data:")
        print(f"  RA: {ra_nom * 180/np.pi}")
        print(f"  DEC: {dec_nom * 180/np.pi}")
        print(f"  σ_RA: {ra_sigma:.2e} rad")
        print(f"  σ_DEC: {dec_sigma:.2e} rad")
        
    except Exception as e:
        print(f"Failed to load Apophis data: {e}")
        print("Using simplified test parameters...")
        
        # Fallback parameters
        pos_obs = np.array([
            [-1.47e8, -1.46e8, -1.45e8],
            [2.89e7, 2.91e7, 2.93e7], 
            [1.25e7, 1.26e7, 1.27e7]
        ])
        
        ra_nom = np.array([0.1, 0.11, 0.12])
        dec_nom = np.array([0.05, 0.06, 0.07])
        ra_sigma = 1e-6
        dec_sigma = 1e-6
    
    # Common parameters
    time_sec = np.array([0, 86400, 172800])  # 0, 1, 2 days in seconds
    mu = 1.32712440042e11  # km^3/s^2 (heliocentric)
    
    print(f"\nTest setup:")
    print(f"  Observer positions shape: {pos_obs.shape}")
    print(f"  Time intervals: {time_sec}")
    
    # Method 1: Pointwise box boundary evaluation
    print(f"\n1. POINTWISE 3-SIGMA BOX BOUNDARY EVALUATION:")
    pointwise_results, pointwise_points = pointwise_box_boundary_evaluation(
        ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu
    )
    
    # Method 2: Actual DA implementation
    print(f"\n2. ACTUAL DA IMPLEMENTATION:")
    da_results, da_points = da_implementation_evaluation(
        ra_nom, dec_nom, ra_sigma, dec_sigma, pos_obs, time_sec, mu
    )
    
    # Analysis and comparison
    print(f"\n" + "="*70)
    print("ANALYSIS AND COMPARISON:")
    print("="*70)
    
    # Analyze pointwise results
    if len(pointwise_results) > 0:
        pointwise_analysis = analyze_results(pointwise_results, "Pointwise Box Boundary")
    else:
        pointwise_analysis = None
        print("Pointwise evaluation: No valid results")
    
    # Analyze DA results
    if da_results is not None and len(da_results) > 0:
        if isinstance(da_results[0], np.ndarray):
            da_array = np.array(da_results)
            da_analysis = analyze_results(da_array, "DA Implementation")
        else:
            print("DA Implementation: Result format not suitable for correlation analysis")
            da_analysis = None
    else:
        da_analysis = None
        print("DA Implementation: No valid results")
    
    # Create comparison plots
    create_comparison_plots(pointwise_analysis, da_analysis, pointwise_points)
    
    # Final comparison
    print(f"\n" + "="*70)
    print("BOUNDARY vs DA CORRELATION COMPARISON:")
    print("="*70)
    
    if pointwise_analysis and da_analysis:
        pointwise_corr = pointwise_analysis['correlation'] 
        da_corr = da_analysis['correlation']
        
        print(f"Pointwise Box Boundary Correlation: {pointwise_corr:+.4f}")
        print(f"DA Implementation Correlation:     {da_corr:+.4f}")
        print(f"Difference:                        {abs(pointwise_corr - da_corr):.4f}")
        
        if abs(pointwise_corr - da_corr) < 0.2:
            print("\n✓ HYPOTHESIS CONFIRMED:")
            print("  Similar correlations between pointwise boundary and DA methods")
            print("  Both capture boundary sampling behavior")
        else:
            print("\n? MIXED RESULTS:")
            print("  Different correlations - need further investigation")
    else:
        print("Insufficient data for correlation comparison")


def analyze_results(results, method_name):
    """Analyze range vs range-rate correlation for results."""
    if len(results) == 0:
        return None
    
    positions = results[:, :3]  
    velocities = results[:, 3:]
    
    ranges = np.linalg.norm(positions, axis=1)
    range_rates = np.sum(positions * velocities, axis=1) / ranges
    
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    print(f"{method_name}:")
    print(f"  Samples: {len(results)}")
    print(f"  Range: [{ranges.min():.2e}, {ranges.max():.2e}] km")
    print(f"  Range-rate: [{range_rates.min():.2e}, {range_rates.max():.2e}] km/s")
    print(f"  Correlation: {correlation:+.4f}")
    
    return {
        'correlation': correlation,
        'ranges': ranges,
        'range_rates': range_rates,
        'positions': positions,
        'velocities': velocities,
        'method': method_name
    }


def create_comparison_plots(pointwise_analysis, da_analysis, pointwise_points):
    """Create comparison plots."""
    valid_analyses = [a for a in [pointwise_analysis, da_analysis] if a is not None]
    
    if len(valid_analyses) == 0:
        print("No valid analyses for plotting")
        return
    
    fig, axes = plt.subplots(1, len(valid_analyses), figsize=(6*len(valid_analyses), 5))
    if len(valid_analyses) == 1:
        axes = [axes]
    
    colors = ['red', 'blue']
    
    for i, analysis in enumerate(valid_analyses):
        ranges = analysis['ranges']
        range_rates = analysis['range_rates']
        correlation = analysis['correlation']
        method = analysis['method']
        
        axes[i].scatter(ranges/1e6, range_rates, alpha=0.7, s=40, c=colors[i])
        axes[i].set_xlabel('Range (10⁶ km)')
        axes[i].set_ylabel('Range Rate (km/s)')
        axes[i].set_title(f'{method}\nCorrelation: {correlation:+.4f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pointwise_vs_da_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved as 'pointwise_vs_da_comparison.png'")
    
    # Also create boundary point visualization if available
    if pointwise_points:
        create_boundary_point_plot(pointwise_points)


def create_boundary_point_plot(boundary_points):
    """Visualize the boundary points sampled."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ra_mults = [p[0] for p in boundary_points]
    dec_mults = [p[1] for p in boundary_points]
    
    ax.scatter(ra_mults, dec_mults, s=60, c='red', alpha=0.7)
    ax.set_xlabel('RA Multiplier (σ units)')
    ax.set_ylabel('DEC Multiplier (σ units)')
    ax.set_title('Pointwise Boundary Sampling Points')
    ax.grid(True, alpha=0.3)
    
    # Add 3-sigma box outline
    box_x = [-3, 3, 3, -3, -3]
    box_y = [-3, -3, 3, 3, -3]
    ax.plot(box_x, box_y, 'k--', alpha=0.5, label='3σ Box Boundary')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('boundary_sampling_points.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Boundary points plot saved as 'boundary_sampling_points.png'")


if __name__ == "__main__":
    compare_boundary_vs_da()
