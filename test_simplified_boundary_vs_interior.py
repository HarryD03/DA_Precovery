"""
Simplified conceptual test: Direct demonstration of boundary vs interior sampling
correlation patterns using a controlled mathematical model.

This bypasses IOD numerical issues and directly demonstrates the correlation
inversion mechanism using a simplified orbital mechanics model.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

def simplified_orbital_model(ra_obs, dec_obs, observer_pos):
    """
    Simplified orbital mechanics model to convert observations to range/range-rate.
    This avoids the numerical complexities of full IOD while demonstrating the concept.
    
    Args:
        ra_obs: Right ascension observations [3x1]
        dec_obs: Declination observations [3x1] 
        observer_pos: Observer positions [3x3]
    
    Returns:
        range: Distance to object at epoch 2
        range_rate: Range rate at epoch 2
    """
    # Convert observations to line-of-sight unit vectors
    los_vectors = np.zeros((3, 3))
    for i in range(3):
        los_vectors[:,i] = np.array([
            np.cos(dec_obs[i]) * np.cos(ra_obs[i]),
            np.cos(dec_obs[i]) * np.sin(ra_obs[i]),
            np.sin(dec_obs[i])
        ])
    
    # Simplified geometric solution
    # Assume object is at distance ~ 1.5 AU from Sun
    # Use triangulation-like approach for simplicity
    
    # Average line-of-sight (central observation)
    los_central = los_vectors[:,1]  # Middle observation
    
    # Simplified range estimation based on observation geometry
    # This is a geometric approximation, not full IOD
    observer_central = observer_pos[:,1]
    
    # Estimate range based on line-of-sight and observer position
    # Use a simplified model: range ~ |observer| + perturbation based on angles
    observer_distance = np.linalg.norm(observer_central)
    
    # Geometric perturbation based on observation angles
    angle_perturbation = np.sum(los_central * observer_central) / observer_distance
    
    # Simplified range calculation (approximation)
    range_estimate = observer_distance * (1.2 + 0.1 * angle_perturbation)
    
    # Simplified range-rate based on observer motion and angle changes
    # Use finite differences in observations
    delta_ra = ra_obs[2] - ra_obs[0]
    delta_dec = dec_obs[2] - dec_obs[0]
    delta_t = 2 * 86400  # 2 days in seconds
    
    # Angular velocity
    angular_velocity = np.sqrt(delta_ra**2 + delta_dec**2) / delta_t
    
    # Simplified range-rate (geometric approximation)
    range_rate_estimate = -range_estimate * angular_velocity * 0.1  # Approximate scaling
    
    return range_estimate, range_rate_estimate


def test_boundary_vs_interior_simplified():
    """
    Test boundary vs interior sampling using simplified orbital model.
    """
    print("="*70)
    print("SIMPLIFIED BOUNDARY vs INTERIOR SAMPLING TEST")
    print("="*70)
    
    # Nominal observation parameters
    ra_nom = np.array([1.93, 1.94, 1.95])    # ~110 degrees
    dec_nom = np.array([0.326, 0.327, 0.328]) # ~18.7 degrees
    ra_sigma = 1e-5  # 2 arcsec
    dec_sigma = 1e-5
    
    # Observer positions (Earth heliocentric)
    observer_pos = np.array([
        [-1.47e8, -1.46e8, -1.45e8],
        [2.89e7, 2.91e7, 2.93e7], 
        [1.25e7, 1.26e7, 1.27e7]
    ])
    
    print(f"Test setup:")
    print(f"  Nominal RA: {ra_nom * 180/np.pi} degrees")
    print(f"  Nominal DEC: {dec_nom * 180/np.pi} degrees")
    print(f"  Uncertainties: {ra_sigma * 3600 * 180/np.pi:.2f} arcsec")
    
    # Method 1: Interior sampling (Monte Carlo style)
    print(f"\n1. INTERIOR SAMPLING (Monte Carlo style):")
    interior_results = interior_sampling_test(ra_nom, dec_nom, ra_sigma, dec_sigma, observer_pos)
    
    # Method 2: Boundary sampling (DA style)
    print(f"\n2. BOUNDARY SAMPLING (DA style):")
    boundary_results = boundary_sampling_test(ra_nom, dec_nom, ra_sigma, dec_sigma, observer_pos)
    
    # Analysis and comparison
    print(f"\n" + "="*70)
    print("CORRELATION ANALYSIS:")
    print("="*70)
    
    interior_analysis = analyze_simple_correlation(interior_results, "Interior Sampling (MC-like)")
    boundary_analysis = analyze_simple_correlation(boundary_results, "Boundary Sampling (DA-like)")
    
    # Create comparison plots
    create_simple_comparison_plots(interior_analysis, boundary_analysis)
    
    # Final comparison
    print(f"\n" + "="*70)
    print("BOUNDARY vs INTERIOR HYPOTHESIS TEST:")
    print("="*70)
    
    if interior_analysis and boundary_analysis:
        int_corr = interior_analysis['correlation']
        bound_corr = boundary_analysis['correlation']
        
        print(f"Interior Sampling Correlation:     {int_corr:+.4f}")
        print(f"Boundary Sampling Correlation:     {bound_corr:+.4f}")
        print(f"Correlation Difference:            {abs(int_corr - bound_corr):.4f}")
        
        # Test hypothesis
        if int_corr > 0.3 and bound_corr < -0.3:
            print(f"\n✓ STRONG HYPOTHESIS CONFIRMATION:")
            print(f"  Interior sampling → Positive correlation (+{int_corr:.3f})")
            print(f"  Boundary sampling → Negative correlation ({bound_corr:.3f})")
            print(f"  Mechanism: Different uncertainty space sampling strategies")
        elif int_corr > 0.1 and bound_corr < -0.1:
            print(f"\n✓ MODERATE HYPOTHESIS SUPPORT:")
            print(f"  Opposite correlation signs as predicted")
        elif abs(int_corr - bound_corr) > 0.4:
            print(f"\n✓ CORRELATION DIFFERENCE DETECTED:")
            print(f"  Significant difference between methods")
        else:
            print(f"\n? INCONCLUSIVE RESULTS:")
            print(f"  Similar correlations between methods")
    else:
        print("Analysis failed - insufficient data")


def interior_sampling_test(ra_nom, dec_nom, ra_sigma, dec_sigma, observer_pos, n_samples=100):
    """Interior sampling (Monte Carlo style) - samples from interior of uncertainty ellipse."""
    rng = default_rng(42)
    results = []
    
    for _ in range(n_samples):
        # Gaussian sampling (interior of uncertainty distribution)
        ra_pert = ra_nom + rng.normal(0, ra_sigma, 3)
        dec_pert = dec_nom + rng.normal(0, dec_sigma, 3)
        
        # 3-sigma clipping
        ra_pert = np.clip(ra_pert, ra_nom - 3*ra_sigma, ra_nom + 3*ra_sigma)
        dec_pert = np.clip(dec_pert, dec_nom - 3*dec_sigma, dec_nom + 3*dec_sigma)
        
        # Simplified orbital model
        try:
            range_est, range_rate_est = simplified_orbital_model(ra_pert, dec_pert, observer_pos)
            if not (np.isnan(range_est) or np.isnan(range_rate_est)):
                results.append([range_est, range_rate_est])
        except:
            continue
    
    print(f"  Interior sampling: {len(results)} valid results")
    return np.array(results)


def boundary_sampling_test(ra_nom, dec_nom, ra_sigma, dec_sigma, observer_pos):
    """Boundary sampling (DA style) - systematic evaluation at boundary points."""
    results = []
    
    # Systematic boundary evaluation (like DA domain evaluation)
    boundary_combinations = [
        [-1, -1], [-1, 1], [1, -1], [1, 1],    # Corners
        [-1, 0], [1, 0], [0, -1], [0, 1],      # Edge midpoints
        [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]  # Intermediate points
    ]
    
    for ra_mult, dec_mult in boundary_combinations:
        # DA-like scaling: 3σ * domain_variable
        ra_pert = ra_nom + ra_mult * 3 * ra_sigma
        dec_pert = dec_nom + dec_mult * 3 * dec_sigma
        
        try:
            range_est, range_rate_est = simplified_orbital_model(ra_pert, dec_pert, observer_pos)
            if not (np.isnan(range_est) or np.isnan(range_rate_est)):
                results.append([range_est, range_rate_est])
        except:
            continue
    
    print(f"  Boundary sampling: {len(results)} valid results")
    return np.array(results)


def analyze_simple_correlation(results, method_name):
    """Analyze correlation for simplified results."""
    if len(results) < 2:
        print(f"{method_name}: Insufficient data")
        return None
    
    ranges = results[:, 0]
    range_rates = results[:, 1]
    
    # Filter valid data
    valid_mask = ~(np.isnan(ranges) | np.isnan(range_rates) | 
                   np.isinf(ranges) | np.isinf(range_rates))
    
    if np.sum(valid_mask) < 2:
        print(f"{method_name}: No valid data after filtering")
        return None
    
    ranges = ranges[valid_mask]
    range_rates = range_rates[valid_mask]
    
    correlation = np.corrcoef(ranges, range_rates)[0,1]
    
    print(f"{method_name}:")
    print(f"  Valid samples: {len(ranges)}")
    print(f"  Range spread: {ranges.max() - ranges.min():.2e} km")
    print(f"  Range-rate spread: {range_rates.max() - range_rates.min():.2e} km/s")
    print(f"  Correlation: {correlation:+.4f}")
    
    return {
        'correlation': correlation,
        'ranges': ranges,
        'range_rates': range_rates,
        'method': method_name
    }


def create_simple_comparison_plots(interior_analysis, boundary_analysis):
    """Create comparison plots."""
    analyses = [interior_analysis, boundary_analysis]
    valid_analyses = [a for a in analyses if a is not None]
    
    if len(valid_analyses) == 0:
        print("No valid data for plotting")
        return
    
    fig, axes = plt.subplots(1, len(valid_analyses), figsize=(6*len(valid_analyses), 5))
    if len(valid_analyses) == 1:
        axes = [axes]
    
    colors = ['blue', 'red']
    
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
        
        # Add trend line
        z = np.polyfit(ranges, range_rates, 1)
        p = np.poly1d(z)
        axes[i].plot(ranges/1e6, p(ranges), "--", alpha=0.8, color=colors[i])
    
    plt.tight_layout()
    plt.savefig('simplified_boundary_vs_interior_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Simplified test plot saved as 'simplified_boundary_vs_interior_test.png'")


if __name__ == "__main__":
    test_boundary_vs_interior_simplified()
