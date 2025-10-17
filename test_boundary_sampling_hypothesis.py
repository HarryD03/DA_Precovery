"""
Test script to validate the boundary vs interior sampling hypothesis for range/range-rate correlations.

This script compares three methods:
1. Standard Monte Carlo (interior sampling)
2. Boundary Monte Carlo (boundary sampling)
3. DA method (polynomial boundary approximation)

Expected results based on hypothesis:
- Standard MC: Positive correlation (+0.99)
- Boundary MC: Negative correlation (-0.91) 
- DA method: Negative correlation (-0.91)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.typing import NDArray
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import utils.iod
import utils.time_reference as time_ref
from utils.Classical_IOD import monte_carlo_gauss_PWiod, sample_los, los_jacobian
from utils.lambert_izzo import lambert_izzo

def boundary_sample_los(RA, DEC, Sigma_ra_dec, method='corners'):
    """
    Sample line-of-sight vectors from the BOUNDARY of the uncertainty ellipsoid.
    
    Args:
        RA: Nominal right ascension (radians)
        DEC: Nominal declination (radians)  
        Sigma_ra_dec: 2x2 covariance matrix
        method: 'corners' for corner sampling, 'boundary' for boundary sampling
        
    Returns:
        u: Unit vector sampled from boundary
    """
    # Extract standard deviations
    sigma_ra = np.sqrt(Sigma_ra_dec[0,0])
    sigma_dec = np.sqrt(Sigma_ra_dec[1,1])
    
    if method == 'corners':
        # Sample from 8 corners of 3-sigma box (like DA method)
        corner_combinations = [
            [3*sigma_ra, 3*sigma_dec],
            [3*sigma_ra, -3*sigma_dec], 
            [-3*sigma_ra, 3*sigma_dec],
            [-3*sigma_ra, -3*sigma_dec],
            [3*sigma_ra, 0],
            [-3*sigma_ra, 0],
            [0, 3*sigma_dec], 
            [0, -3*sigma_dec]
        ]
        
        # Randomly select one corner
        rng = default_rng()
        corner_idx = rng.integers(0, len(corner_combinations))
        d_ra, d_dec = corner_combinations[corner_idx]
        
    elif method == 'boundary':
        # Sample uniformly from the boundary of the 3-sigma ellipse
        rng = default_rng()
        theta = rng.uniform(0, 2*np.pi)  # Angle around ellipse
        
        # Ellipse boundary in RA-DEC space (3-sigma)
        d_ra = 3*sigma_ra * np.cos(theta)
        d_dec = 3*sigma_dec * np.sin(theta)
    
    # Convert to unit vector using same transformation as standard MC
    u0 = np.array([np.cos(DEC)*np.cos(RA), np.cos(DEC)*np.sin(RA), np.sin(DEC)])
    J = los_jacobian(RA, DEC)
    
    u = u0 + J @ np.array([d_ra, d_dec])
    return u / np.linalg.norm(u)


def boundary_monte_carlo_iod(num_simulations: int, observer_position: NDArray, 
                           RA_nom, DEC_nom, time_sec: NDArray, RA_sigma, DEC_sigma, 
                           mu: float, prograde_bool_, method='corners'):
    """
    Boundary sampling Monte Carlo IOD - samples from uncertainty boundary instead of interior.
    
    This mimics the DA approach by sampling from the edges/corners of the uncertainty space.
    """
    results_DAIOD = np.zeros((num_simulations, 6))
    results_gauss = np.zeros((num_simulations, 6))
    
    if isinstance(RA_sigma, (int, float)):
        RA_sigma = np.ones(3) * RA_sigma
    if isinstance(DEC_sigma, (int, float)):
        DEC_sigma = np.ones(3) * DEC_sigma

    # Calculate covariance matrices
    cov = []
    for i in range(len(RA_nom)):
        cov_tmp = np.array([[(RA_sigma[i])**2, 0.0],
                           [0.0, (DEC_sigma[i])**2]])
        cov.append(cov_tmp)
        
    successful_sims = 0
    sim_idx = 0
    
    while successful_sims < num_simulations:
        # Generate boundary samples for line-of-sight vectors
        u = np.zeros((3,3))
        for j in range(len(RA_nom)): 
            u[:,j] = boundary_sample_los(RA_nom[j], DEC_nom[j], cov[j], method=method)

        try:
            # Same IOD process as standard Monte Carlo
            r, range_vec, range_mag, v_2 = utils.iod.Guass_8th_seed(observer_position, u, time_sec, mu)
            X0_Gauss = np.concatenate((r[:,1], v_2))
            
            results_gauss[successful_sims, :] = X0_Gauss

            # DAIOD step
            range_mag_L1_nom = utils.iod.DAIOD_1Scipy_invert(range_mag, u, time_sec, None, 
                                                      observer_position, mu, tol=1e-9, 
                                                      prograde_bool=prograde_bool_)
            
            # Convert refined magnitude to State Vector
            range_vec = range_mag_L1_nom * u
            r_vec = range_vec + observer_position
            dt1 = time_sec[1] - time_sec[0]
            dt2 = time_sec[2] - time_sec[1]

            velocities1 = lambert_izzo(r_vec[:,0], r_vec[:,1], dt1, mu, 0, prograde=prograde_bool_)
            v2 = velocities1[0][:,1]

            X_epoch = np.concatenate((r_vec[:,1], v2))
            results_DAIOD[successful_sims, :] = X_epoch
            
            successful_sims += 1
 
        except Exception as e:
            print(f"Simulation {sim_idx} failed: {e}")
            sim_idx += 1
            if sim_idx > num_simulations * 3:  # Prevent infinite loop
                print(f"Too many failures, stopping at {successful_sims} successful simulations")
                break
            continue
        
        sim_idx += 1

    return results_DAIOD[:successful_sims, :], results_gauss[:successful_sims, :]


def test_sampling_methods():
    """
    Test the boundary vs interior sampling hypothesis by comparing correlations.
    """
    print("Testing Boundary vs Interior Sampling Hypothesis")
    print("="*60)
    
    # Test parameters (using similar setup to your existing analysis)
    num_simulations = 1000
    mu = 1.32712440042e11  # km^3/s^2 (heliocentric)
    prograde_bool = True
    
    # Observer positions (example - you may need to adjust based on your data)
    observer_position = np.array([
        [-1.47e8, 2.89e7, 1.25e7],
        [-1.46e8, 2.91e7, 1.26e7], 
        [-1.45e8, 2.93e7, 1.27e7]
    ]).T
    
    # Nominal observations (example - adjust based on your Apophis data)
    RA_nom = np.array([0.1, 0.11, 0.12])  # radians
    DEC_nom = np.array([0.05, 0.06, 0.07])  # radians
    time_sec = np.array([0, 3600, 7200])  # seconds
    
    # Observation uncertainties
    RA_sigma = 1e-6  # radians (about 0.2 arcsec)
    DEC_sigma = 1e-6  # radians
    
    print(f"Running {num_simulations} simulations for each method...")
    
    # Method 1: Standard Monte Carlo (interior sampling)
    print("\n1. Standard Monte Carlo (Interior Sampling)...")
    try:
        mc_results, mc_gauss = monte_carlo_gauss_PWiod(
            num_simulations, observer_position, RA_nom, DEC_nom, 
            time_sec, RA_sigma, DEC_sigma, mu, prograde_bool
        )
        
        # Calculate range and range-rate
        mc_positions = mc_results[:, :3]
        mc_velocities = mc_results[:, 3:]
        mc_ranges = np.linalg.norm(mc_positions, axis=1)
        mc_range_rates = np.sum(mc_positions * mc_velocities, axis=1) / mc_ranges
        mc_correlation = np.corrcoef(mc_ranges, mc_range_rates)[0,1]
        
        print(f"   Standard MC Correlation: {mc_correlation:.4f}")
        
    except Exception as e:
        print(f"   Standard MC failed: {e}")
        mc_correlation = np.nan
    
    # Method 2: Boundary Monte Carlo (corner sampling)
    print("\n2. Boundary Monte Carlo - Corner Sampling...")
    try:
        boundary_results, boundary_gauss = boundary_monte_carlo_iod(
            num_simulations, observer_position, RA_nom, DEC_nom,
            time_sec, RA_sigma, DEC_sigma, mu, prograde_bool, method='corners'
        )
        
        # Calculate range and range-rate
        boundary_positions = boundary_results[:, :3]
        boundary_velocities = boundary_results[:, 3:]
        boundary_ranges = np.linalg.norm(boundary_positions, axis=1)
        boundary_range_rates = np.sum(boundary_positions * boundary_velocities, axis=1) / boundary_ranges
        boundary_correlation = np.corrcoef(boundary_ranges, boundary_range_rates)[0,1]
        
        print(f"   Boundary MC Correlation: {boundary_correlation:.4f}")
        
    except Exception as e:
        print(f"   Boundary MC failed: {e}")
        boundary_correlation = np.nan
    
    # Method 3: Boundary Monte Carlo (ellipse boundary sampling)
    print("\n3. Boundary Monte Carlo - Ellipse Boundary Sampling...")
    try:
        ellipse_results, ellipse_gauss = boundary_monte_carlo_iod(
            num_simulations, observer_position, RA_nom, DEC_nom,
            time_sec, RA_sigma, DEC_sigma, mu, prograde_bool, method='boundary'
        )
        
        # Calculate range and range-rate
        ellipse_positions = ellipse_results[:, :3]
        ellipse_velocities = ellipse_results[:, 3:]
        ellipse_ranges = np.linalg.norm(ellipse_positions, axis=1)
        ellipse_range_rates = np.sum(ellipse_positions * ellipse_velocities, axis=1) / ellipse_ranges
        ellipse_correlation = np.corrcoef(ellipse_ranges, ellipse_range_rates)[0,1]
        
        print(f"   Ellipse Boundary MC Correlation: {ellipse_correlation:.4f}")
        
    except Exception as e:
        print(f"   Ellipse Boundary MC failed: {e}")
        ellipse_correlation = np.nan
    
    # Summary
    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULTS:")
    print("="*60)
    print(f"Standard MC (Interior):    {mc_correlation:+.4f}")
    print(f"Boundary MC (Corners):     {boundary_correlation:+.4f}")
    print(f"Boundary MC (Ellipse):     {ellipse_correlation:+.4f}")
    print()
    
    # Hypothesis validation
    if not np.isnan(mc_correlation) and not np.isnan(boundary_correlation):
        if mc_correlation > 0.5 and boundary_correlation < -0.5:
            print("✓ HYPOTHESIS CONFIRMED:")
            print("  Interior sampling → Positive correlation")
            print("  Boundary sampling → Negative correlation")
        elif abs(mc_correlation - boundary_correlation) < 0.1:
            print("✗ HYPOTHESIS REJECTED:")
            print("  No significant difference between sampling methods")
        else:
            print("? INCONCLUSIVE:")
            print("  Results don't clearly support or reject hypothesis")
    else:
        print("! INCOMPLETE:")
        print("  Some methods failed to complete")
    
    # Create plots if successful
    if not np.isnan(mc_correlation) and not np.isnan(boundary_correlation):
        create_comparison_plots(
            mc_ranges, mc_range_rates, mc_correlation,
            boundary_ranges, boundary_range_rates, boundary_correlation,
            ellipse_ranges if not np.isnan(ellipse_correlation) else None,
            ellipse_range_rates if not np.isnan(ellipse_correlation) else None,
            ellipse_correlation if not np.isnan(ellipse_correlation) else None
        )


def create_comparison_plots(mc_ranges, mc_range_rates, mc_corr,
                          boundary_ranges, boundary_range_rates, boundary_corr,
                          ellipse_ranges=None, ellipse_range_rates=None, ellipse_corr=None):
    """
    Create comparison plots for the different sampling methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Standard Monte Carlo
    axes[0].scatter(mc_ranges, mc_range_rates, alpha=0.6, s=10, c='blue')
    axes[0].set_title(f'Standard MC (Interior)\nCorrelation: {mc_corr:+.4f}')
    axes[0].set_xlabel('Range (km)')
    axes[0].set_ylabel('Range Rate (km/s)')
    axes[0].grid(True, alpha=0.3)
    
    # Boundary Monte Carlo (corners)
    axes[1].scatter(boundary_ranges, boundary_range_rates, alpha=0.6, s=10, c='red')
    axes[1].set_title(f'Boundary MC (Corners)\nCorrelation: {boundary_corr:+.4f}')
    axes[1].set_xlabel('Range (km)')
    axes[1].set_ylabel('Range Rate (km/s)')
    axes[1].grid(True, alpha=0.3)
    
    # Ellipse boundary or comparison
    if ellipse_ranges is not None:
        axes[2].scatter(ellipse_ranges, ellipse_range_rates, alpha=0.6, s=10, c='green')
        axes[2].set_title(f'Boundary MC (Ellipse)\nCorrelation: {ellipse_corr:+.4f}')
    else:
        # Overlay comparison
        axes[2].scatter(mc_ranges, mc_range_rates, alpha=0.4, s=8, c='blue', label=f'Interior ({mc_corr:+.3f})')
        axes[2].scatter(boundary_ranges, boundary_range_rates, alpha=0.4, s=8, c='red', label=f'Boundary ({boundary_corr:+.3f})')
        axes[2].set_title('Comparison: Interior vs Boundary')
        axes[2].legend()
    
    axes[2].set_xlabel('Range (km)')
    axes[2].set_ylabel('Range Rate (km/s)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boundary_sampling_hypothesis_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'boundary_sampling_hypothesis_test.png'")


if __name__ == "__main__":
    test_sampling_methods()
