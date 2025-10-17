import os
import sys
from pathlib import Path
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from daceypy import DA, ADS, array

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def load_and_examine_raw_data():
    """Load and examine the raw data before coordinate transformations"""
    
    script_dir = Path(__file__).parent
    simulation_data_dir = script_dir / "Simulation_data"
    query_data_dir = script_dir / "Query_data"
    
    # Load DA parameters first
    da_params_pattern = str(simulation_data_dir / "arc_*_da_params_*.pkl")
    da_params_files = glob.glob(da_params_pattern)
    
    if da_params_files:
        da_params_file = max(da_params_files, key=os.path.getctime)
        with open(da_params_file, 'rb') as f:
            da_params = pickle.load(f)
        DA.init(da_params['da_order'], da_params['da_nvars'])
        print(f"DA initialized: order={da_params['da_order']}, nvars={da_params['da_nvars']}")
    
    # Load query results
    result_files = glob.glob(str(query_data_dir / "arc_*_query_results_*.pkl"))
    if not result_files:
        print("No query result files found!")
        return None
    
    result_file = max(result_files, key=os.path.getctime)
    with open(result_file, 'rb') as f:
        result_data = pickle.load(f)
    
    return result_data

def examine_coordinate_transformations(result_data):
    """Examine the coordinate transformation chain for different methods"""
    
    print("\n=== Examining Coordinate Transformation Chain ===")
    
    time_idx = 0
    
    # Check if we have access to intermediate coordinate frames
    print("\nAvailable data keys:")
    for key in sorted(result_data.keys()):
        if isinstance(result_data[key], (np.ndarray, list, dict)):
            if isinstance(result_data[key], np.ndarray):
                print(f"  {key}: {type(result_data[key])}, shape: {result_data[key].shape}")
            elif isinstance(result_data[key], list):
                print(f"  {key}: {type(result_data[key])}, length: {len(result_data[key])}")
            elif isinstance(result_data[key], dict):
                print(f"  {key}: {type(result_data[key])}, keys: {list(result_data[key].keys())}")
        else:
            print(f"  {key}: {type(result_data[key])} = {result_data[key]}")
    
    # Look for cartesian state data (before obs conversion)
    cartesian_keys = [k for k in result_data.keys() if 'cartesian' in k.lower() or 'helio' in k.lower() or 'eci' in k.lower()]
    print(f"\nCartesian/Heliocentric state keys: {cartesian_keys}")
    
    # Look for observational data (after obs conversion)
    obs_keys = [k for k in result_data.keys() if 'obs' in k.lower() or 'geocentric' in k.lower()]
    print(f"Observational state keys: {obs_keys}")

def examine_uncertainty_propagation_methods(result_data):
    """Compare how different methods propagate uncertainties"""
    
    print("\n=== Examining Uncertainty Propagation Methods ===")
    
    time_idx = 0
    
    # 1. Examine DA/ADS perimeter evaluation
    print("\n1. DA/ADS Perimeter Evaluation:")
    
    if 'DAIOD_ADS_perimeter' in result_data:
        perimeter_data = result_data['DAIOD_ADS_perimeter']
        if 'final_map' in perimeter_data:
            manifold = perimeter_data['final_map'][time_idx]
            print(f"   Manifold shape: {manifold.shape}")
            
            # Examine the state components
            print("   State components (first 5 points):")
            for i in range(min(5, manifold.shape[0])):
                state = manifold[i, :, 0]  # First subdomain
                print(f"   Point {i}: RA={state[0]:.6f}, DEC={state[1]:.6f}, Range={state[2]/1e6:.3f}Mm, "
                      f"RA_dot={state[3]:.8f}, DEC_dot={state[4]:.8f}, Range_dot={state[5]:.6f}")
    
    # 2. Examine Monte Carlo observational data
    print("\n2. Monte Carlo Observational Data:")
    
    for method_name, data_key in [('DAIOD_MC', 'X_DAIOD_MC_geocentric_obs'), 
                                  ('GAUSS_MC', 'X_GAUSS_MC_geocentric_obs')]:
        if data_key in result_data:
            obs_data = result_data[data_key]
            print(f"   {method_name} shape: {obs_data.shape}")
            
            # Sample a few points
            sample_indices = np.random.choice(obs_data.shape[0], min(5, obs_data.shape[0]), replace=False)
            print(f"   Sample states at time {time_idx} (first 5 samples):")
            
            for i, idx in enumerate(sample_indices):
                state = obs_data[idx, :, time_idx]
                print(f"   Sample {i}: RA={state[0]:.6f}, DEC={state[1]:.6f}, Range={state[2]/1e6:.3f}Mm, "
                      f"RA_dot={state[3]:.8f}, DEC_dot={state[4]:.8f}, Range_dot={state[5]:.6f}")

def analyze_state_correlations(result_data):
    """Analyze correlations between state components"""
    
    print("\n=== Analyzing State Component Correlations ===")
    
    time_idx = 0
    methods_data = {}
    
    # Extract data for each method
    # DA/ADS method
    if 'DAIOD_ADS_perimeter' in result_data:
        perimeter_data = result_data['DAIOD_ADS_perimeter']
        if 'final_map' in perimeter_data:
            manifold = perimeter_data['final_map'][time_idx]
            # Extract all state vectors
            states = manifold[:, :, 0]  # All points, all states, first subdomain
            methods_data['DAIOD_DA'] = states
    
    # Monte Carlo methods
    for method_name, data_key in [('DAIOD_MC', 'X_DAIOD_MC_geocentric_obs'), 
                                  ('GAUSS_MC', 'X_GAUSS_MC_geocentric_obs')]:
        if data_key in result_data:
            obs_data = result_data[data_key]
            states = obs_data[:, :, time_idx]  # All samples, all states, at time_idx
            methods_data[method_name] = states
    
    # Compute and compare correlations
    for method_name, states in methods_data.items():
        print(f"\n{method_name} Correlations:")
        
        # Focus on Range (column 2) and Range-rate (column 5)
        ranges = states[:, 2]
        range_rates = states[:, 5]
        
        # Remove invalid values
        valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
        ranges_valid = ranges[valid_mask]
        range_rates_valid = range_rates[valid_mask]
        
        if len(ranges_valid) > 1:
            correlation = np.corrcoef(ranges_valid, range_rates_valid)[0, 1]
            print(f"   Range vs Range-rate correlation: {correlation:.4f}")
            print(f"   Range std: {np.std(ranges_valid)/1e6:.3f} Mm")
            print(f"   Range-rate std: {np.std(range_rates_valid):.6f} km/s")
            
            # Also check correlations with angular components
            ra = states[:, 0][valid_mask]
            dec = states[:, 1][valid_mask]
            
            if len(ra) > 1:
                ra_range_corr = np.corrcoef(ra, ranges_valid)[0, 1]
                dec_range_corr = np.corrcoef(dec, ranges_valid)[0, 1]
                ra_rr_corr = np.corrcoef(ra, range_rates_valid)[0, 1]
                dec_rr_corr = np.corrcoef(dec, range_rates_valid)[0, 1]
                
                print(f"   RA vs Range correlation: {ra_range_corr:.4f}")
                print(f"   DEC vs Range correlation: {dec_range_corr:.4f}")
                print(f"   RA vs Range-rate correlation: {ra_rr_corr:.4f}")
                print(f"   DEC vs Range-rate correlation: {dec_rr_corr:.4f}")

def create_detailed_scatter_plots(result_data):
    """Create detailed scatter plots to visualize the differences"""
    
    time_idx = 0
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods_data = {}
    colors = {'DAIOD_DA': 'green', 'DAIOD_MC': 'orange', 'GAUSS_MC': 'purple'}
    
    # Extract data for each method
    if 'DAIOD_ADS_perimeter' in result_data:
        perimeter_data = result_data['DAIOD_ADS_perimeter']
        if 'final_map' in perimeter_data:
            manifold = perimeter_data['final_map'][time_idx]
            states = manifold[:, :, 0]
            methods_data['DAIOD_DA'] = states
    
    for method_name, data_key in [('DAIOD_MC', 'X_DAIOD_MC_geocentric_obs'), 
                                  ('GAUSS_MC', 'X_GAUSS_MC_geocentric_obs')]:
        if data_key in result_data:
            obs_data = result_data[data_key]
            states = obs_data[:, :, time_idx]
            methods_data[method_name] = states
    
    # Plot 1: Range vs Range-rate (main plot)
    ax = axes[0, 0]
    for method, states in methods_data.items():
        ranges = states[:, 2] / 1e6  # Convert to Mm
        range_rates = states[:, 5]
        valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
        ax.scatter(ranges[valid_mask], range_rates[valid_mask], 
                  c=colors[method], alpha=0.6, s=2, label=method)
    ax.set_xlabel('Range [Mm]')
    ax.set_ylabel('Range Rate [km/s]')
    ax.set_title('Range vs Range-rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RA vs Range
    ax = axes[0, 1]
    for method, states in methods_data.items():
        ra = states[:, 0]
        ranges = states[:, 2] / 1e6
        valid_mask = np.isfinite(ra) & np.isfinite(ranges)
        ax.scatter(ra[valid_mask], ranges[valid_mask], 
                  c=colors[method], alpha=0.6, s=2, label=method)
    ax.set_xlabel('RA [rad]')
    ax.set_ylabel('Range [Mm]')
    ax.set_title('RA vs Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: DEC vs Range-rate
    ax = axes[0, 2]
    for method, states in methods_data.items():
        dec = states[:, 1]
        range_rates = states[:, 5]
        valid_mask = np.isfinite(dec) & np.isfinite(range_rates)
        ax.scatter(dec[valid_mask], range_rates[valid_mask], 
                  c=colors[method], alpha=0.6, s=2, label=method)
    ax.set_xlabel('DEC [rad]')
    ax.set_ylabel('Range Rate [km/s]')
    ax.set_title('DEC vs Range-rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Range distribution
    ax = axes[1, 0]
    for method, states in methods_data.items():
        ranges = states[:, 2] / 1e6
        valid_mask = np.isfinite(ranges)
        ax.hist(ranges[valid_mask], bins=50, alpha=0.6, label=method, color=colors[method])
    ax.set_xlabel('Range [Mm]')
    ax.set_ylabel('Frequency')
    ax.set_title('Range Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Range-rate distribution
    ax = axes[1, 1]
    for method, states in methods_data.items():
        range_rates = states[:, 5]
        valid_mask = np.isfinite(range_rates)
        ax.hist(range_rates[valid_mask], bins=50, alpha=0.6, label=method, color=colors[method])
    ax.set_xlabel('Range Rate [km/s]')
    ax.set_ylabel('Frequency')
    ax.set_title('Range-rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Angular distribution (RA vs DEC)
    ax = axes[1, 2]
    for method, states in methods_data.items():
        ra = states[:, 0]
        dec = states[:, 1]
        valid_mask = np.isfinite(ra) & np.isfinite(dec)
        ax.scatter(ra[valid_mask], dec[valid_mask], 
                  c=colors[method], alpha=0.6, s=2, label=method)
    ax.set_xlabel('RA [rad]')
    ax.set_ylabel('DEC [rad]')
    ax.set_title('Angular Distribution (RA vs DEC)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main diagnostic function"""
    
    print("=== Detailed Range/Range-rate Diagnostic ===")
    
    # Load data
    result_data = load_and_examine_raw_data()
    if result_data is None:
        return
    
    # Examine coordinate transformations
    examine_coordinate_transformations(result_data)
    
    # Examine uncertainty propagation methods
    examine_uncertainty_propagation_methods(result_data)
    
    # Analyze correlations
    analyze_state_correlations(result_data)
    
    # Create detailed plots
    create_detailed_scatter_plots(result_data)
    
    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    main()
