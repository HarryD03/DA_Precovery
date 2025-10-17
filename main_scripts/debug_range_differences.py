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

def load_da_params_and_data():
    """Load DA parameters and simulation data to examine range differences"""
    
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
    
    print(f"Loaded data from: {Path(result_file).name}")
    return result_data

def analyze_range_data_structures(result_data):
    """Analyze the range data structures for different methods"""
    
    print("\n=== Analyzing Range Data Structures ===")
    
    # For simplicity, assume new format (query_results separate)
    if 'query_results' in result_data:
        arc_data = result_data
        time_idx = 0  # Look at first time step
    else:
        arc_data = result_data
        time_idx = 0
    
    methods_info = {}
    
    # 1. ADS Methods (DAIOD_ADS_ADS and DAIOD_ADS)
    print("\n1. ADS Methods Analysis:")
    
    if 'DAIOD_ADS_perimeter' in arc_data:
        perimeter_data = arc_data['DAIOD_ADS_perimeter']
        print(f"   DAIOD_ADS_perimeter structure: {type(perimeter_data)}")
        if isinstance(perimeter_data, dict) and 'final_map' in perimeter_data:
            final_map = perimeter_data['final_map']
            print(f"   final_map length: {len(final_map)}")
            if len(final_map) > time_idx:
                manifold = final_map[time_idx]
                print(f"   Manifold shape at time {time_idx}: {manifold.shape}")
                print(f"   Manifold type: {type(manifold)}")
                
                # Extract ranges and range rates
                try:
                    ranges = manifold[:, 2, :].flatten()  # Range component (3rd state)
                    range_rates = manifold[:, 5, :].flatten()  # Range-rate component (6th state)
                    valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
                    ranges_valid = ranges[valid_mask]
                    range_rates_valid = range_rates[valid_mask]
                    
                    methods_info['DAIOD_ADS_ADS'] = {
                        'type': 'ADS_perimeter',
                        'n_points': len(ranges_valid),
                        'range_mean': np.mean(ranges_valid),
                        'range_std': np.std(ranges_valid),
                        'range_min': np.min(ranges_valid),
                        'range_max': np.max(ranges_valid),
                        'rangerate_mean': np.mean(range_rates_valid),
                        'rangerate_std': np.std(range_rates_valid),
                        'rangerate_min': np.min(range_rates_valid),
                        'rangerate_max': np.max(range_rates_valid),
                    }
                    
                    print(f"   ADS Ranges: {len(ranges_valid)} points")
                    print(f"   Range: {np.min(ranges_valid)/1e6:.2f} - {np.max(ranges_valid)/1e6:.2f} Mm")
                    print(f"   Range-rate: {np.min(range_rates_valid):.4f} - {np.max(range_rates_valid):.4f} km/s")
                    
                except Exception as e:
                    print(f"   Error extracting ADS data: {e}")
    
    # 2. DA Method (DAIOD_DA)
    print("\n2. DA Method Analysis:")
    
    if 'DAIOD_DA_perimeter' in arc_data:
        da_perimeter = arc_data['DAIOD_DA_perimeter']
        print(f"   DAIOD_DA_perimeter structure: {type(da_perimeter)}, length: {len(da_perimeter)}")
        if len(da_perimeter) > time_idx:
            da_points = da_perimeter[time_idx]
            print(f"   DA points shape at time {time_idx}: {da_points.shape}")
            print(f"   DA points type: {type(da_points)}")
            
            try:
                ranges = da_points[:, 2, 0]  # Range component
                range_rates = da_points[:, 5, 0]  # Range-rate component
                
                # Convert DA objects to constants if needed
                if hasattr(ranges[0], 'cons'):
                    ranges = np.array([r.cons() for r in ranges])
                    range_rates = np.array([rr.cons() for rr in range_rates])
                
                methods_info['DAIOD_DA'] = {
                    'type': 'DA_perimeter',
                    'n_points': len(ranges),
                    'range_mean': np.mean(ranges),
                    'range_std': np.std(ranges),
                    'range_min': np.min(ranges),
                    'range_max': np.max(ranges),
                    'rangerate_mean': np.mean(range_rates),
                    'rangerate_std': np.std(range_rates),
                    'rangerate_min': np.min(range_rates),
                    'rangerate_max': np.max(range_rates),
                }
                
                print(f"   DA Ranges: {len(ranges)} points")
                print(f"   Range: {np.min(ranges)/1e6:.2f} - {np.max(ranges)/1e6:.2f} Mm")
                print(f"   Range-rate: {np.min(range_rates):.4f} - {np.max(range_rates):.4f} km/s")
                
            except Exception as e:
                print(f"   Error extracting DA data: {e}")
    
    # 3. Monte Carlo Methods
    print("\n3. Monte Carlo Methods Analysis:")
    
    for method_name, data_key in [('DAIOD_MC', 'X_DAIOD_MC_geocentric_obs'), 
                                  ('GAUSS_MC', 'X_GAUSS_MC_geocentric_obs')]:
        if data_key in arc_data:
            obs_data = arc_data[data_key]
            print(f"   {method_name} structure: {type(obs_data)}, shape: {obs_data.shape}")
            
            try:
                if time_idx < obs_data.shape[2]:
                    ranges = obs_data[:, 2, time_idx]  # All samples, range component
                    range_rates = obs_data[:, 5, time_idx]  # All samples, range-rate component
                    
                    # Remove invalid points
                    valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
                    ranges_valid = ranges[valid_mask]
                    range_rates_valid = range_rates[valid_mask]
                    
                    methods_info[method_name] = {
                        'type': 'Monte_Carlo',
                        'n_points': len(ranges_valid),
                        'range_mean': np.mean(ranges_valid),
                        'range_std': np.std(ranges_valid),
                        'range_min': np.min(ranges_valid),
                        'range_max': np.max(ranges_valid),
                        'rangerate_mean': np.mean(range_rates_valid),
                        'rangerate_std': np.std(range_rates_valid),
                        'rangerate_min': np.min(range_rates_valid),
                        'rangerate_max': np.max(range_rates_valid),
                    }
                    
                    print(f"   {method_name} Ranges: {len(ranges_valid)} points")
                    print(f"   Range: {np.min(ranges_valid)/1e6:.2f} - {np.max(ranges_valid)/1e6:.2f} Mm")
                    print(f"   Range-rate: {np.min(range_rates_valid):.4f} - {np.max(range_rates_valid):.4f} km/s")
                    
            except Exception as e:
                print(f"   Error extracting {method_name} data: {e}")
    
    return methods_info

def compare_methods(methods_info):
    """Compare the different methods and analyze differences"""
    
    print("\n=== Method Comparison Summary ===")
    
    if not methods_info:
        print("No method data available for comparison.")
        return
    
    # Create comparison table
    print(f"{'Method':<15} {'Type':<12} {'N_Points':<8} {'Range_Span_Mm':<15} {'RangeRate_Span':<15}")
    print("-" * 75)
    
    for method_name, info in methods_info.items():
        range_span = (info['range_max'] - info['range_min']) / 1e6
        rangerate_span = info['rangerate_max'] - info['rangerate_min']
        
        print(f"{method_name:<15} {info['type']:<12} {info['n_points']:<8} "
              f"{range_span:<15.3f} {rangerate_span:<15.6f}")
    
    # Analyze the differences
    print("\n=== Analysis of Differences ===")
    
    # Compare DA/ADS methods to Monte Carlo methods
    da_ads_methods = [k for k, v in methods_info.items() if v['type'] in ['ADS_perimeter', 'DA_perimeter']]
    mc_methods = [k for k, v in methods_info.items() if v['type'] == 'Monte_Carlo']
    
    if da_ads_methods and mc_methods:
        print("\nDA/ADS vs Monte Carlo Comparison:")
        
        # Range comparison
        da_ads_range_spans = [((methods_info[m]['range_max'] - methods_info[m]['range_min']) / 1e6) 
                              for m in da_ads_methods]
        mc_range_spans = [((methods_info[m]['range_max'] - methods_info[m]['range_min']) / 1e6) 
                          for m in mc_methods]
        
        print(f"DA/ADS range span: {np.mean(da_ads_range_spans):.3f} ± {np.std(da_ads_range_spans):.3f} Mm")
        print(f"Monte Carlo range span: {np.mean(mc_range_spans):.3f} ± {np.std(mc_range_spans):.3f} Mm")
        
        # Range-rate comparison
        da_ads_rr_spans = [(methods_info[m]['rangerate_max'] - methods_info[m]['rangerate_min']) 
                           for m in da_ads_methods]
        mc_rr_spans = [(methods_info[m]['rangerate_max'] - methods_info[m]['rangerate_min']) 
                       for m in mc_methods]
        
        print(f"DA/ADS range-rate span: {np.mean(da_ads_rr_spans):.6f} ± {np.std(da_ads_rr_spans):.6f} km/s")
        print(f"Monte Carlo range-rate span: {np.mean(mc_rr_spans):.6f} ± {np.std(mc_rr_spans):.6f} km/s")
        
        # Statistical significance
        range_ratio = np.mean(da_ads_range_spans) / np.mean(mc_range_spans)
        rr_ratio = np.mean(da_ads_rr_spans) / np.mean(mc_rr_spans)
        
        print(f"\nRatio (DA/ADS : Monte Carlo):")
        print(f"Range span ratio: {range_ratio:.2f}")
        print(f"Range-rate span ratio: {rr_ratio:.2f}")
        
        if range_ratio > 2 or range_ratio < 0.5:
            print("WARNING: Significant difference in range spans detected!")
        if rr_ratio > 2 or rr_ratio < 0.5:
            print("WARNING: Significant difference in range-rate spans detected!")

def create_comparison_plot(methods_info):
    """Create a visual comparison of the range distributions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Range comparison
    methods = list(methods_info.keys())
    range_means = [methods_info[m]['range_mean']/1e6 for m in methods]
    range_stds = [methods_info[m]['range_std']/1e6 for m in methods]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(methods)]
    
    ax1.bar(methods, range_means, yerr=range_stds, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('Range [Mm]')
    ax1.set_title('Range Distribution Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Range-rate comparison
    rr_means = [methods_info[m]['rangerate_mean'] for m in methods]
    rr_stds = [methods_info[m]['rangerate_std'] for m in methods]
    
    ax2.bar(methods, rr_means, yerr=rr_stds, capsize=5, color=colors, alpha=0.7)
    ax2.set_ylabel('Range Rate [km/s]')
    ax2.set_title('Range-Rate Distribution Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main analysis function"""
    
    print("=== Range Distribution Analysis ===")
    
    # Load data
    result_data = load_da_params_and_data()
    if result_data is None:
        return
    
    # Analyze structures
    methods_info = analyze_range_data_structures(result_data)
    
    # Compare methods
    compare_methods(methods_info)
    
    # Create plots
    if methods_info:
        create_comparison_plot(methods_info)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
