import os
import sys
from pathlib import Path
import pickle
import glob
import numpy as np
import pandas as pd
from daceypy import DA, ADS, array
from astropy.time import Time, TimeDelta
from astropy import units as u
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import utils.post_process as post

def load_da_params_for_arc(arc_index, simulation_data_dir):
    """Load DA parameters for a specific arc to initialize DA before loading main data"""
    
    # Find DA parameters file for this arc
    da_params_pattern = str(simulation_data_dir / f"arc_{arc_index:03d}_da_params_*.pkl")
    da_params_files = glob.glob(da_params_pattern)
    
    if not da_params_files:
        print(f"No DA parameters file found for arc {arc_index}")
        return None
    
    # Use the most recent if multiple exist
    da_params_file = max(da_params_files, key=os.path.getctime)
    
    try:
        with open(da_params_file, 'rb') as f:
            da_params = pickle.load(f)
        
        print(f"Loaded DA params for arc {arc_index}: order={da_params['da_order']}, nvars={da_params['da_nvars']}")
        return da_params
        
    except Exception as e:
        print(f"Error loading DA parameters for arc {arc_index}: {e}")
        return None

def initialize_da_from_params(da_params):
    """Initialize DA with given parameters"""
    if da_params is None:
        print("No DA parameters provided, using default initialization")
        DA.init(6, 6)
        return
    
    DA.init(da_params['da_order'], da_params['da_nvars'])
    print(f"DA initialized with order={da_params['da_order']}, nvars={da_params['da_nvars']}")

def load_all_simulation_data():
    """Load all pickle files from the Simulation_data folder"""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    simulation_data_dir = script_dir / "Simulation_data"
    
    # Find all main result pickle files
    pickle_pattern = str(simulation_data_dir / "arc_*_complete_results_*.pkl")
    pickle_files = glob.glob(pickle_pattern)
    
    if not pickle_files:
        print(f"No pickle files found in {simulation_data_dir}")
        return None
    
    print(f"Found {len(pickle_files)} simulation data files")
    
    # Load and organize all data
    all_arc_data = {}
    
    for pickle_file in sorted(pickle_files):
        try:
            print(f"Loading: {Path(pickle_file).name}")
            
            # Extract arc index from filename
            filename = Path(pickle_file).name
            arc_index = int(filename.split('_')[1])
            
            # Load DA parameters first and initialize DA
            da_params = load_da_params_for_arc(arc_index, simulation_data_dir)
            initialize_da_from_params(da_params)
            
            # Now load the main arc data (ADS objects should work properly)
            with open(pickle_file, 'rb') as f:
                arc_data = pickle.load(f)
            
            all_arc_data[arc_index] = arc_data
            
            print(f"  Arc {arc_index}: dt={arc_data['current_dt']:.3f}d, "
                  f"{arc_data['Ts']} time steps, "
                  f"{len(arc_data['alphashapes']['DAIOD_ADS_DA'])} alphashapes per method")
            
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_arc_data)} arc datasets")
    return all_arc_data

def create_mock_query_results(arc_data):
    """Create mock query results that match the expected format for plots.py"""
    
    # Use the methods that actually exist in the simulation data
    # Note: plots.py expects different method names, but we'll use what we have
    methods = ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
    n_times = len(arc_data['tgrid'])
    n_methods = len(methods)
    
    # Create mock query results with zeros (since we're not actually querying)
    # but preserve the structure expected by plots.py
    query_results = {
        'N_images': np.zeros((n_times, n_methods)),
        'TP': np.zeros((n_times, n_methods)),
        'FP': np.zeros((n_times, n_methods)),
        'FN': np.zeros((n_times, n_methods)),
        'precision': np.zeros((n_times, n_methods)),
        'recall': np.zeros((n_times, n_methods)),
        'query_times': np.zeros((n_times, n_methods)),
        
        # Add geometry information storage (empty but with correct structure)
        'geometry_info': {
            'shape_types': np.full((n_times, n_methods), '', dtype='U30'),
            'vertex_counts': np.zeros((n_times, n_methods), dtype=int),
            'areas': np.zeros((n_times, n_methods)),
            'was_fixed': np.zeros((n_times, n_methods), dtype=bool),
            'fix_methods': np.full((n_times, n_methods), '', dtype='U30'),
            'processing_notes': np.full((n_times, n_methods), '', dtype='U100')
        }
    }
    
    return query_results

def format_simulation_data_for_plots(all_arc_data):
    """Format simulation data to match the structure expected by plots.py"""
    
    formatted_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for arc_index, arc_data in all_arc_data.items():
        
        # Create mock query results (plots.py expects these even if they're empty)
        query_results = create_mock_query_results(arc_data)
        
        # Create alphashape_areas structure that plots.py expects
        # Use pre-calculated areas from main_final.py (no need to recalculate)
        if 'alphashape_areas' in arc_data:
            # Use existing alphashape_areas structure from simulation data
            alphashape_areas = arc_data['alphashape_areas']
        else:
            # Fallback: create structure with dummy areas if not available
            methods = ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
            n_times = len(arc_data['tgrid'])
            alphashape_areas = {}
            for method in methods:
                alphashape_areas[method] = [0.0] * n_times
            print(f"Warning: 'alphashape_areas' not found in arc data, using zeros")
        
        # Create processing info to match query.py format
        n_times = len(arc_data['tgrid'])
        query_processing_info = {
            'query_indices': list(range(n_times)),
            'required_indices': [],
            'equally_spaced_indices': list(range(0, n_times, max(1, n_times//12))),
            'total_times_available': n_times,
            'total_times_processed': n_times
        }
        
        # Structure the data exactly as plots.py expects it
        # This matches the format from query.py's save_query_results function
        complete_arc_data = {
            # Copy all original arc data
            **arc_data,
            
            # Add alphashape_areas that plots.py expects
            'alphashape_areas': alphashape_areas,
            
            # Add mock query-specific results
            'query_results': query_results,
            'query_timestamp': timestamp,
            'query_processing_complete': True,
            
            # Query performance metrics by method
            'query_performance': {
                'methods': ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC'],
                'processed_time_indices': query_processing_info['query_indices'],
                'required_time_indices': query_processing_info['required_indices'],
                'equally_spaced_indices': query_processing_info['equally_spaced_indices'],
                'total_times_available': query_processing_info['total_times_available'],
                'total_times_processed': query_processing_info['total_times_processed'],
                'total_images_found': {
                    method: 0  # No actual queries performed
                    for method in ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
                },
                'total_true_positives': {
                    method: 0  # No actual queries performed
                    for method in ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
                },
                'average_precision': {
                    method: 0  # No actual queries performed
                    for method in ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
                },
                'average_recall': {
                    method: 0  # No actual queries performed
                    for method in ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
                },
                'total_query_time': {
                    method: 0  # No actual queries performed
                    for method in ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
                },
                
                # Add geometry statistics (empty but with correct structure)
                'geometry_statistics': {
                    method: {
                        'shape_type_counts': {
                            shape_type: 0
                            for shape_type in ['buffered_alphashape', 'buffered_convex_hull', 'alphashape', 'convex_hull', '']
                        },
                        'average_vertices': 0,
                        'average_area': 0,
                        'fix_rate': 0,
                        'successful_queries': 0,
                        'buffer_usage_rate': 0
                    }
                    for method in ['DAIOD_DA', 'DAIOD_ADS_DA', 'DAIOD_MC', 'GAUSS_MC']
                }
            }
        }
        
        # Format the result to match what plots.py expects
        # plots.py expects: result_data['arc_data'] and result_data['query_results']
        formatted_results[arc_index] = {
            'arc_data': complete_arc_data,
            'query_results': query_results,
            'query_processing_info': query_processing_info
        }

        print(f"Formatted arc {arc_index} for plots.py compatibility")
    
    return formatted_results

def save_formatted_results(formatted_results, output_dir):
    """Save formatted results to files that plots.py can read"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create Query_data directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for arc_idx, result_data in formatted_results.items():
        arc_data = result_data['arc_data']
        
        # Extract arc length for filename
        dt_days = arc_data['current_dt']
        
        # Save formatted arc results file (matching query.py naming pattern)
        arc_results_path = output_dir / f"arc_{arc_idx:03d}_dt_{dt_days:.3f}d_formatted_results_{timestamp}.pkl"
        
        try:
            with open(arc_results_path, 'wb') as f:
                pickle.dump(arc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = arc_results_path.stat().st_size / (1024**2)
            print(f"✓ Arc {arc_idx} formatted results saved: {arc_results_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Arc length: {dt_days:.3f} days")
            
            saved_files.append(arc_results_path)
            
        except Exception as e:
            print(f"✗ Error saving arc {arc_idx} formatted results: {e}")
        
        # Also save DA parameters (matching query.py pattern)
        da_params_path = output_dir / f"arc_{arc_idx:03d}_formatted_da_params_{timestamp}.pkl"
        da_params = {
            'da_order': arc_data.get('algorithm_params', {}).get('da_order', 5),
            'da_nvars': arc_data.get('algorithm_params', {}).get('da_nvars', 6),
            'arc_index': arc_idx,
            'timestamp': timestamp,
            'formatted_for_plots': True
        }
        
        try:
            with open(da_params_path, 'wb') as f:
                pickle.dump(da_params, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Formatted DA parameters saved: {da_params_path}")
        except Exception as e:
            print(f"✗ Error saving formatted DA parameters for arc {arc_idx}: {e}")
        
        print(f"Arc {arc_idx} formatting complete.\n")
    
    print(f"\n=== Formatted Results Saved ===")
    print(f"Processed {len(formatted_results)} arcs")
    print(f"Individual arc files: {len(saved_files)}")
    print(f"Files saved to: {output_dir}")
    print(f"\nThese files can now be used with plots.py")
    
    return saved_files

def main():
    """Main formatting function"""
    
    print("=== Simulation Data Formatting for Plots ===")
    print("This script formats simulation data to be compatible with plots.py")
    print("without performing actual database queries.\n")
    
    # Load all simulation data
    all_arc_data = load_all_simulation_data()
    if all_arc_data is None:
        return
    
    # Format data for plots.py compatibility
    print("\nFormatting data for plots.py compatibility...")
    formatted_results = format_simulation_data_for_plots(all_arc_data)
    
    # Save formatted results to Query_data folder
    script_dir = Path(__file__).parent
    query_output_dir = script_dir / "Query_data"
    
    if formatted_results:
        saved_files = save_formatted_results(formatted_results, query_output_dir)
        
        print(f"\n=== Formatting Complete ===")
        print(f"Formatted {len(formatted_results)} arcs")
        print(f"Saved {len(saved_files)} individual arc result files")
        print(f"Results directory: {query_output_dir}")
        print(f"\nYou can now run plots.py to visualize the simulation results!")
    else:
        print("No data to format.")

if __name__ == "__main__":
    main()