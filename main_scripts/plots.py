import os
import sys
from pathlib import Path
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from astropy.time import Time
from astropy import units as u
from daceypy import DA, ADS, array

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import utils.plotting as plot_utils
import utils.post_process as post

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

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
        DA.init(4, 6)
        return
    
    DA.init(da_params['da_order'], da_params['da_nvars'])
    print(f"DA initialized with order={da_params['da_order']}, nvars={da_params['da_nvars']}")

def get_arc_data(result_data):
    """Helper function to extract arc data from result_data, handling both old and new formats"""
    if 'arc_data' in result_data:
        # Old format: data is nested under 'arc_data' key
        return result_data['arc_data'], result_data['query_results']
    else:
        # New format: arc data is directly at top level, query_results is a separate key
        return result_data, result_data['query_results']

def load_query_results():
    """Load all individual arc query results with proper DA initialization"""
    
    script_dir = Path(__file__).parent
    query_data_dir = script_dir / "Query_data"
    simulation_data_dir = script_dir / "Simulation_data"  # For DA params
    
    # Find all individual arc query result files
    pattern = str(query_data_dir / "arc_*_query_results_*.pkl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print("No individual arc query result files found!")
        print(f"Looked in: {query_data_dir}")
        return None
    
    print(f"Found {len(result_files)} individual arc query result files")
    
    # Find all DA parameter files to initialize DA properly
    da_params_pattern = str(simulation_data_dir / "arc_*_da_params_*.pkl")
    da_params_files = glob.glob(da_params_pattern)
    
    if not da_params_files:
        print("No DA parameter files found! Using default DA initialization.")
        DA.init(4, 6)  # Default initialization
    else:
        # Get unique arc indices from DA parameter files
        arc_indices = set()
        for da_file in da_params_files:
            try:
                # Extract arc index from filename like "arc_000_da_params_*.pkl"
                arc_idx = int(Path(da_file).name.split('_')[1])
                arc_indices.add(arc_idx)
            except (ValueError, IndexError):
                continue
        
        if arc_indices:
            # Initialize DA with parameters from the first arc (they should all be the same)
            first_arc = min(arc_indices)
            print(f"Initializing DA using parameters from arc {first_arc}...")
            da_params = load_da_params_for_arc(first_arc, simulation_data_dir)
            initialize_da_from_params(da_params)
        else:
            print("Could not determine arc indices from DA parameter files. Using default DA initialization.")
            DA.init(4, 6)
    
    # Load all individual arc query results
    print("Loading individual arc query results...")
    all_query_results = {}
    
    for result_file in sorted(result_files):
        try:
            # Extract arc index from filename like "arc_000_query_results_*.pkl"
            filename = Path(result_file).name
            arc_idx = int(filename.split('_')[1])
            
            print(f"Loading arc {arc_idx} from: {filename}")
            
            with open(result_file, 'rb') as f:
                arc_result_data = pickle.load(f)
            
            all_query_results[arc_idx] = arc_result_data
            
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    
    print(f"Successfully loaded query results for {len(all_query_results)} arcs")
    return all_query_results

def extract_data_for_plotting(all_query_results):
    """Extract and organize data for plotting"""
    
    methods = ['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC']
    method_colors = {'DAIOD_ADS_ADS': 'red', 'DAIOD_ADS': 'blue', 'DAIOD_DA': 'green', 
                     'DAIOD_MC': 'orange', 'GAUSS_MC': 'purple'}
    method_markers = {'DAIOD_ADS_ADS': 'o', 'DAIOD_ADS': 's', 'DAIOD_DA': '^', 
                      'DAIOD_MC': 'D', 'GAUSS_MC': 'v'}
    
    # Collect all data
    plot_data = []
    
    for arc_idx, result_data in all_query_results.items():
        # Use helper function to handle both old and new data formats
        arc_data, query_results = get_arc_data(result_data)
        
        dt_days = arc_data['current_dt']
        method_clock_time = arc_data['Method_Clock_time']
        
        # Get propagation times in days
        t_prop = arc_data['t_propagation']
        tgrid = arc_data['tgrid']
        prop_times_days = (t_prop - t_prop[0]).to(u.day).value
        
        for time_idx in range(len(prop_times_days)):
            for method_idx, method_name in enumerate(methods):
                
                # Calculate total wall clock time (OD + Prop + Eval + Query)
                total_wall_time = (method_clock_time[method_idx, :4].sum() + 
                                 query_results['query_times'][time_idx, method_idx])
                
                plot_data.append({
                    'arc_index': arc_idx,
                    'dt_days': dt_days,
                    'time_idx': time_idx,
                    'prop_time_days': prop_times_days[time_idx],
                    'method': method_name,
                    'N_images': query_results['N_images'][time_idx, method_idx],
                    'TP': query_results['TP'][time_idx, method_idx],
                    'FP': query_results['FP'][time_idx, method_idx],
                    'FN': query_results['FN'][time_idx, method_idx],
                    'precision': query_results['precision'][time_idx, method_idx],
                    'recall': query_results['recall'][time_idx, method_idx],
                    'f1_score': (2 * query_results['precision'][time_idx, method_idx] * 
                               query_results['recall'][time_idx, method_idx]) / 
                               (query_results['precision'][time_idx, method_idx] + 
                                query_results['recall'][time_idx, method_idx] + 1e-10),
                    'total_wall_time': total_wall_time,
                    'od_time': method_clock_time[method_idx, 0],
                    'prop_time': method_clock_time[method_idx, 1],
                    'eval_time': method_clock_time[method_idx, 2],
                    'query_time': query_results['query_times'][time_idx, method_idx]
                })
    
    df = pd.DataFrame(plot_data)
    return df, methods, method_colors, method_markers

def plot_parameter_heatmaps(df, methods):
    """Plot heatmaps for Arc length vs Propagation time vs various metrics"""
    
    metrics = ['N_images', 'precision', 'recall', 'f1_score', 'total_wall_time']
    metric_labels = ['Number of Images', 'Precision', 'Recall', 'F1 Score', 'Total Wall Clock Time [s]']
    
    fig, axes = plt.subplots(len(methods), len(metrics), figsize=(20, 16))
    fig.suptitle('Parameter Space Analysis: Arc Length vs Propagation Time', fontsize=16)
    
    for method_idx, method in enumerate(methods):
        method_data = df[df['method'] == method]
        
        for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[method_idx, metric_idx]
            
            # Create pivot table for heatmap
            pivot_data = method_data.pivot_table(
                index='dt_days', 
                columns='prop_time_days', 
                values=metric, 
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                im = ax.imshow(pivot_data.values, aspect='auto', origin='lower',
                              extent=[pivot_data.columns.min(), pivot_data.columns.max(),
                                     pivot_data.index.min(), pivot_data.index.max()],
                              cmap='viridis')
                
                plt.colorbar(im, ax=ax, label=label)
            
            ax.set_title(f'{method} - {label}')
            ax.set_xlabel('Propagation Time [days]')
            ax.set_ylabel('Arc Length [days]')
    
    plt.tight_layout()
    plt.show()

def plot_radec_propagation(all_query_results, methods, method_colors):
    """Plot RA×DEC propagated states with separate figures for each method and arc length combination"""
    
    # Get arc information
    arc_info = {}
    for arc_idx, result_data in all_query_results.items():
        arc_data, _ = get_arc_data(result_data)
        arc_info[arc_idx] = {
            'dt_days': arc_data['current_dt'],
            'tgrid': arc_data['tgrid'],
            't_propagation': arc_data['t_propagation']
        }
    
    arc_lengths = [f"{info['dt_days']:.2f}d" for info in arc_info.values()]
    print(f"Found {len(arc_info)} arcs with lengths: {arc_lengths}")
    
    # Determine number of time steps to show (use all time steps for time series)
    sample_arc_data, _ = get_arc_data(list(all_query_results.values())[0])
    total_time_steps = len(sample_arc_data['tgrid'])
    all_time_indices = list(range(total_time_steps))
    
    print(f"Showing complete time series with {total_time_steps} time steps")
    
    # Load true Apophis data for comparison
    try:
        t_prop_full = sample_arc_data['t_propagation']
        
        # Get true Apophis ephemeris for the full time range
        eph_true, _ = post.load_Apophis_Ephemeris(
            t_prop_full[0].iso, t_prop_full[-1].iso, '30min'
        )
        ra_true = eph_true['RA'].to(u.deg).value
        dec_true = eph_true['DEC'].to(u.deg).value
        
    except Exception as e:
        print(f"Could not load true Apophis data: {e}")
        ra_true = dec_true = None
    
    # Create separate figure for each method and arc length combination
    for method in methods:
        for arc_idx, result_data in all_query_results.items():
            arc_data, _ = get_arc_data(result_data)
            dt_days = arc_data['current_dt']
            t_prop = arc_data['t_propagation']
            
            # Create single figure for this method-arc combination showing time evolution
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.suptitle(f'{method} - RA×DEC Time Series\nArc Length: {dt_days:.2f} days, Arc #{arc_idx}', 
                        fontsize=14, y=0.95)
            
            try:
                # Extract RA/DEC points and alphashapes for this method
                ra_dec_points = arc_data['ra_dec_points'][method]
                ra_dec_alphashapes = arc_data.get('ra_dec_alphashapes', {}).get(method, [])
                
                # Plot time evolution on single axes
                for time_idx in all_time_indices:
                    if time_idx < len(ra_dec_points):
                        points = ra_dec_points[time_idx]
                        
                        if points is not None and len(points) > 0:
                            # Color based on time progression
                            time_fraction = time_idx / (len(all_time_indices) - 1) if len(all_time_indices) > 1 else 0
                            alpha_val = 0.3 + 0.5 * time_fraction  # Fade from light to darker
                            
                            # Plot scatter points (no individual labels to avoid clutter)
                            scatter = ax.scatter(points[:, 0], points[:, 1], 
                                               c=time_fraction, cmap='plasma', 
                                               alpha=alpha_val, s=1,
                                               vmin=0, vmax=1)
                            
                            # Overlay alpha shape if available - only for selected time steps to avoid clutter
                            step_interval = max(1, len(all_time_indices)//5) if len(all_time_indices) > 5 else 1
                            if (time_idx % step_interval == 0 and 
                                time_idx < len(ra_dec_alphashapes)):
                                alphashape = ra_dec_alphashapes[time_idx]
                                if alphashape is not None and hasattr(alphashape, 'exterior'):
                                    # Extract boundary coordinates
                                    boundary_coords = np.array(alphashape.exterior.coords)
                                    
                                    # Plot alpha shape boundary
                                    ax.plot(boundary_coords[:, 0], boundary_coords[:, 1], 
                                           color=plt.cm.plasma(time_fraction), linewidth=1.5, 
                                           alpha=0.8, linestyle='-')
                
                # Add colorbar to show time progression
                sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='Time Progression (0=Start, 1=End)')
                
                # Plot true Apophis trajectory if available
                if ra_true is not None:
                    # Subsample true trajectory for clarity
                    n_true_points = min(len(ra_true), 50)
                    true_indices = np.linspace(0, len(ra_true)-1, n_true_points, dtype=int)
                    
                    ax.plot(ra_true[true_indices], dec_true[true_indices], 
                           'k-', linewidth=2, label='True Apophis Trajectory', zorder=10, alpha=0.8)
                    ax.scatter(ra_true[true_indices[0]], dec_true[true_indices[0]], 
                             c='green', marker='*', s=150, label='True Start', zorder=11)
                    ax.scatter(ra_true[true_indices[-1]], dec_true[true_indices[-1]], 
                             c='red', marker='*', s=150, label='True End', zorder=11)
                
                ax.set_xlabel('RA [deg]')
                ax.set_ylabel('DEC [deg]')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                
                # Set equal aspect ratio for proper sky projection
                ax.set_aspect('equal', adjustable='box')
                
                # Add time range info to the plot
                time_start = t_prop[0].iso[:19]
                time_end = t_prop[-1].iso[:19]
                ax.text(0.02, 0.98, f'Time Range:\n{time_start}\nto\n{time_end}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error plotting {method} arc {arc_idx}: {e}")
                ax.text(0.5, 0.5, f'Error plotting {method} arc {arc_idx}:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, right=0.85)
            plt.show()

def plot_range_rangerate_phase_space(all_query_results, methods, method_colors):
    """Plot Range×Range-rate phase space for all methods at each time step"""
    
    # Get the number of time steps from first arc
    arc_data, _ = get_arc_data(list(all_query_results.values())[0])
    n_time_steps = len(arc_data['tgrid'])
    
    # Create subplots for each time step
    n_cols = min(3, n_time_steps)
    n_rows = (n_time_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Ensure axes is always iterable
    if n_time_steps == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols == 1:
        axes = [axes]  
    elif n_rows == 1:
        axes = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    # Skip true Apophis range data for now (requires additional conversion functions)
    apophis_true_obs = None
    
    for time_idx in range(n_time_steps):
        if time_idx < len(axes):
            ax = axes[time_idx]
        else:
            continue
            
        # Plot each method at this time step
        for method_idx, method in enumerate(methods):
            
            for arc_idx, result_data in all_query_results.items():
                arc_data, _ = get_arc_data(result_data)
                
                try:
                    # Extract observational data based on method type
                    if 'ADS' in method:
                        # For ADS methods, extract from perimeter evaluation
                        if method == 'DAIOD_ADS_ADS':
                            perimeter_data = arc_data['DAIOD_ADS_perimeter']
                        else:
                            perimeter_data = arc_data['DAIOD_perimeter']
                            
                        if time_idx < len(perimeter_data['final_map']):
                            manifold = perimeter_data['final_map'][time_idx]
                            # Extract range (element 2) and range-rate (element 5)
                            ranges = manifold[:, 2, :].flatten()  # All ranges
                            range_rates = manifold[:, 5, :].flatten()  # All range-rates
                            
                            # Remove invalid points
                            valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
                            ranges = ranges[valid_mask]
                            range_rates = range_rates[valid_mask]
                            
                    elif method == 'DAIOD_DA':
                        # For DA method, extract from DA perimeter
                        if time_idx < len(arc_data['DAIOD_DA_perimeter']):
                            da_points = arc_data['DAIOD_DA_perimeter'][time_idx]
                            ranges = da_points[:, 2, 0]  # Range component
                            range_rates = da_points[:, 5, 0]  # Range-rate component
                            
                    else:
                        # For MC methods, extract from propagated observational data
                        if method == 'DAIOD_MC':
                            obs_data = arc_data['X_DAIOD_MC_geocentric_obs']
                        else:  # GAUSS_MC
                            obs_data = arc_data['X_GAUSS_MC_geocentric_obs']
                            
                        if time_idx < obs_data.shape[2]:
                            ranges = obs_data[:, 2, time_idx]  # All samples, range component
                            range_rates = obs_data[:, 5, time_idx]  # All samples, range-rate component
                            
                            # Remove invalid points
                            valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
                            ranges = ranges[valid_mask]
                            range_rates = range_rates[valid_mask]
                    
                    # Plot the data
                    if len(ranges) > 0 and len(range_rates) > 0:
                        ax.scatter(ranges/1e6, range_rates, c=method_colors[method], 
                                 alpha=0.6, s=2, label=f'{method}' if arc_idx == 0 else "")
                    
                except Exception as e:
                    print(f"Error plotting range data for {method} at time {time_idx}: {e}")
                    continue
        
        # Plot true Apophis if available
        if apophis_true_obs and time_idx < len(apophis_true_obs):
            true_range, true_range_rate = apophis_true_obs[time_idx]
            ax.scatter(true_range/1e6, true_range_rate, c='black', marker='*', 
                      s=100, label='True Apophis', zorder=10)
        
        ax.set_xlabel('Range [Mm]')
        ax.set_ylabel('Range Rate [km/s]')
        ax.set_title(f'Range×Range-rate Phase Space - Time {time_idx}')
        ax.grid(True, alpha=0.3)
        if time_idx == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_time_steps, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_ads_split_history(all_query_results):
    """Plot ADS domain split history over propagation time"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ads_methods = ['DAIOD_ADS_ADS', 'DAIOD_ADS']
    colors = ['red', 'blue']
    
    for arc_idx, result_data in all_query_results.items():
        arc_data, _ = get_arc_data(result_data)
        tgrid = arc_data['tgrid']
        prop_times_days = (arc_data['t_propagation'] - arc_data['t_propagation'][0]).to(u.day).value
        
        for method_idx, method in enumerate(ads_methods):
            try:
                if method == 'DAIOD_ADS_ADS':
                    final_lists = arc_data['final_lists_ADS']
                else:
                    final_lists = arc_data['final_lists_ADS_DAIOD']
                
                nsplits = [len(domains) for domains in final_lists]
                
                ax.plot(prop_times_days[:len(nsplits)], nsplits, 
                       color=colors[method_idx], marker='o', 
                       label=f'{method} (Arc {arc_idx})', linewidth=2)
                
            except Exception as e:
                print(f"Error plotting split history for {method}: {e}")
    
    ax.set_xlabel('Propagation Time [days]')
    ax.set_ylabel('Number of ADS Domains')
    ax.set_title('ADS Domain Split History')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

def plot_wall_clock_comparison(df, methods):
    """Plot wall clock time comparison as bar chart"""
    
    # Aggregate data by method
    method_times = df.groupby('method').agg({
        'od_time': 'mean',
        'prop_time': 'mean', 
        'eval_time': 'mean',
        'query_time': 'mean'
    }).reset_index()
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.6
    
    bottom = np.zeros(len(methods))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Orbit Determination', 'Propagation', 'Evaluation', 'Query']
    
    for i, (col, color, label) in enumerate(zip(['od_time', 'prop_time', 'eval_time', 'query_time'], 
                                               colors, labels)):
        values = [method_times[method_times['method'] == method][col].iloc[0] 
                 if len(method_times[method_times['method'] == method]) > 0 else 0 
                 for method in methods]
        
        ax.bar(x, values, width, bottom=bottom, label=label, color=color)
        bottom += values
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Wall Clock Time [s]')
    ax.set_title('Computation Time Breakdown by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_precision_vs_walltime_scatter(df, method_colors, method_markers):
    """Plot precision vs wall clock time scatter (Pareto analysis)"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        ax.scatter(method_data['total_wall_time'], method_data['precision'],
                  c=method_colors[method], marker=method_markers[method], 
                  s=60, alpha=0.7, label=method, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Total Wall Clock Time [s]')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Computational Cost (Pareto Analysis)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add trend line for each method
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        if len(method_data) > 1:
            z = np.polyfit(method_data['total_wall_time'], method_data['precision'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(method_data['total_wall_time'].min(), 
                                method_data['total_wall_time'].max(), 100)
            ax.plot(x_trend, p(x_trend), '--', color=method_colors[method], alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_cumulative_walltime(df, methods, method_colors):
    """Plot cumulative wall clock time vs propagation time"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in methods:
        method_data = df[df['method'] == method].sort_values(['arc_index', 'prop_time_days'])
        
        # Group by arc and calculate cumulative time
        for arc_idx in method_data['arc_index'].unique():
            arc_data = method_data[method_data['arc_index'] == arc_idx]
            
            cumulative_time = arc_data['total_wall_time'].cumsum()
            
            ax.plot(arc_data['prop_time_days'], cumulative_time, 
                   color=method_colors[method], marker='o', 
                   label=f'{method} (Arc {arc_idx})', linewidth=2, markersize=4)
    
    ax.set_xlabel('Propagation Time [days]')
    ax.set_ylabel('Cumulative Wall Clock Time [s]')
    ax.set_title('Cumulative Computational Cost Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_orbital_motion_validation(all_query_results, methods, method_colors):
    """Plot motion in orbital frame to validate physical consistency
    
    Converts from geocentric ICRS observational coordinates to heliocentric perifocal frame:
    1. Observational -> Geocentric Cartesian ICRS
    2. Geocentric -> Heliocentric ICRS (translation)
    3. Heliocentric ICRS -> Heliocentric Perifocal (plane change)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Load true Apophis orbital motion for comparison
    try:
        arc_data, _ = get_arc_data(list(all_query_results.values())[0])
        t_prop = arc_data['t_propagation']
        
        # Get true Apophis vectors in heliocentric frame
        _, vec_true = post.load_Apophis_Ephemeris(
            t_prop[0].iso, t_prop[-1].iso, '1h', location='500@10'
        )
        
        # True Apophis heliocentric ICRS state
        apophis_helio_true = np.array([
            vec_true['x'].to(u.km).value,
            vec_true['y'].to(u.km).value, 
            vec_true['z'].to(u.km).value,
            vec_true['vx'].to(u.km/u.s).value,
            vec_true['vy'].to(u.km/u.s).value,
            vec_true['vz'].to(u.km/u.s).value
        ])
        
        # Convert all time steps to orbital frame for true trajectory
        orbital_true_traj = []
        for i in range(apophis_helio_true.shape[1]):
            orbital_state = plot_utils.state_to_orbital_frame(apophis_helio_true[:, i], mu=1.32712440018e11)
            orbital_true_traj.append(orbital_state[:3])  # Position only
        orbital_true_traj = np.array(orbital_true_traj)
        
    except Exception as e:
        print(f"Could not load true Apophis orbital data: {e}")
        orbital_true_traj = None
    
    for method_idx, method in enumerate(methods):
        if method_idx >= len(axes):
            continue
            
        ax = axes[method_idx]
        
        # Collect all orbital motion data for this method
        orbital_positions = []
        
        for arc_idx, result_data in all_query_results.items():
            arc_data, _ = get_arc_data(result_data)
            earth_pos = arc_data['Earth_propagated_position']  # [6, n_times] heliocentric ICRS
            
            try:
                # Extract observational data based on method type and convert to orbital
                if 'ADS' in method:
                    # For ADS methods, extract from perimeter evaluation
                    if method == 'DAIOD_ADS_ADS':
                        perimeter_data = arc_data['DAIOD_ADS_perimeter']
                    else:
                        perimeter_data = arc_data['DAIOD_perimeter']
                        
                    for time_idx in range(len(perimeter_data['final_map'])):
                        if time_idx >= earth_pos.shape[1]:
                            break
                            
                        manifold = perimeter_data['final_map'][time_idx]
                        earth_state_t = earth_pos[:, time_idx]
                        
                        # Process each sample in the manifold
                        for sample_idx in range(min(10, manifold.shape[0])):  # Limit samples for clarity
                            for param_idx in range(min(5, manifold.shape[2])):  # Limit parameters
                                obs_state = manifold[sample_idx, :, param_idx]
                                
                                if np.all(np.isfinite(obs_state)):
                                    try:
                                        # Step 1: Observational -> Geocentric Cartesian ICRS
                                        cart_geo = post.convert_observational_to_cartesian(obs_state)
                                        
                                        # Step 2: Geocentric -> Heliocentric ICRS (translation)
                                        cart_helio = cart_geo + earth_state_t
                                        
                                        # Step 3: Heliocentric ICRS -> Heliocentric Perifocal
                                        orbital_state = plot_utils.state_to_orbital_frame(cart_helio)
                                        orbital_positions.append(orbital_state[:3])  # Position only
                                        
                                    except Exception as conv_error:
                                        continue  # Skip invalid conversions
                
                elif method == 'DAIOD_DA':
                    # For DA method, extract from DA perimeter
                    da_perimeter = arc_data.get('DAIOD_DA_perimeter', [])
                    for time_idx, da_points in enumerate(da_perimeter):
                        if time_idx >= earth_pos.shape[1]:
                            break
                            
                        earth_state_t = earth_pos[:, time_idx]
                        
                        for point_idx, point in enumerate(da_points):
                            if point_idx >= 20:  # Limit points for clarity
                                break
                                
                            obs_state = point[:, 0]  # Extract DA constant values
                            
                            if np.all(np.isfinite(obs_state)):
                                try:
                                    # Step 1: Observational -> Geocentric Cartesian ICRS
                                    cart_geo = post.convert_observational_to_cartesian(obs_state)
                                    
                                    # Step 2: Geocentric -> Heliocentric ICRS (translation)
                                    cart_helio = cart_geo + earth_state_t
                                    
                                    # Step 3: Heliocentric ICRS -> Heliocentric Perifocal
                                    orbital_state = plot_utils.state_to_orbital_frame(cart_helio)
                                    orbital_positions.append(orbital_state[:3])  # Position only
                                    
                                except Exception as conv_error:
                                    continue  # Skip invalid conversions
                
                else:
                    # For MC methods, extract from observational data
                    if method == 'DAIOD_MC':
                        obs_data = arc_data['X_DAIOD_MC_geocentric_obs']
                    else:  # GAUSS_MC
                        obs_data = arc_data['X_GAUSS_MC_geocentric_obs']
                        
                    for time_idx in range(min(obs_data.shape[2], earth_pos.shape[1])):
                        earth_state_t = earth_pos[:, time_idx]
                        
                        # Sample subset of Monte Carlo samples for clarity
                        sample_indices = np.linspace(0, obs_data.shape[0]-1, 
                                                   min(50, obs_data.shape[0]), dtype=int)
                        
                        for sample_idx in sample_indices:
                            obs_state = obs_data[sample_idx, :, time_idx]
                            
                            if np.all(np.isfinite(obs_state)):
                                try:
                                    # Step 1: Observational -> Geocentric Cartesian ICRS
                                    cart_geo = post.convert_observational_to_cartesian(obs_state)
                                    
                                    # Step 2: Geocentric -> Heliocentric ICRS (translation)
                                    cart_helio = cart_geo + earth_state_t
                                    
                                    # Step 3: Heliocentric ICRS -> Heliocentric Perifocal
                                    orbital_state = plot_utils.state_to_orbital_frame(cart_helio)
                                    orbital_positions.append(orbital_state[:3])  # Position only
                                    
                                except Exception as conv_error:
                                    continue  # Skip invalid conversions
                
            except Exception as e:
                print(f"Error processing orbital data for {method} arc {arc_idx}: {e}")
                continue
        
        # Plot orbital positions in perifocal frame
        if orbital_positions:
            orbital_positions = np.array(orbital_positions)
            
            # Plot radial vs along-track (perifocal x-y plane)
            ax.scatter(orbital_positions[:, 0]/1e6, orbital_positions[:, 1]/1e6, 
                      c=method_colors[method], alpha=0.6, s=1, label=method)
        
        # Plot true Apophis trajectory if available  
        if orbital_true_traj is not None:
            ax.plot(orbital_true_traj[:, 0]/1e6, orbital_true_traj[:, 1]/1e6, 
                   'k-', linewidth=2, label='True Apophis', zorder=10)
            ax.scatter(orbital_true_traj[0, 0]/1e6, orbital_true_traj[0, 1]/1e6,
                      c='black', marker='*', s=100, zorder=11)  # Mark start
        
        ax.set_xlabel('Radial Distance (Perifocal) [Mm]')
        ax.set_ylabel('Along-track Distance (Perifocal) [Mm]')
        ax.set_title(f'{method} - Heliocentric Perifocal Motion')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    # Hide empty subplots
    for i in range(len(methods), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main plotting function"""
    
    print("=== Apophis Results Plotting ===")
    
    # Load query results
    all_query_results = load_query_results()
    if all_query_results is None:
        return
    
    # Extract data for plotting
    df, methods, method_colors, method_markers = extract_data_for_plotting(all_query_results)
    print(f"Loaded data: {len(df)} records across {len(methods)} methods")
    
    # Generate all plots
    print("\n1. Generating parameter space heatmaps...")
    plot_parameter_heatmaps(df, methods)
    
    print("2. Generating RA×DEC propagation plots...")
    plot_radec_propagation(all_query_results, methods, method_colors)
    
    print("3. Generating Range×Range-rate phase space plots...")
    plot_range_rangerate_phase_space(all_query_results, methods, method_colors)
    
    print("4. Generating ADS split history...")
    plot_ads_split_history(all_query_results)
    
    print("5. Generating wall clock time comparison...")
    plot_wall_clock_comparison(df, methods)
    
    print("6. Generating precision vs wall time scatter...")
    plot_precision_vs_walltime_scatter(df, method_colors, method_markers)
    
    print("7. Generating cumulative wall time plot...")
    plot_cumulative_walltime(df, methods, method_colors)
    
    print("8. Generating orbital motion validation...")
    plot_orbital_motion_validation(all_query_results, methods, method_colors)
    
    print("\n=== Plotting Complete ===")

if __name__ == "__main__":
    main()