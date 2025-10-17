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
from astropy.time import Time, TimeDelta
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

def setup_plot_directory(plot_type):
    """Create plot directory structure and return the path"""
    script_dir = Path(__file__).parent
    plot_dir = script_dir / "Plots" / plot_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def generate_plot_filename(plot_type, base_name, extension='png', timestamp=True):
    """Generate standardized filename for plots"""
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp_str}.{extension}"
    else:
        filename = f"{base_name}.{extension}"
    
    plot_dir = setup_plot_directory(plot_type)
    return plot_dir / filename

def save_figure_with_metadata(fig, filepath, dpi=300, bbox_inches='tight'):
    """Save figure with consistent settings and metadata"""
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving plot to {filepath}: {e}")
        return False

def load_da_params_for_arc(arc_index, simulation_data_dir):
    """Load DA parameters for a specific arc to initialize DA before loading main data"""
    
    # Find DA parameters file for this arc - check Query_data first, then Simulation_data
    query_data_dir = simulation_data_dir.parent / "Query_data"
    da_params_pattern_query = str(query_data_dir / f"arc_{arc_index:03d}_formatted_da_params_*.pkl")
    da_params_pattern_sim = str(simulation_data_dir / f"arc_{arc_index:03d}_da_params_*.pkl")
    
    da_params_files = glob.glob(da_params_pattern_query)
    if not da_params_files:
        da_params_files = glob.glob(da_params_pattern_sim)
    
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
    # Updated pattern to match format_for_plots.py output: arc_XXX_dt_X.XXXd_formatted_results_TIMESTAMP.pkl
    pattern = str(query_data_dir / "arc_*_dt_*_formatted_results_*.pkl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print("No individual arc query result files found!")
        print(f"Looked in: {query_data_dir}")
        print(f"Looking for pattern: arc_*_dt_*_formatted_results_*.pkl")
        return None
    
    print(f"Found {len(result_files)} individual arc query result files")
    
    # Find all DA parameter files to initialize DA properly
    # First try Query_data directory (from format_for_plots.py), then Simulation_data
    da_params_pattern_query = str(query_data_dir / "arc_*_formatted_da_params_*.pkl")
    da_params_pattern_sim = str(simulation_data_dir / "arc_*_da_params_*.pkl")
    
    da_params_files = glob.glob(da_params_pattern_query)
    if not da_params_files:
        da_params_files = glob.glob(da_params_pattern_sim)
    
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
            # Extract arc index from filename like "arc_000_dt_0.180d_formatted_results_*.pkl"
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
    
    # Updated methods to match stored structure: ['DAIOD_ADS_DA', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC']
    methods = ['DAIOD_ADS_DA', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC']
    # methods = ['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC']  # Old methods list
    method_colors = {'DAIOD_ADS_DA': 'red', 'DAIOD_DA': 'green', 
                     'DAIOD_MC': 'orange', 'GAUSS_MC': 'purple'}
    # method_colors = {'DAIOD_ADS_ADS': 'red', 'DAIOD_ADS': 'blue', 'DAIOD_DA': 'green', 
    #                  'DAIOD_MC': 'orange', 'GAUSS_MC': 'purple'}  # Old method colors
    method_markers = {'DAIOD_ADS_DA': 'o', 'DAIOD_DA': '^', 
                      'DAIOD_MC': 'D', 'GAUSS_MC': 'v'}
    # method_markers = {'DAIOD_ADS_ADS': 'o', 'DAIOD_ADS': 's', 'DAIOD_DA': '^', 
    #                   'DAIOD_MC': 'D', 'GAUSS_MC': 'v'}  # Old method markers
    
    # Collect all data
    plot_data = []
    
    for arc_idx, result_data in all_query_results.items():
        # Use helper function to handle both old and new data formats
        arc_data, query_results = get_arc_data(result_data)
        
        dt_days = arc_data['current_dt']
        method_clock_time = arc_data['Method_Clock_time']
        # restructure method_clock_time 
        method_clock_time[2,2] = method_clock_time[2,1]
        method_clock_time = np.delete(method_clock_time, 1, axis=0) 

        # Get propagation times in days
        t_prop = arc_data['t_propagation']
        tgrid = arc_data['tgrid']
        prop_times_days = (t_prop - t_prop[0]).to(u.day).value
        
        for time_idx in range(len(prop_times_days)):
            for method_idx, method_name in enumerate(methods):
                
                # Calculate total wall clock time (OD + Prop + + conversion + eval + alphashape + Query)
                total_wall_time = (method_clock_time[method_idx, :5].sum() + 
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
                    'conversion_time': method_clock_time[method_idx, 3],
                    'alphashape_time': method_clock_time[method_idx, 4],
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
    
    # Save the figure
    filepath = generate_plot_filename('parameter_heatmaps', 'arc_vs_prop_time_heatmaps')
    save_figure_with_metadata(fig, filepath)
    
    plt.show()

def plot_radec_propagation(all_query_results, methods, method_colors, selected_times=None):
    """Plot RA×DEC alpha shapes at selected time instances with separate figures for each method
    
    Parameters:
    -----------
    all_query_results : dict
        Dictionary containing query results for all arcs
    methods : list
        List of method names to plot
    method_colors : dict
        Dictionary mapping method names to colors
    selected_times : list, optional
        List of time instances to plot. Can be either:
        - List of seconds from start (int/float): [0, 60, 300, 600, ...]
        - List of Time objects: [Time('2029-04-13T21:46:00'), Time('2029-04-13T21:47:00'), ...]
        - List of ISO strings: ['2029-04-13T21:46:00', '2029-04-13T21:47:00', ...]
        Default: [0, 60, 300, 600, 1200, 1800, 3600] (0s, 1min, 5min, 10min, 20min, 30min, 1hr)
    """
    
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
    
    # Use default selected time instances if not provided
    if selected_times is None:
        # Use 52 equally spaced time steps from the total propagation time array
        sample_arc_data, _ = get_arc_data(list(all_query_results.values())[0])
        total_time_steps = len(sample_arc_data['tgrid'])
        
        # Create 52 equally spaced indices across the full time range
        if total_time_steps >= 52:
            time_indices = np.linspace(0, total_time_steps-1, 52, dtype=int)
        else:
            time_indices = np.arange(total_time_steps)  # Use all available if less than 52
        
        # Convert indices to actual Time objects
        selected_times = [sample_arc_data['t_propagation'][i] for i in time_indices]
    
    # Get reference time grid from first arc
    sample_arc_data, _ = get_arc_data(list(all_query_results.values())[0])
    total_time_steps = len(sample_arc_data['tgrid'])
    t_prop_ref = sample_arc_data['t_propagation']
    t_start_ref = t_prop_ref[0]  # Reference start time
    
    print(f"Total time steps available: {total_time_steps}")
    print(f"Reference start time: {t_start_ref.iso}")
    
    # Convert selected_times to delta t values and Time objects
    # NOTE: Propagation goes BACKWARD in time (into the past)
    selected_times_processed = []
    selected_deltat_sec = []
    
    for selected_time in selected_times:
        if isinstance(selected_time, (int, float)):
            # Input is delta t in seconds (positive values go backward in time)
            delta_t_sec = float(selected_time)
            target_time = t_start_ref - TimeDelta(delta_t_sec, format='sec')  # SUBTRACT for backward propagation
        elif isinstance(selected_time, str):
            # Input is ISO string
            target_time = Time(selected_time)
            delta_t_sec = (t_start_ref - target_time).sec  # Positive delta_t = further into past
        elif hasattr(selected_time, 'iso'):
            # Input is Time object
            target_time = selected_time
            delta_t_sec = (t_start_ref - target_time).sec  # Positive delta_t = further into past
        else:
            print(f"Warning: Unrecognized time format: {selected_time}, skipping")
            continue
        
        selected_times_processed.append(target_time)
        selected_deltat_sec.append(delta_t_sec)
    
    print(f"Selected time instances (delta t): {[f'{dt:.0f}s' for dt in selected_deltat_sec]}")
    print(f"Selected time instances (absolute): {[t.iso for t in selected_times_processed]}")
    
    # Load true Apophis data for comparison
    try:
        # Get true Apophis ephemeris for the full time range
        # Note: t_prop_ref is in descending order (backward propagation), so t_prop_ref[-1] is earliest
        t_earliest = t_prop_ref[-1]  # Last element = earliest time (furthest into past)
        t_latest = t_prop_ref[0]     # First element = latest time (closest to observation)
        
        eph_true, _ = post.load_Apophis_Ephemeris(
            t_earliest.iso, t_latest.iso, '10min'  # Higher resolution for better interpolation
        )
        ra_true = eph_true['RA'].to(u.deg).value
        dec_true = eph_true['DEC'].to(u.deg).value
        # Convert datetime strings to Time objects - handle both single strings and arrays
        t_true = Time(eph_true['datetime_jd'], format='jd')
        
    except Exception as e:
        print(f"Could not load true Apophis data: {e}")
        ra_true = dec_true = t_true = None
    
    # Create arc color mapping
    arc_colors = plt.cm.Set1(np.linspace(0, 1, len(arc_info)))
    arc_color_map = {arc_idx: arc_colors[i] for i, arc_idx in enumerate(sorted(arc_info.keys()))}
    
    # Create method title mapping
    method_titles = {
        'DAIOD_ADS_DA': 'DAIOD+ADS Orbit Determination, DA propagation',
        'DAIOD_DA': 'DAIOD Orbit Determination, DA Propagation',
        'DAIOD_MC': 'DAIOD Monte Carlo Orbit Determination, Pointwise Propagation',
        'GAUSS_MC': 'Gauss Monte Carlo Orbit Determination, Pointwise Propagation'
    }
    # method_titles = {
    #     'DAIOD_ADS_ADS': 'DAIOD+ADS Orbit Determination, ADS propagation',
    #     'DAIOD_ADS': 'DAIOD Orbit Determination, ADS propagation',
    #     'DAIOD_DA': 'DAIOD Orbit Determination, DA Propagation',
    #     'DAIOD_MC': 'DAIOD Monte Carlo Orbit Determination, Pointwise Propagation',
    #     'GAUSS_MC': 'Gauss Monte Carlo Orbit Determination, Pointwise Propagation'
    # }  # Old method titles
    
    # Group time instances into sets of 5 for multiple figures
    timesteps_per_figure = 5
    n_figures = (len(selected_times_processed) + timesteps_per_figure - 1) // timesteps_per_figure
    
    # Create separate figure for each method
    for method in methods:
        print(f"Creating {n_figures} figures for method {method}")
        
        # Create multiple figures, each with 5 timesteps
        for fig_idx in range(n_figures):
            start_idx = fig_idx * timesteps_per_figure
            end_idx = min(start_idx + timesteps_per_figure, len(selected_times_processed))
            
            current_times = selected_times_processed[start_idx:end_idx]
            current_deltas = selected_deltat_sec[start_idx:end_idx]
            
            plt.figure(figsize=(14, 10))
            
            # Create time range string for title
            first_time = current_times[0]
            last_time = current_times[-1]
            time_range_str = f"{first_time.iso[:19]} to {last_time.iso[:19]}"
            
            plt.suptitle(f'{method_titles.get(method, method)} - RA×DEC Alpha Shapes\nFigure {fig_idx+1}/{n_figures} | Time range: {time_range_str}', 
                        fontsize=16, y=0.95)
            
            legend_elements = []
            
            try:
                # For each selected time instance in this figure
                for i, (target_time, delta_t_sec) in enumerate(zip(current_times, current_deltas)):
                    
                    # Find true Apophis position at this time
                    true_ra_at_time = None
                    true_dec_at_time = None
                    
                    if ra_true is not None:
                        # Find closest time in true ephemeris
                        time_diffs = np.abs((t_true - target_time).sec)
                        closest_true_idx = np.argmin(time_diffs)
                        
                        true_ra_at_time = ra_true[closest_true_idx]
                        true_dec_at_time = dec_true[closest_true_idx]
                    
                    # Plot alpha shapes for each arc at this time
                    for arc_idx, result_data in all_query_results.items():
                        arc_data, _ = get_arc_data(result_data)
                        dt_days = arc_data['current_dt']
                        t_prop = arc_data['t_propagation']
                        
                        # Find closest time index in this arc's time grid
                        time_diffs = np.abs((t_prop - target_time).sec)
                        closest_time_idx = np.argmin(time_diffs)
                        actual_time_diff = time_diffs[closest_time_idx]
                        
                        # Skip if time difference is too large (more than 30 seconds)
                        if actual_time_diff > 30:
                            continue
                        
                        # Get alpha shapes for this method and time
                        ra_dec_alphashapes = arc_data.get('alphashapes', {}).get(method, [])
                        
                        if closest_time_idx < len(ra_dec_alphashapes):
                            alphashape = ra_dec_alphashapes[closest_time_idx]
                            
                            if alphashape is not None and hasattr(alphashape, 'exterior'):
                                # Extract boundary coordinates
                                boundary_coords = np.array(alphashape.exterior.coords)
                                
                                # Plot alpha shape boundary
                                line_alpha = 0.7 if delta_t_sec == 0 else 0.6  # First time more prominent
                                line_width = 2.0 if delta_t_sec == 0 else 1.5
                                
                                plt.plot(boundary_coords[:, 0], boundary_coords[:, 1], 
                                       color=arc_color_map[arc_idx], linewidth=line_width, 
                                       alpha=line_alpha, linestyle='-')
                        
                        print(f"Plotted arc {arc_idx} at Δt={delta_t_sec:.0f}s (closest match: {actual_time_diff:.1f}s)")
                    
                    # Plot true Apophis position at this time instance with smaller marker
                    if true_ra_at_time is not None and true_dec_at_time is not None:
                        marker_size = 30 if delta_t_sec == 0 else 20  # Reduced from 100/60 to 30/20
                        plt.scatter(true_ra_at_time, true_dec_at_time, 
                                 c='black', marker='x', s=marker_size, linewidth=1.5,
                                 zorder=10, alpha=0.9)
                        
                        # Add delta t annotation at true Apophis position
                        if abs(delta_t_sec) < 60:
                            time_label = f'{delta_t_sec:.0f}s'
                        elif abs(delta_t_sec) < 3600:
                            time_label = f'{delta_t_sec/60:.1f}min'
                        else:
                            time_label = f'{delta_t_sec/3600:.1f}hr'
                        
                        # Offset annotation slightly to avoid overlap with marker
                        plt.annotate(time_label, (true_ra_at_time, true_dec_at_time), 
                                   xytext=(8, 8), textcoords='offset points',
                                   fontsize=10, color='black', weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                   zorder=11)
                
                # Create legend for arc lengths (only once per figure)
                for arc_idx in sorted(arc_info.keys()):
                    dt_days = arc_info[arc_idx]['dt_days']
                    legend_elements.append(plt.Line2D([0], [0], color=arc_color_map[arc_idx], 
                                                     linewidth=2, alpha=0.7, 
                                                     label=f'Arc {arc_idx}: {dt_days:.2f}d'))
                
                # Add true Apophis to legend
                legend_elements.append(plt.Line2D([0], [0], marker='x', color='black', 
                                                linewidth=0, markersize=6, markeredgewidth=1.5,
                                                label='True Apophis'))
                
                plt.xlabel('RA [deg]')
                plt.ylabel('DEC [deg]')
                plt.grid(True, alpha=0.3)
                plt.legend(handles=legend_elements, loc='upper right')
                
                # Set fixed axis limits for full celestial coordinate range
                
                # Set equal aspect ratio for proper sky projection
                plt.gca().set_aspect('equal', adjustable='box')
                    
            except Exception as e:
                print(f"Error plotting {method} figure {fig_idx+1}: {e}")
                plt.text(0.5, 0.5, f'Error plotting {method}:\n{str(e)}', 
                       transform=plt.gca().transAxes, ha='center', va='center')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            
            # Save the figure
            first_time = current_times[0].iso[:19].replace(':', '-')
            last_time = current_times[-1].iso[:19].replace(':', '-')
            filename = f"radec_propagation_{method}_fig{fig_idx+1}_{first_time}_to_{last_time}"
            filepath = generate_plot_filename('radec_propagation', filename, timestamp=False)
            save_figure_with_metadata(plt.gcf(), filepath)
            
            plt.close()
        
        print(f"Completed {n_figures} figures for method {method}")

def plot_alphashape_area(all_query_results, methods, method_colors):
    """Plot alphashape area evolution over propagation time for each method
    
    Creates separate figures for each arc length, showing how the uncertainty
    region area changes over time for different orbital determination methods.
    Uses pre-computed alphashape areas stored in arc_data.
    """
    
    # Method title mapping
    method_titles = {
        'DAIOD_ADS_DA': 'DAIOD+ADS Orbit Determination, DA propagation',
        'DAIOD_DA': 'DAIOD Orbit Determination, DA Propagation',
        'DAIOD_MC': 'DAIOD Monte Carlo Orbit Determination, Pointwise Propagation',
        'GAUSS_MC': 'Gauss Monte Carlo Orbit Determination, Pointwise Propagation'
    }
    # method_titles = {
    #     'DAIOD_ADS_ADS': 'DAIOD+ADS Orbit Determination, ADS propagation',
    #     'DAIOD_ADS': 'DAIOD Orbit Determination, ADS propagation',
    #     'DAIOD_DA': 'DAIOD Orbit Determination, DA Propagation',
    #     'DAIOD_MC': 'DAIOD Monte Carlo Orbit Determination, Pointwise Propagation',
    #     'GAUSS_MC': 'Gauss Monte Carlo Orbit Determination, Pointwise Propagation'
    # }  # Old method titles
    
    # Method marker shapes
    method_markers = {
        'DAIOD_ADS_DA': 'o',       # Circle
        'DAIOD_DA': '^',           # Triangle up
        'DAIOD_MC': 'D',           # Diamond
        'GAUSS_MC': 'v'            # Triangle down
    }
    # method_markers = {
    #     'DAIOD_ADS_ADS': 'o',      # Circle
    #     'DAIOD_ADS': 's',          # Square
    #     'DAIOD_DA': '^',           # Triangle up
    #     'DAIOD_MC': 'D',           # Diamond
    #     'GAUSS_MC': 'v'            # Triangle down
    # }  # Old method markers
    
    # Method line styles - each method has unique line pattern
    method_linestyles = {
        'DAIOD_ADS_DA': '--',      # Dashed
        'DAIOD_DA': ':',           # Dotted
        'DAIOD_MC': '-',           # Solid
        'GAUSS_MC': '-.'           # Dash-dot
    }
    # method_linestyles = {
    #     'DAIOD_ADS_ADS': '--',     # Dashed
    #     'DAIOD_ADS': '-.',         # Dash-dot
    #     'DAIOD_DA': ':',           # Dotted
    #     'DAIOD_MC': '--',          # Dashed
    #     'GAUSS_MC': '-.'           # Dash-dot
    # }  # Old method line styles
    
    # Group arcs by arc length for separate figures
    arc_groups = {}
    for arc_idx, result_data in all_query_results.items():
        arc_data, _ = get_arc_data(result_data)
        dt_days = arc_data['current_dt']
        
        # Round to avoid floating point precision issues
        dt_key = round(dt_days, 3)
        if dt_key not in arc_groups:
            arc_groups[dt_key] = []
        arc_groups[dt_key].append(arc_idx)
    
    # Create separate figure for each arc length
    for dt_days, arc_indices in arc_groups.items():
        plt.figure(figsize=(12, 8))
        plt.title(f'Uncertainty Region Area Evolution - Arc Length: {dt_days:.3f} days', 
                 fontsize=14, pad=20)
        
        # Debug: Check what methods are available in the first arc
        first_arc_data, _ = get_arc_data(all_query_results[arc_indices[0]])
        if 'alphashape_areas' in first_arc_data:
            available_methods = list(first_arc_data['alphashape_areas'].keys())
            print(f"Available methods in alphashape_areas: {available_methods}")
            print(f"Expected methods: {methods}")
        else:
            print("No alphashape_areas found in first arc")
        
        # Process each method
        for method_idx, method in enumerate(methods):
            method_areas = []
            method_times = []
            
            print(f"Processing method: {method}")
            
            # Collect data from all arcs with this arc length
            for arc_idx in arc_indices:
                result_data = all_query_results[arc_idx]
                arc_data, _ = get_arc_data(result_data)
                
                # Get propagation times (convert to days for x-axis)
                t_prop = arc_data['t_propagation']
                prop_times_days = (t_prop - t_prop[0]).to(u.day).value
                
                try:
                    # Extract pre-computed alphashape areas from arc_data
                    if 'alphashape_areas' in arc_data:
                        alphashape_areas = arc_data['alphashape_areas']
                        
                        # Extract areas for this method using method name as key
                        if method in alphashape_areas:
                            time_areas = alphashape_areas[method]
                            
                            # Add the areas and corresponding times
                            method_areas.extend(time_areas)
                            method_times.extend(prop_times_days[:len(time_areas)])
                        else:
                            print(f"Warning: Method '{method}' not found in alphashape_areas for arc {arc_idx}")
                            print(f"Available methods: {list(alphashape_areas.keys())}")
                            continue
                    else:
                        print(f"Warning: alphashape_areas not found in arc_data for arc {arc_idx}")
                        continue
                
                except Exception as e:
                    print(f"Error processing {method} for arc {arc_idx}: {e}")
                    continue
            
            # Plot the method data if we have any
            print(f"Method {method}: Found {len(method_areas)} area values")
            if method_areas and method_times:
                # Convert to numpy arrays and sort by time
                times = np.array(method_times)
                areas = np.array(method_areas)
                
                # Remove NaN values
                valid_mask = ~np.isnan(areas)
                times = times[valid_mask]
                areas = areas[valid_mask]
                print(f"Method {method}: After removing NaN, {len(areas)} valid values")
                
                if len(times) > 0:
                    # Sort by time
                    sort_idx = np.argsort(times)
                    times = times[sort_idx]
                    areas = areas[sort_idx]
                    
                    # Convert times to absolute values since we're going backward in time
                    times = np.abs(times)
                    
                    # Plot as lines with method-specific colors and line styles
                    plt.plot(times, areas, 
                             color=method_colors[method],
                             linestyle=method_linestyles.get(method, '-'),
                             linewidth=2.5, 
                             alpha=0.8,
                             label=method_titles.get(method, method))
            print(f"Finished processing method: {method}")
            
        
        plt.xlabel('Propagation Time [days]')
        plt.ylabel('Area of Uncertainty Region [deg²]')
        plt.axhline(41253/2, color='gray', linestyle='--', label='Half Celestial Sphere Area (20626.5 deg²)')
        plt.axhline(78.54, color='gray', linestyle=':', label='Sky Survey FOV (Calar Alto Observatory)') # https://link.springer.com/chapter/10.1007/978-94-011-1146-1_11
        plt.axhline(0.196, color='gray', linestyle='-.', label='Typical FOV (ESO-NTT)') # Example: LSST ~9.6 deg², HST ~0.196 deg²
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')  # Use log scale for area as it can vary widely
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"alphashape_area_evolution_arc_{dt_days:.3f}d"
        filepath = generate_plot_filename('alphashape_area', filename, timestamp=False)
        save_figure_with_metadata(plt.gcf(), filepath)
        
        plt.show()

def plot_range_rangerate_phase_space(all_query_results, methods, method_colors):
    """Plot Range×Range-rate phase space - separate figures for each arc length and timestep"""
    
    # Group arcs by arc length
    arc_groups = {}
    for arc_idx, result_data in all_query_results.items():
        arc_data, _ = get_arc_data(result_data)
        dt_days = arc_data['current_dt']
        
        # Round to avoid floating point precision issues
        dt_key = round(dt_days, 3)
        if dt_key not in arc_groups:
            arc_groups[dt_key] = []
        arc_groups[dt_key].append(arc_idx)
    
    print(f"Found {len(arc_groups)} arc length groups: {sorted(arc_groups.keys())}")
    
    # Get the total time steps from first arc
    arc_data, _ = get_arc_data(list(all_query_results.values())[0])
    total_time_steps = len(arc_data['tgrid'])
    t_prop = arc_data['t_propagation']
    
    # Select 6 equally spaced time steps
    if total_time_steps >= 6:
        selected_time_indices = np.linspace(0, total_time_steps-1, 6, dtype=int)
    else:
        selected_time_indices = np.arange(total_time_steps)  # Use all available if less than 6

    n_time_steps = len(selected_time_indices)
    
    # Load true Apophis range data for comparison (will be used for all arc lengths)
    try:
        # Get true Apophis ephemeris for the full time range
        # Note: t_prop is in descending order (backward propagation), so t_prop[-1] is earliest
        t_earliest = t_prop[-1]  # Last element = earliest time (furthest into past)
        t_latest = t_prop[0]     # First element = latest time (epoch/observation time)
        
        print(f"Loading true Apophis data from {t_earliest.iso} to {t_latest.iso}")
        
        eph_true, vec_true = post.load_Apophis_Ephemeris(
            t_earliest.iso, t_latest.iso, '1h', location = '500'  # Higher resolution for better interpolation
        )
        
        # Convert to range and range-rate at selected time steps
        apophis_true_obs = []
        for i, time_idx in enumerate(selected_time_indices):
            target_time = t_prop[time_idx]
            
            # Find closest time in true ephemeris
            t_true = Time(eph_true['datetime_jd'], format='jd')
            time_diffs = np.abs((t_true - target_time).sec)
            closest_true_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_true_idx] > 3600:  # More than 1 hour difference
                print(f"Warning: Large time difference ({time_diffs[closest_true_idx]:.0f}s) for time step {time_idx}")
            
            # Calculate range and range-rate from geocentric position and velocity
            # Assuming vec_true contains geocentric Cartesian coordinates
          
                # Position vector (geocentric)
            x =(vec_true['x'][closest_true_idx]* u.AU).to(u.km).value
            y = (vec_true['y'][closest_true_idx] * u.AU).to(u.km).value
            z = (vec_true['z'][closest_true_idx] * u.AU).to(u.km).value

                # Velocity vector (geocentric)
            vx = (vec_true['vx'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value
            vy = (vec_true['vy'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value
            vz = (vec_true['vz'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value

                # Calculate range (distance from Earth center)
            true_range = (vec_true['range'][closest_true_idx] * u.AU).to(u.km).value  # In km
                
                # Calculate range-rate (radial velocity)
                # range_rate = (r · v) / |r|
            position_vec = np.array([x, y, z])
            velocity_vec = np.array([vx, vy, vz])
            true_range_rate = (vec_true['range_rate'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value  # In km/s
                
            
            apophis_true_obs.append((true_range, true_range_rate))
        
    except Exception as e:
        print(f"Could not load true Apophis range data: {e}")
        apophis_true_obs = None

    # Create separate figures for each arc length group
    for dt_days, arc_indices in arc_groups.items():
        print(f"\nCreating figures for arc length: {dt_days:.3f} days (arcs: {arc_indices})")
        
        # Create separate figure for each timestep within this arc length group
        for i, time_idx in enumerate(selected_time_indices):
            # Create new figure for this timestep and arc length
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Get target time for this timestep
            target_time = t_prop[time_idx]
            time_since_start = (t_prop[0] - target_time).to(u.day).value  # Days into past
            
            print(f"  Creating figure {i+1}/{n_time_steps} for timestep {time_idx} (t = {time_since_start:.2f} days)")
            
            # Plot each method at this timestep for this arc length group
            for method_idx, method in enumerate(methods):
                method_ranges = []
                method_range_rates = []
                
                # Only process arcs in this arc length group
                for arc_idx in arc_indices:
                    result_data = all_query_results[arc_idx]
                    arc_data, _ = get_arc_data(result_data)
                
                try:
                    ranges = []
                    range_rates = []
                    
                    # Extract observational data based on method type
                    if 'ADS' in method:
                        # For ADS methods, extract from perimeter evaluation
                        if method == 'DAIOD_ADS_DA':
                            perimeter_data = arc_data['DAIOD_ADS_DA_perimeter']
                        # else:
                        #     perimeter_data = arc_data['DAIOD_perimeter']  # Old DAIOD_ADS method

                        if time_idx < len(perimeter_data[0]):
                            manifold = perimeter_data[time_idx]
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
                    
                    # Collect data for this method
                    if len(ranges) > 0 and len(range_rates) > 0:
                        method_ranges.extend(ranges)
                        method_range_rates.extend(range_rates)
                    
                except Exception as e:
                    print(f"Error extracting range data for {method} at time {time_idx}: {e}")
                    continue
            
                # Plot data for this method
                if len(method_ranges) > 0 and len(method_range_rates) > 0:
                    ax.scatter(method_ranges, method_range_rates, c=method_colors[method], 
                            marker='o', alpha=0.6, s=20, 
                            edgecolors='black', linewidth=0.3,
                            label=f'{method}')
                    print(f"  Plotted {len(method_ranges)} points for {method}")
        
                # Plot true Apophis at this timestep if available
                if apophis_true_obs and i < len(apophis_true_obs):
                    true_range, true_range_rate = apophis_true_obs[i]
                    if true_range is not None and true_range_rate is not None:
                        ax.scatter(true_range, true_range_rate, c='black', marker='*', 
                                s=200, zorder=10, 
                                edgecolors='white', linewidth=1)
            
                ax.set_xlabel('Range [km]')
                ax.set_ylabel('Range Rate [km/s]')
                ax.set_title(f'Range×Range-rate Phase Space - All Methods\nArc Length: {dt_days:.3f} days | Time: {target_time.iso} ({time_since_start:.2f} days)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            plt.tight_layout()
            
            # Save the figure
            time_str = target_time.iso[:19].replace(':', '-')
            filename = f"range_rangerate_all_methods_arc_{dt_days:.3f}d_time{i+1}_{time_str}"
            filepath = generate_plot_filename('range_rangerate/all_methods', filename, timestamp=False)
            save_figure_with_metadata(fig, filepath)
            
            plt.show()
    
    total_figures = len(arc_groups) * n_time_steps
    print(f"Completed {total_figures} range×range-rate plots ({len(arc_groups)} arc lengths × {n_time_steps} timesteps)")
    return None  # Multiple figures, no single figure to return

def plot_range_rangerate_no_daiod_da(all_query_results, methods, method_colors):
    """Plot Range×Range-rate phase space without DAIOD_DA method - separate figures for each arc length and timestep"""
    
    # Filter out DAIOD_DA from methods
    filtered_methods = [method for method in methods if method != 'DAIOD_DA']
    
    if not filtered_methods:
        print("No methods available after filtering out DAIOD_DA")
        return
    
    # Group arcs by arc length
    arc_groups = {}
    for arc_idx, result_data in all_query_results.items():
        arc_data, _ = get_arc_data(result_data)
        dt_days = arc_data['current_dt']
        
        # Round to avoid floating point precision issues
        dt_key = round(dt_days, 3)
        if dt_key not in arc_groups:
            arc_groups[dt_key] = []
        arc_groups[dt_key].append(arc_idx)
    
    print(f"Found {len(arc_groups)} arc length groups: {sorted(arc_groups.keys())}")
    
    # Get the total time steps from first arc
    arc_data, _ = get_arc_data(list(all_query_results.values())[0])
    total_time_steps = len(arc_data['tgrid'])
    t_prop = arc_data['t_propagation']
    
    # Select 6 equally spaced time steps (same as main function)
    if total_time_steps >= 6:
        selected_time_indices = np.linspace(0, total_time_steps-1, 6, dtype=int)
    else:
        selected_time_indices = np.arange(total_time_steps)  # Use all available if less than 6

    n_time_steps = len(selected_time_indices)
    
    # Load true Apophis range data for comparison (will be used for all arc lengths)
    try:
        # Get true Apophis ephemeris for the full time range
        # Note: t_prop is in descending order (backward propagation), so t_prop[-1] is earliest
        t_earliest = t_prop[-1]  # Last element = earliest time (furthest into past)
        t_latest = t_prop[0]     # First element = latest time (epoch/observation time)
        
        print(f"Loading true Apophis data from {t_earliest.iso} to {t_latest.iso}")
        
        eph_true, vec_true = post.load_Apophis_Ephemeris(
            t_earliest.iso, t_latest.iso, '1h', location = '500'  # Higher resolution for better interpolation
        )
        
        # Convert to range and range-rate at selected time steps
        apophis_true_obs = []
        for i, time_idx in enumerate(selected_time_indices):
            target_time = t_prop[time_idx]
            
            # Find closest time in true ephemeris
            t_true = Time(eph_true['datetime_jd'], format='jd')
            time_diffs = np.abs((t_true - target_time).sec)
            closest_true_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_true_idx] > 3600:  # More than 1 hour difference
                print(f"Warning: Large time difference ({time_diffs[closest_true_idx]:.0f}s) for time step {time_idx}")
            
            # Calculate range and range-rate from geocentric position and velocity
            # Assuming vec_true contains geocentric Cartesian coordinates
          
                # Position vector (geocentric)
            x =(vec_true['x'][closest_true_idx]* u.AU).to(u.km).value
            y = (vec_true['y'][closest_true_idx] * u.AU).to(u.km).value
            z = (vec_true['z'][closest_true_idx] * u.AU).to(u.km).value

                # Velocity vector (geocentric)
            vx = (vec_true['vx'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value
            vy = (vec_true['vy'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value
            vz = (vec_true['vz'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value

                # Calculate range (distance from Earth center)
            true_range = (vec_true['range'][closest_true_idx] * u.AU).to(u.km).value  # In km
                
                # Calculate range-rate (radial velocity)
                # range_rate = (r · v) / |r|
            position_vec = np.array([x, y, z])
            velocity_vec = np.array([vx, vy, vz])
            true_range_rate = (vec_true['range_rate'][closest_true_idx] * u.AU/u.day).to(u.km/u.s).value  # In km/s
                
            
            apophis_true_obs.append((true_range, true_range_rate))
        
    except Exception as e:
        print(f"Could not load true Apophis range data: {e}")
        apophis_true_obs = None
    
    # Create separate figures for each arc length group (excluding DAIOD_DA)
    for dt_days, arc_indices in arc_groups.items():
        print(f"\nCreating figures for arc length: {dt_days:.3f} days (arcs: {arc_indices}) - Without DAIOD_DA")
        
        # Create separate figure for each timestep within this arc length group
        for i, time_idx in enumerate(selected_time_indices):
            # Create new figure for this timestep and arc length
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Get target time for this timestep
            target_time = t_prop[time_idx]
            time_since_start = (t_prop[0] - target_time).to(u.day).value  # Days into past
            
            print(f"  Creating figure {i+1}/{n_time_steps} for timestep {time_idx} (t = {time_since_start:.2f} days)")
            
            # Plot each method at this timestep for this arc length group (excluding DAIOD_DA)
            for method_idx, method in enumerate(filtered_methods):
                method_ranges = []
                method_range_rates = []
                
                # Only process arcs in this arc length group
                for arc_idx in arc_indices:
                    result_data = all_query_results[arc_idx]
                    arc_data, _ = get_arc_data(result_data)
                
                try:
                    ranges = []
                    range_rates = []
                    
                    # Extract observational data based on method type
                    if 'ADS' in method:
                        # For ADS methods, extract from perimeter evaluation
                        if method == 'DAIOD_ADS_DA':
                            perimeter_data = arc_data['DAIOD_ADS_DA_perimeter']
                        # else:
                        #     perimeter_data = arc_data['DAIOD_perimeter']  # Old DAIOD_ADS method

                        if time_idx < len(perimeter_data[0]):
                            manifold = perimeter_data[time_idx]
                            # Extract range (element 2) and range-rate (element 5)
                            ranges = manifold[:, 2, :].flatten()  # All ranges
                            range_rates = manifold[:, 5, :].flatten()  # All range-rates
                            
                            # Remove invalid points
                            valid_mask = np.isfinite(ranges) & np.isfinite(range_rates)
                            ranges = ranges[valid_mask]
                            range_rates = range_rates[valid_mask]
                            
                    elif method == 'DAIOD_DA':
                        # For DA method, extract from DA perimeter (this won't be executed since DAIOD_DA is filtered out)
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
                    
                    # Collect data for this method
                    if len(ranges) > 0 and len(range_rates) > 0:
                        method_ranges.extend(ranges)
                        method_range_rates.extend(range_rates)
                    
                except Exception as e:
                    print(f"Error extracting range data for {method} at time {time_idx}: {e}")
                    continue
            
                # Plot data for this method
                if len(method_ranges) > 0 and len(method_range_rates) > 0:
                    ax.scatter(method_ranges, method_range_rates, c=method_colors[method], 
                            marker='o', alpha=0.6, s=20, 
                            edgecolors='black', linewidth=0.3,
                            label=f'{method}')
                    print(f"  Plotted {len(method_ranges)} points for {method}")
        
                # Plot true Apophis at this timestep if available
                if apophis_true_obs and i < len(apophis_true_obs):
                    true_range, true_range_rate = apophis_true_obs[i]
                    if true_range is not None and true_range_rate is not None:
                        ax.scatter(true_range, true_range_rate, c='black', marker='*', 
                                s=200, zorder=10, 
                                edgecolors='white', linewidth=1)
            
                ax.set_ylabel('Range Rate [km/s]')
                ax.set_title(f'Range×Range-rate Phase Space - Without DAIOD_DA\nArc Length: {dt_days:.3f} days | Time: {target_time.iso} ({time_since_start:.2f} days)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            
            # Save the figure
            time_str = target_time.iso[:19].replace(':', '-')
            filename = f"range_rangerate_no_daiod_da_arc_{dt_days:.3f}d_time{i+1}_{time_str}"
            filepath = generate_plot_filename('range_rangerate/no_daiod_da', filename, timestamp=False)
            save_figure_with_metadata(fig, filepath)
            
            plt.show()
    
    total_figures = len(arc_groups) * n_time_steps
    print(f"Completed {total_figures} range×range-rate plots without DAIOD_DA ({len(arc_groups)} arc lengths × {n_time_steps} timesteps)")
    return None  # Multiple figures, no single figure to return

def plot_ads_split_history(all_query_results):
    """Plot ADS domain split history over propagation time"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ads_methods = ['DAIOD_ADS_DA']  # Updated to match stored structure
    # ads_methods = ['DAIOD_ADS_ADS', 'DAIOD_ADS']  # Old ADS methods
    colors = ['red']
    
    # Method marker shapes (matching alphashape area plots)
    method_markers = {
        'DAIOD_ADS_DA': 'o',       # Circle
    }
    # method_markers = {
    #     'DAIOD_ADS_ADS': 'o',      # Circle
    #     'DAIOD_ADS': 's',          # Square
    # }  # Old method markers
    
    # Method line styles (matching alphashape area plots)
    method_linestyles = {
        'DAIOD_ADS_DA': '--',      # Dashed
    }
    # method_linestyles = {
    #     'DAIOD_ADS_ADS': '--',     # Dashed
    #     'DAIOD_ADS': '-.',         # Dash-dot
    # }  # Old method line styles
    
    for arc_idx, result_data in all_query_results.items():
        arc_data, _ = get_arc_data(result_data)
        tgrid = arc_data['tgrid']
        prop_times_days = (arc_data['t_propagation'] - arc_data['t_propagation'][0]).to(u.day).value
        
        # Convert to absolute delta t (0 at left, increasing to right)
        abs_delta_t_days = np.abs(prop_times_days)
        
        for method_idx, method in enumerate(ads_methods):
            try:
                if method == 'DAIOD_ADS_DA':
                    final_lists = arc_data['final_lists_ADS']
                # else:
                #     final_lists = arc_data['final_lists_ADS_DAIOD']  # Old DAIOD_ADS method
                
                nsplits = [len(domains) for domains in final_lists]
                
                ax.plot(abs_delta_t_days[:len(nsplits)], nsplits, 
                       color=colors[method_idx], 
                       marker=method_markers[method],
                       linestyle=method_linestyles[method],
                       label=f'{method} (Arc {arc_idx})', linewidth=2)
                
            except Exception as e:
                print(f"Error plotting split history for {method}: {e}")
    
    ax.set_xlabel('Propagation Time [days]')
    ax.set_ylabel('Number of ADS Domains')
    ax.set_title('ADS Domain Split History')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save the figure
    filepath = generate_plot_filename('ads_split_history', 'ads_domain_split_history', timestamp=False)
    save_figure_with_metadata(fig, filepath)
    
    plt.show()

def plot_wall_clock_comparison(df, methods):
    """Plot wall clock time comparison as bar chart"""
    
    # Aggregate data by method
    method_times = df.groupby('method').agg({
        'od_time': 'mean',
        'prop_time': 'mean', 
        'eval_time': 'mean',
        'conversion_time': 'mean',
        'alphashape_time': 'mean',
        'query_time': 'mean'
    }).reset_index()
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.6
    
    bottom = np.zeros(len(methods))
    colors = ['#1f77b4', "#ff7700", '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    labels = ['Orbit Determination', 'Propagation', 'Evaluation', 'Query', 'Conversion', 'AlphaShape']

    for i, (col, color, label) in enumerate(zip(['od_time', 'prop_time', 'eval_time', 'query_time', 'conversion_time', 'alphashape_time'], 
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
    
    # Save the first figure
    filepath1 = generate_plot_filename('wall_clock_time', 'computation_time_breakdown_with_alphashape', timestamp=False)
    save_figure_with_metadata(fig, filepath1)
    
    plt.show()
    
    # Create second bar chart excluding alphashape_time
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    bottom2 = np.zeros(len(methods))
    colors2 = ['#1f77b4', "#ff7700", '#2ca02c', '#d62728']
    labels2 = ['Orbit Determination', 'Propagation', 'Conversion', 'Query']

    for i, (col, color, label) in enumerate(zip(['od_time', 'prop_time', 'conversion_time', 'query_time'], 
                                               colors2, labels2)):
        values = [method_times[method_times['method'] == method][col].iloc[0] 
                 if len(method_times[method_times['method'] == method]) > 0 else 0 
                 for method in methods]
        
        ax2.bar(x, values, width, bottom=bottom2, label=label, color=color)
        bottom2 += values
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Wall Clock Time [s]')
    ax2.set_title('Computation Time Breakdown by Method (Excluding AlphaShape)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the second figure
    filepath2 = generate_plot_filename('wall_clock_time', 'computation_time_breakdown_no_alphashape', timestamp=False)
    save_figure_with_metadata(fig2, filepath2)
    
    plt.show()

def plot_precision_vs_walltime_scatter(df, method_colors, method_markers):
    """Plot True Positive Rate vs wall clock time scatter with multi-dimensional encoding"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique values for mapping
    arc_lengths = sorted(df['dt_days'].unique())
    prop_horizons = sorted(df['prop_time_days'].unique(), key=abs)  # Sort by absolute value
    methods = sorted(df['method'].unique())
    
    print(f"Arc lengths found: {arc_lengths}")
    print(f"Propagation horizons found: {[f'{h:.2f}' for h in prop_horizons]}")
    print(f"Methods found: {methods}")
    
    # Calculate wall time excluding query time (OD + Prop + Eval only)
    df_plot = df.copy()
    df_plot['wall_time_no_query'] = df_plot['od_time'] + df_plot['prop_time'] + df_plot['eval_time']
    df_plot['abs_prop_time_days'] = np.abs(df_plot['prop_time_days'])
    
    # Create size mapping for arc lengths
    min_size, max_size = 20, 120
    if len(arc_lengths) == 1:
        arc_size_map = {arc_lengths[0]: (min_size + max_size) / 2}
    else:
        min_arc, max_arc = min(arc_lengths), max(arc_lengths)
        arc_size_map = {}
        for arc_length in arc_lengths:
            if max_arc == min_arc:
                arc_size_map[arc_length] = (min_size + max_size) / 2
            else:
                normalized = (arc_length - min_arc) / (max_arc - min_arc)
                arc_size_map[arc_length] = min_size + (max_size - min_size) * normalized
    
    # Plot individual points with method-specific colors and markers
    for _, row in df_plot.iterrows():
        ax.scatter(row['wall_time_no_query'], row['precision'],
                  c=method_colors[row['method']], 
                  marker=method_markers[row['method']], 
                  s=arc_size_map[row['dt_days']], 
                  alpha=0.7, 
                  edgecolors='black', linewidth=0.5)
    
    # Create legend elements
    legend_elements = []
    
    # Method legend (shapes and colors)
    legend_elements.append(plt.scatter([], [], c='white', marker='o', s=0, 
                                     label='Methods:', alpha=0))
    for method in methods:
        legend_elements.append(plt.scatter([], [], c=method_colors[method], 
                                         marker=method_markers[method], s=60, 
                                         alpha=0.7, edgecolors='black', linewidth=0.5,
                                         label=method))
    
    # Arc length legend (sizes) - show representative sizes
    legend_elements.append(plt.scatter([], [], c='white', marker='o', s=0, 
                                     label='Arc Lengths (Sizes):', alpha=0))
    # Show min, mid, and max arc length sizes
    sample_arc_lengths = [arc_lengths[0], arc_lengths[len(arc_lengths)//2], arc_lengths[-1]] if len(arc_lengths) >= 3 else arc_lengths
    for arc_length in sample_arc_lengths:
        legend_elements.append(plt.scatter([], [], c='gray', marker='o', 
                                         s=arc_size_map[arc_length], alpha=0.7,
                                         edgecolors='black', linewidth=0.5,
                                         label=f'{arc_length:.3f} days'))
    
    ax.set_xlabel('Wall Clock Time [s] (excluding Query)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Computational Cost\n(Shape/Color=Method, Size=Arc Length)')
    ax.grid(True, alpha=0.3)
    
    # Place legend outside plot
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the figure
    filepath = generate_plot_filename('precision_scatter', 'precision_vs_walltime_scatter', timestamp=False)
    save_figure_with_metadata(fig, filepath)
    
    plt.show()

def plot_cumulative_walltime(df, methods, method_colors):
    """Plot wall clock time segments vs arc length - separate figure for each time segment"""
    
    # Method line styles (matching alphashape area plots)
    method_linestyles = {
        'DAIOD_ADS_DA': '--',      # Dashed
        'DAIOD_DA': ':',           # Dotted
        'DAIOD_MC': '-',           # Solid
        'GAUSS_MC': '-.'           # Dash-dot
    }
    
    # Method titles for legend
    method_titles = {
        'DAIOD_ADS_DA': 'DAIOD+ADS OD, DA propagation',
        'DAIOD_DA': 'DAIOD OD, DA propagation',
        'DAIOD_MC': 'DAIOD Monte Carlo OD, Pointwise Propagation',
        'GAUSS_MC': 'Gauss Monte Carlo OD, Pointwise Propagation'
    }
    
    # Define time segments to plot
    time_segments = {
        'od_time': 'Orbit Determination Time',
        'prop_time': 'Propagation Time', 
        'eval_time': 'Evaluation Time',
        'conversion_time': 'Conversion Time',
        'alphashape_time': 'AlphaShape Time'
    }
    
    # Get unique arc lengths and sort them
    arc_lengths = sorted(df['dt_days'].unique())
    
    # Create separate figure for each time segment
    for time_col, time_label in time_segments.items():
        plt.figure(figsize=(12, 8))
        plt.title(f'{time_label} vs Arc Length', fontsize=14, pad=20)
        
        # Plot each method as a line
        for method in methods:
            method_data = df[df['method'] == method]
            
            # Calculate mean time for each arc length
            arc_times = []
            for arc_length in arc_lengths:
                arc_method_data = method_data[method_data['dt_days'] == arc_length]
                if len(arc_method_data) > 0:
                    mean_time = arc_method_data[time_col].mean()
                    arc_times.append(mean_time)
                else:
                    arc_times.append(0)
            
            # Plot the line for this method
            plt.plot(arc_lengths, arc_times,
                   color=method_colors[method], 
                   linestyle=method_linestyles.get(method, '-'),
                   label=method_titles.get(method, method), 
                   linewidth=2.5, alpha=0.8,
                   marker='o', markersize=6)
        
        plt.xlabel('Arc Length [days]')
        plt.ylabel('Wall Clock Time [s]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"cumulative_walltime_{time_col}_{time_label.replace(' ', '_').replace('[', '_').replace(']', '')}"
        filepath = generate_plot_filename('cumulative_walltime', filename, timestamp=False)
        save_figure_with_metadata(plt.gcf(), filepath)
        
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
                    if method == 'DAIOD_ADS_DA':
                        perimeter_data = arc_data['DAIOD_ADS_perimeter']
                    # else:
                    #     perimeter_data = arc_data['DAIOD_perimeter']  # Old DAIOD_ADS method
                        
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
    
    # Save the figure
    filepath = generate_plot_filename('orbital_motion', 'heliocentric_perifocal_motion_validation', timestamp=False)
    save_figure_with_metadata(fig, filepath)
    
    plt.show()

def fit_uncertainty_growth_models(x, y):
    """
    Fit models specifically designed for uncertainty growth patterns
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    
    models = {}
    
    # 1. Linear growth: area = a + b*t
    try:
        def linear_model(t, a, b):
            return a + b * t
        popt, _ = curve_fit(linear_model, x, y)
        models['linear_growth'] = (linear_model, popt, r2_score(y, linear_model(x, *popt)))
    except:
        pass
    
    # 2. Quadratic growth: area = a + b*t + c*t²
    try:
        def quadratic_model(t, a, b, c):
            return a + b * t + c * t**2
        popt, _ = curve_fit(quadratic_model, x, y)
        models['quadratic_growth'] = (quadratic_model, popt, r2_score(y, quadratic_model(x, *popt)))
    except:
        pass
    
    # 3. Exponential growth: area = a * exp(b*t)
    try:
        def exponential_model(t, a, b):
            return a * np.exp(b * t)
        # Use log-linear fit as initial guess
        log_y = np.log(np.abs(y) + 1e-10)
        coeffs = np.polyfit(x, log_y, 1)
        p0 = [np.exp(coeffs[1]), coeffs[0]]
        popt, _ = curve_fit(exponential_model, x, y, p0=p0)
        models['exponential_growth'] = (exponential_model, popt, r2_score(y, exponential_model(x, *popt)))
    except:
        pass
    
    # 4. Power law growth: area = a * t^b
    try:
        def power_model(t, a, b):
            return a * (t + 1e-6)**b  # Add small offset to avoid t=0
        # Use log-log fit as initial guess
        log_x = np.log(x + 1e-6)
        log_y = np.log(np.abs(y) + 1e-10)
        coeffs = np.polyfit(log_x, log_y, 1)
        p0 = [np.exp(coeffs[1]), coeffs[0]]
        popt, _ = curve_fit(power_model, x, y, p0=p0)
        models['power_growth'] = (power_model, popt, r2_score(y, power_model(x, *popt)))
    except:
        pass
    
    # Select best model
    if models:
        best_name = max(models.keys(), key=lambda k: models[k][2])
        return models[best_name], best_name, models
    else:
        return None, None, {}

def main():
    """Main plotting function"""
    import time
    print("=== Apophis Results Plotting ===")
    
    # Load query results
    all_query_results = load_query_results()
    if all_query_results is None:
        return
    
    # Extract data for plotting
    df, methods, method_colors, method_markers = extract_data_for_plotting(all_query_results)
    print(f"Loaded data: {len(df)} records across {len(methods)} methods")
    
    # Generate all plots
    #print("\n1. Generating parameter space heatmaps...")
    #plot_parameter_heatmaps(df, methods)
    
    time_ref = time.time()
    print("2. Generating RA×DEC propagation plots (52 equally spaced time steps)...")
    # RA×DEC plots will automatically use 52 equally spaced time steps
    plot_radec_propagation(all_query_results, methods, method_colors)  # Every 86400 seconds (~1 day if 1h steps)
    time_elapsed = time.time() - time_ref
    print(f"RAxDEC plotting time: {time_elapsed} seconds")
    print("RAxDEC plots complete.")


    #print("3. Generating alphashape area evolution plots...") # Validated
    time_ref = time.time()
    plot_alphashape_area(all_query_results, methods, method_colors)
    time_elapsed = time.time() - time_ref
    print(f"Alphashape area plotting time: {time_elapsed} seconds")
    print("Alphashape area plots complete.")

    time_ref = time.time()
    print("4. Generating Range×Range-rate phase space plots (12 equally spaced time steps)...")
    print("4a. All methods including DAIOD_DA...")
    plot_range_rangerate_phase_space(all_query_results, methods, method_colors)
    print("4b. Without DAIOD_DA (MC methods only)...")
    plot_range_rangerate_no_daiod_da(all_query_results, methods, method_colors)
    time_elapsed = time.time() - time_ref
    print(f"Range×Range-rate plotting time: {time_elapsed} seconds")
    print("Range×Range-rate plots complete.")
    #print("4. Generating ADS split history...") #Needs testing (re-run sim with 2 splits)
    #plot_ads_split_history(all_query_results)
    
    #print("5. Generating wall clock time comparison...") # validated
    time_ref = time.time()
    plot_wall_clock_comparison(df, methods)
    time_elapsed = time.time() - time_ref
    print(f"Wall clock time plotting time: {time_elapsed} seconds")
    print(f"Wall clock time comparison complete.")
    #print("6. Generating precision vs wall time scatter...")        # validated
    #plot_precision_vs_walltime_scatter(df, method_colors, method_markers)
    
    #print("7. Generating cumulative wall time plot...") -> Cannot be done as only have final computional times for different arcs
    time_ref = time.time()
    plot_cumulative_walltime(df, methods, method_colors) # Validated
    time_elapsed = time.time() - time_ref
    print(f"Cumulative wall time plotting time: {time_elapsed} seconds")
    print(f"Cumulative wall time plots complete.")
    #print("8. Generating orbital motion validation...")
    #plot_orbital_motion_validation(all_query_results, methods, method_colors)
    
    #print("\n=== Plotting Complete ===")

if __name__ == "__main__":
    main()