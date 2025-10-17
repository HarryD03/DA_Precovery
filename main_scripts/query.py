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

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import utils.post_process as post

# Add after line 17 (after imports):

def detect_and_fix_segment_overlaps(alphashape, method_name, tolerance_rad=1e-8):
    """
    Detect and fix overlapping line segments in alphashape
    
    :param alphashape: Input alphashape
    :param method_name: Method name for logging
    :param tolerance_rad: Tolerance for detecting overlaps (radians)
    :return: (fixed_shape, was_fixed, diagnostics)
    """
    if alphashape is None:
        return None, False, "Null alphashape"
    
    try:
        from shapely.geometry import Polygon, LineString
        from shapely.ops import unary_union
        import numpy as np
        
        # Get coordinates
        coords = np.array(alphashape.exterior.coords)
        n_coords = len(coords) - 1  # Exclude duplicate last point
        
        print(f"    {method_name}: Checking {n_coords} segments for overlaps...")
        
        # Create line segments
        segments = []
        for i in range(n_coords):
            seg = LineString([coords[i], coords[(i+1) % n_coords]])
            segments.append(seg)
        
        # Check for segment intersections/overlaps
        overlaps_found = []
        
        for i in range(len(segments)):
            for j in range(i+2, len(segments)):  # Skip adjacent segments
                seg1 = segments[i]
                seg2 = segments[j]
                
                # Check if segments intersect (beyond just touching at endpoints)
                if seg1.intersects(seg2):
                    intersection = seg1.intersection(seg2)
                    
                    # Check if intersection is more than just a point
                    if hasattr(intersection, 'length') and intersection.length > tolerance_rad:
                        overlaps_found.append((i, j, intersection.length))
                    elif hasattr(intersection, 'geoms'):
                        # Multiple intersection points/lines
                        total_length = sum(geom.length for geom in intersection.geoms 
                                         if hasattr(geom, 'length'))
                        if total_length > tolerance_rad:
                            overlaps_found.append((i, j, total_length))
        
        diagnostics = {
            'original_segments': n_coords,
            'overlaps_found': len(overlaps_found),
            'overlap_details': overlaps_found[:5],  # First 5 overlaps
            'is_valid_shapely': alphashape.is_valid
        }
        
        print(f"      Found {len(overlaps_found)} segment overlaps")
        
        # If no overlaps, return original
        if len(overlaps_found) == 0:
            return alphashape, False, diagnostics
        
        # Fix overlaps using different strategies
        fixed_shape = None
        fix_method = "none"
        
        # Strategy 1: Simplify the polygon to remove overlaps
        try:
            print(f"      → Trying polygon simplification...")
            from shapely import simplify
            
            # Calculate appropriate tolerance (1% of average segment length)
            avg_segment_length = np.mean([seg.length for seg in segments])
            simplify_tolerance = avg_segment_length * 0.01
            
            simplified = simplify(alphashape, simplify_tolerance)
            
            # Check if simplification removed overlaps
            if simplified.is_valid and hasattr(simplified, 'exterior'):
                simple_coords = np.array(simplified.exterior.coords)
                simple_segments = []
                
                for i in range(len(simple_coords) - 1):
                    seg = LineString([simple_coords[i], simple_coords[i+1]])
                    simple_segments.append(seg)
                
                # Quick overlap check on simplified polygon
                simple_overlaps = 0
                for i in range(len(simple_segments)):
                    for j in range(i+2, len(simple_segments)):
                        if simple_segments[i].intersects(simple_segments[j]):
                            intersection = simple_segments[i].intersection(simple_segments[j])
                            if hasattr(intersection, 'length') and intersection.length > tolerance_rad:
                                simple_overlaps += 1
                
                if simple_overlaps == 0:
                    fixed_shape = simplified
                    fix_method = "simplify"
                    print(f"      ✓ Simplification removed overlaps")
                else:
                    print(f"      ✗ Simplification still has {simple_overlaps} overlaps")
            
        except Exception as e:
            print(f"      ✗ Simplification failed: {e}")
        
        # Strategy 2: Buffer with very small distance (can fix self-intersections)
        if fixed_shape is None:
            try:
                print(f"      → Trying tiny buffer...")
                buffer_distance = alphashape.area**0.5 * 1e-8  # Very small buffer
                buffered = alphashape.buffer(buffer_distance)
                
                if buffered.is_valid and hasattr(buffered, 'exterior'):
                    # Handle MultiPolygon case
                    if buffered.geom_type == 'MultiPolygon':
                        buffered = max(buffered.geoms, key=lambda p: p.area)
                    
                    if buffered.is_valid:
                        fixed_shape = buffered
                        fix_method = "tiny_buffer"
                        print(f"      ✓ Tiny buffer successful")
                
            except Exception as e:
                print(f"      ✗ Tiny buffer failed: {e}")
        
        # Strategy 3: Remove problematic vertices
        if fixed_shape is None and len(overlaps_found) < 10:  # Only if few overlaps
            try:
                print(f"      → Trying vertex removal...")
                
                # Identify vertices involved in overlaps
                problem_vertices = set()
                for i, j, length in overlaps_found:
                    problem_vertices.add(i)
                    problem_vertices.add(j)
                    problem_vertices.add((i+1) % n_coords)
                    problem_vertices.add((j+1) % n_coords)
                
                # Remove problematic vertices (but keep at least 4 for a valid polygon)
                if len(problem_vertices) < n_coords - 3:
                    clean_coords = []
                    for i in range(n_coords):
                        if i not in problem_vertices:
                            clean_coords.append(coords[i])
                    
                    # Close polygon
                    if len(clean_coords) >= 3:
                        clean_coords.append(clean_coords[0])
                        cleaned_polygon = Polygon(clean_coords)
                        
                        if cleaned_polygon.is_valid:
                            fixed_shape = cleaned_polygon
                            fix_method = "vertex_removal"
                            print(f"      ✓ Vertex removal successful ({len(clean_coords)-1} vertices)")
                
            except Exception as e:
                print(f"      ✗ Vertex removal failed: {e}")
        
        # Strategy 4: Convex hull (guaranteed to work, but changes shape)
        if fixed_shape is None:
            try:
                print(f"      → Using convex hull (shape will change)...")
                from shapely import convex_hull
                hull = convex_hull(alphashape)
                
                if hull.is_valid:
                    fixed_shape = hull
                    fix_method = "convex_hull"
                    print(f"      ✓ Convex hull successful")
            
            except Exception as e:
                print(f"      ✗ Convex hull failed: {e}")
        
        # Return results
        if fixed_shape is not None and fixed_shape.is_valid:
            diagnostics.update({
                'was_fixed': True,
                'fix_method': fix_method,
                'final_segments': len(np.array(fixed_shape.exterior.coords)) - 1,
                'area_ratio': fixed_shape.area / alphashape.area if alphashape.area > 0 else 1.0
            })
            
            print(f"      → Fixed using {fix_method}, area ratio: {diagnostics['area_ratio']:.3f}")
            return fixed_shape, True, diagnostics
        else:
            diagnostics['was_fixed'] = False
            print(f"      ✗ All overlap removal strategies failed")
            return alphashape, False, diagnostics
    
    except Exception as e:
        return alphashape, False, f"Overlap detection error: {e}"

def test_database_compatibility(shape, method_name):
    """
    Test if a shape will work with the database by checking for common issues
    INCLUDING segment overlaps (the main cause of database errors)
    """
    if shape is None:
        return False, "Null shape"
    
    try:
        from shapely.geometry import LineString
        
        # Basic validity
        if not shape.is_valid:
            return False, "Invalid geometry"
        
        # Check area
        area_rad2 = shape.area
        if area_rad2 > 3e-4:  # Roughly equivalent to 0.1 deg² in radians²
            return False, f"Area too large: {area_rad2:.6e} rad²"
        
        # Check vertex count
        coords = np.array(shape.exterior.coords)
        n_vertices = len(coords) - 1
        if n_vertices > 20:
            return False, f"Too many vertices: {n_vertices}"
        
        # Check for tiny segments
        min_segment_length = float('inf')
        for i in range(n_vertices):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            length = np.sqrt(dx**2 + dy**2)
            min_segment_length = min(min_segment_length, length)
        
        if min_segment_length < 1e-10:
            return False, f"Tiny segments detected: {min_segment_length:.2e}"
        
        # CRITICAL: Check for overlapping line segments (main database error cause)
        segments = []
        for i in range(n_vertices):
            seg = LineString([coords[i], coords[(i+1) % n_vertices]])
            segments.append(seg)
        
        overlaps_found = 0
        tolerance_rad = 1e-15
        
        # Check all non-adjacent segment pairs for overlaps
        for i in range(len(segments)):
            for j in range(i+2, len(segments)):
                # Skip the wrap-around edge for closed polygons
                if i == 0 and j == len(segments) - 1:
                    continue
                
                seg1 = segments[i]
                seg2 = segments[j]
                
                if seg1.intersects(seg2):
                    intersection = seg1.intersection(seg2)
                    
                    # Check if intersection is more than just touching at endpoints
                    if hasattr(intersection, 'length') and intersection.length > tolerance_rad:
                        overlaps_found += 1
                    elif hasattr(intersection, 'geoms'):
                        # Multiple intersections
                        for geom in intersection.geoms:
                            if hasattr(geom, 'length') and geom.length > tolerance_rad:
                                overlaps_found += 1
                
                # Stop early if we found overlaps (for performance)
                if overlaps_found > 0:
                    break
            if overlaps_found > 0:
                break
        
        if overlaps_found > 0:
            return False, f"Overlapping segments detected: {overlaps_found} overlaps"
        
        print(f"    {method_name}: Compatibility check PASSED (no overlaps)")
        return True, "Compatible"
        
    except Exception as e:
        return False, f"Compatibility check error: {e}"


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

def load_all_simulation_data():
    """Load all pickle files from the Simulation_data folder"""
    
    # Get the directory where query.py is located
    script_dir = Path(__file__).parent
    simulation_data_dir = script_dir / "Simulation_data"  # Read from Simulation_data
    
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
                  f"{len(arc_data['alphashapes']['DAIOD_ADS_ADS'])} alphashapes per method")
            
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_arc_data)} arc datasets")
    return all_arc_data

def run_queries_for_arc(arc_data, arc_index):
    """Run database queries for a single arc's data"""
    
    print(f"\n=== Processing Arc {arc_index} ===")
    print(f"Arc length: {arc_data['current_dt']:.3f} days")
    print(f"Time range: {arc_data['tstart'].iso} to {arc_data['tfinal'].iso}")
    
    # Extract key data
    tgrid = arc_data['tgrid']
    t_propagation = arc_data['t_propagation']
    alphashapes = arc_data['alphashapes']
    
    # Initialize result storage
    methods = ['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC']
    n_times = len(tgrid)
    n_methods = len(methods)
    
    query_results = {
        'N_images': np.zeros((n_times, n_methods)),
        'TP': np.zeros((n_times, n_methods)),
        'FP': np.zeros((n_times, n_methods)),
        'FN': np.zeros((n_times, n_methods)),
        'precision': np.zeros((n_times, n_methods)),
        'recall': np.zeros((n_times, n_methods)),
        'query_times': np.zeros((n_times, n_methods)),
        
        # NEW: Add geometry information storage
        'geometry_info': {
            'shape_types': np.full((n_times, n_methods), '', dtype='U20'),  # 'alphashape' or 'convex_hull'
            'vertex_counts': np.zeros((n_times, n_methods), dtype=int),
            'areas': np.zeros((n_times, n_methods)),
            'was_fixed': np.zeros((n_times, n_methods), dtype=bool),
            'fix_methods': np.full((n_times, n_methods), '', dtype='U20'),
            'processing_notes': np.full((n_times, n_methods), '', dtype='U100')
        }
    }
    
    # Run queries for each time step and method
    for time_idx in range(n_times):
        print(f"  Time step {time_idx+1}/{n_times} ({t_propagation[time_idx].iso})")
        
        # Define query time window
        t_start = t_propagation[time_idx]
        t_end = t_start + TimeDelta(30, format='sec')  # 30 second window
        
        for method_idx, method_name in enumerate(methods):
            try:
                # Get the alphashape for this method and time
                alphashape = alphashapes[method_name][time_idx]
                
                if alphashape is None:
                    print(f"    {method_name}: SKIPPED - null alphashape")
                    query_results['geometry_info']['processing_notes'][time_idx, method_idx] = "null_alphashape"
                    continue
                
                # Initialize geometry tracking
                used_shape = alphashape
                shape_type = "alphashape"
                was_fixed = False
                fix_method = "none"
                processing_notes = []
                
                try:
                    # Try with original alphashape first
                    import time
                    query_start = time.time()
                    metrics = post.full_query(alphashape, t_start, t_end)
                    query_time = time.time() - query_start
                    
                    # Store successful results
                    query_results['N_images'][time_idx, method_idx] = metrics['Total_Images']
                    query_results['TP'][time_idx, method_idx] = metrics['TP']
                    query_results['FP'][time_idx, method_idx] = metrics['FP']
                    query_results['FN'][time_idx, method_idx] = metrics['FN']
                    query_results['precision'][time_idx, method_idx] = metrics['precision']
                    query_results['recall'][time_idx, method_idx] = metrics['recall']
                    query_results['query_times'][time_idx, method_idx] = query_time
                    
                    # Store geometry info for successful alphashape
                    query_results['geometry_info']['shape_types'][time_idx, method_idx] = shape_type
                    query_results['geometry_info']['vertex_counts'][time_idx, method_idx] = len(alphashape.exterior.coords) - 1
                    query_results['geometry_info']['areas'][time_idx, method_idx] = alphashape.area
                    query_results['geometry_info']['was_fixed'][time_idx, method_idx] = False
                    query_results['geometry_info']['fix_methods'][time_idx, method_idx] = "none"
                    query_results['geometry_info']['processing_notes'][time_idx, method_idx] = "original_alphashape_success"
                    
                    print(f"    {method_name}: ✓ SUCCESS (ALPHASHAPE) - {metrics['Total_Images']} images, "
                          f"vertices: {len(alphashape.exterior.coords) - 1}, "
                          f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f} ({query_time:.2f}s)")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"    {method_name}: Alphashape failed - {error_msg}")
                    
                    if 'Empty table' in error_msg or 'scalar value' in error_msg:   #No images
                        query_time = time.time() - query_start
                        print(f"      → Database returned no images for this region/time")
                        query_results['geometry_info']['processing_notes'][time_idx, method_idx] = "empty_query_result"
                        # Still record geometry info
                        query_results['geometry_info']['shape_types'][time_idx, method_idx] = "alphashape"
                        query_results['geometry_info']['vertex_counts'][time_idx, method_idx] = len(alphashape.exterior.coords) - 1
                        query_results['geometry_info']['areas'][time_idx, method_idx] = alphashape.area

                        query_results['N_images'][time_idx, method_idx] = 0
                        query_results['TP'][time_idx, method_idx] = None
                        query_results['FP'][time_idx, method_idx] = None
                        query_results['FN'][time_idx, method_idx] = None
                        query_results['precision'][time_idx, method_idx] = None
                        query_results['recall'][time_idx, method_idx] = None
                        query_results['query_times'][time_idx, method_idx] = query_time

                        continue


                    # Try convex hull as fallback
                    if 'spherepoly_from_array' in error_msg:
                        try:
                            print(f"      → Trying convex hull fallback...")
                            test_hull = alphashape.convex_hull
                            
                            import time
                            query_start = time.time()
                            metrics = post.full_query(test_hull, t_start, t_end)
                            query_time = time.time() - query_start
                            
                            # Store convex hull results
                            query_results['N_images'][time_idx, method_idx] = metrics['Total_Images']
                            query_results['TP'][time_idx, method_idx] = metrics['TP']
                            query_results['FP'][time_idx, method_idx] = metrics['FP']
                            query_results['FN'][time_idx, method_idx] = metrics['FN']
                            query_results['precision'][time_idx, method_idx] = metrics['precision']
                            query_results['recall'][time_idx, method_idx] = metrics['recall']
                            query_results['query_times'][time_idx, method_idx] = query_time
                            
                            # Store geometry info for convex hull
                            used_shape = test_hull
                            shape_type = "convex_hull"
                            was_fixed = True
                            fix_method = "convex_hull"
                            
                            query_results['geometry_info']['shape_types'][time_idx, method_idx] = shape_type
                            query_results['geometry_info']['vertex_counts'][time_idx, method_idx] = len(test_hull.exterior.coords) - 1
                            query_results['geometry_info']['areas'][time_idx, method_idx] = test_hull.area
                            query_results['geometry_info']['was_fixed'][time_idx, method_idx] = True
                            query_results['geometry_info']['fix_methods'][time_idx, method_idx] = fix_method
                            query_results['geometry_info']['processing_notes'][time_idx, method_idx] = f"alphashape_failed_convex_hull_success"
                            
                            print(f"      ✓ Convex hull works: {metrics['Total_Images']} images, "
                                  f"vertices: {len(test_hull.exterior.coords) - 1}")
                            print(f"    {method_name}: ✓ SUCCESS (CONVEX HULL) - {metrics['Total_Images']} images, "
                                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f} ({query_time:.2f}s)")
                            
                        except Exception as hull_error:
                            error_msg = str(hull_error)
                            print(f"    {method_name}: Alphashape failed - {error_msg}")

                            if 'Empty table' in error_msg or 'scalar value' in error_msg:   #No images - Save.
                                print(f"      ✗ Even convex hull failed: {error_msg}")
                                
                                query_time = time.time() - query_start
                                print(f"      → Database returned no images for this region/time")
                                query_results['geometry_info']['processing_notes'][time_idx, method_idx] = "empty_query_result"
                                # Still record geometry info
                                query_results['geometry_info']['shape_types'][time_idx, method_idx] = "alphashape"
                                query_results['geometry_info']['vertex_counts'][time_idx, method_idx] = len(alphashape.exterior.coords) - 1
                                query_results['geometry_info']['areas'][time_idx, method_idx] = alphashape.area

                                query_results['N_images'][time_idx, method_idx] = 0
                                query_results['TP'][time_idx, method_idx] = None
                                query_results['FP'][time_idx, method_idx] = None
                                query_results['FN'][time_idx, method_idx] = None
                                query_results['precision'][time_idx, method_idx] = None
                                query_results['recall'][time_idx, method_idx] = None
                                query_results['query_times'][time_idx, method_idx] = query_time                                
                                
                                query_results['geometry_info']['processing_notes'][time_idx, method_idx] = f"both_failed: {str(error_msg)[:50]}"
                                continue
                    else:
                        print(f"    {method_name}: ✗ FAILED - {error_msg}")
                        query_results['geometry_info']['processing_notes'][time_idx, method_idx] = f"error: {str(e)[:50]}"
                        continue
                        
            except Exception as e:
                print(f"    {method_name}: ✗ UNEXPECTED ERROR - {e}")
                query_results['geometry_info']['processing_notes'][time_idx, method_idx] = f"unexpected_error: {str(e)[:50]}"
                continue
    
    return query_results

def save_query_results(all_query_results, output_dir):
    """Save all query results to files in the same format as simulation data files"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create Query_data directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each arc's results as individual files (matching main_final.py format)
    saved_files = []
    
    for arc_idx, result_data in all_query_results.items():
        arc_data = result_data['arc_data']
        query_results = result_data['query_results']
        
        # Extract arc length for filename
        dt_days = arc_data['current_dt']
        
        # Structure the data to match main_final.py format
        complete_arc_data = {
            # Copy all original arc data
            **arc_data,
            
            # Add query-specific results
            'query_results': query_results,
            'query_timestamp': timestamp,
            'query_processing_complete': True,
            
            # Query performance metrics by method
            'query_performance': {
                'methods': ['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'],
                'total_images_found': {
                    method: query_results['N_images'][:, idx].sum() 
                    for idx, method in enumerate(['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'])
                },
                'total_true_positives': {
                    method: query_results['TP'][:, idx].sum() 
                    for idx, method in enumerate(['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'])
                },
                'average_precision': {
                    method: np.mean(query_results['precision'][:, idx][query_results['precision'][:, idx] > 0])
                    for idx, method in enumerate(['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'])
                },
                'average_recall': {
                    method: np.mean(query_results['recall'][:, idx][query_results['recall'][:, idx] > 0])
                    for idx, method in enumerate(['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'])
                },
                'total_query_time': {
                    method: query_results['query_times'][:, idx].sum()
                    for idx, method in enumerate(['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'])
                },
                
                # NEW: Add geometry statistics
                'geometry_statistics': {
                    method: {
                        'shape_type_counts': {
                            shape_type: np.sum(query_results['geometry_info']['shape_types'][:, idx] == shape_type)
                            for shape_type in ['alphashape', 'convex_hull', '']
                        },
                        'average_vertices': np.mean(query_results['geometry_info']['vertex_counts'][:, idx][
                            query_results['geometry_info']['vertex_counts'][:, idx] > 0
                        ]) if np.any(query_results['geometry_info']['vertex_counts'][:, idx] > 0) else 0,
                        'average_area': np.mean(query_results['geometry_info']['areas'][:, idx][
                            query_results['geometry_info']['areas'][:, idx] > 0
                        ]) if np.any(query_results['geometry_info']['areas'][:, idx] > 0) else 0,
                        'fix_rate': np.mean(query_results['geometry_info']['was_fixed'][:, idx]),
                        'successful_queries': np.sum(query_results['N_images'][:, idx] >= 0)  # Count non-zero queries
                    }
                    for idx, method in enumerate(['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC'])
                }
            }
        }
        
        # First save DA parameters separately (matching main_final.py pattern)
        da_params_path = output_dir / f"arc_{arc_idx:03d}_query_da_params_{timestamp}.pkl"
        da_params = {
            'da_order': complete_arc_data.get('algorithm_params', {}).get('da_order', 5),
            'da_nvars': complete_arc_data.get('algorithm_params', {}).get('da_nvars', 6),
            'arc_index': arc_idx,
            'timestamp': timestamp,
            'query_processing': True
        }
        
        try:
            with open(da_params_path, 'wb') as f:
                pickle.dump(da_params, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Query DA parameters saved: {da_params_path}")
        except Exception as e:
            print(f"✗ Error saving query DA parameters for arc {arc_idx}: {e}")
        
        # Save main query results file (matching main_final.py naming pattern)
        arc_results_path = output_dir / f"arc_{arc_idx:03d}_dt_{dt_days:.3f}d_query_results_{timestamp}.pkl"
        
        try:
            with open(arc_results_path, 'wb') as f:
                pickle.dump(complete_arc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = arc_results_path.stat().st_size / (1024**2)
            print(f"✓ Arc {arc_idx} query results saved: {arc_results_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Arc length: {dt_days:.3f} days")
            
            saved_files.append(arc_results_path)
            
        except Exception as e:
            print(f"✗ Error saving arc {arc_idx} query results: {e}")
        
        # Save individual arc summary CSV (matching main_final.py pattern)
        arc_summary_data = {
            'arc_index': arc_idx,
            'dt_days': dt_days,
            'obs_time_center': complete_arc_data.get('obs_time0', 'Unknown').iso if hasattr(complete_arc_data.get('obs_time0', ''), 'iso') else 'Unknown',
            'prop_start': complete_arc_data.get('tstart', 'Unknown').iso if hasattr(complete_arc_data.get('tstart', ''), 'iso') else 'Unknown',
            'prop_end': complete_arc_data.get('tfinal', 'Unknown').iso if hasattr(complete_arc_data.get('tfinal', ''), 'iso') else 'Unknown',
            'n_time_steps': complete_arc_data.get('Ts', 0),
            'total_query_time_sec': query_results['query_times'].sum(),
            'total_images_all_methods': query_results['N_images'].sum(),
            'total_tp_all_methods': query_results['TP'].sum(),
            'save_path': str(arc_results_path),
            'timestamp': timestamp,
            'processing_type': 'query_results'
        }
        
        arc_summary_df = pd.DataFrame([arc_summary_data])
        arc_summary_csv_path = output_dir / f"arc_{arc_idx:03d}_query_summary_{timestamp}.csv"
        arc_summary_df.to_csv(arc_summary_csv_path, index=False)
    
    # Also save a combined summary CSV with all arcs
    if all_query_results:
        combined_summary_data = []
        methods = ['DAIOD_ADS_ADS', 'DAIOD_ADS', 'DAIOD_DA', 'DAIOD_MC', 'GAUSS_MC']
        
        for arc_idx, result_data in all_query_results.items():
            arc_data = result_data['arc_data']
            query_results = result_data['query_results']
            
            for time_idx in range(query_results['N_images'].shape[0]):
                for method_idx, method_name in enumerate(methods):
                    combined_summary_data.append({
                        'arc_index': arc_idx,
                        'dt_days': arc_data['current_dt'],
                        'time_idx': time_idx,
                        'time_iso': arc_data['t_propagation'][time_idx].iso,
                        'method': method_name,
                        'N_images': query_results['N_images'][time_idx, method_idx],
                        'TP': query_results['TP'][time_idx, method_idx],
                        'FP': query_results['FP'][time_idx, method_idx],
                        'FN': query_results['FN'][time_idx, method_idx],
                        'precision': query_results['precision'][time_idx, method_idx],
                        'recall': query_results['recall'][time_idx, method_idx],
                        'query_time_sec': query_results['query_times'][time_idx, method_idx],
                        
                        # NEW: Add geometry information to CSV
                        'shape_type': query_results['geometry_info']['shape_types'][time_idx, method_idx],
                        'vertex_count': query_results['geometry_info']['vertex_counts'][time_idx, method_idx],
                        'area': query_results['geometry_info']['areas'][time_idx, method_idx],
                        'was_fixed': query_results['geometry_info']['was_fixed'][time_idx, method_idx],
                        'fix_method': query_results['geometry_info']['fix_methods'][time_idx, method_idx],
                        'processing_notes': query_results['geometry_info']['processing_notes'][time_idx, method_idx],
                        
                        'timestamp': timestamp
                    })
        
        combined_summary_df = pd.DataFrame(combined_summary_data)
        combined_summary_csv_path = output_dir / f"all_arcs_query_summary_{timestamp}.csv"
        combined_summary_df.to_csv(combined_summary_csv_path, index=False)
        print(f"Combined summary CSV saved: {combined_summary_csv_path}")
    
    print(f"\n=== Query Results Saved in Simulation Data Format ===")
    print(f"Processed {len(all_query_results)} arcs")
    print(f"Individual arc files: {len(saved_files)}")
    print(f"Files saved to: {output_dir}")
    
    return saved_files

def main():
    """Main query processing function"""
    
    print("=== Apophis Query Processing ===")
    
    # Load all simulation data
    all_arc_data = load_all_simulation_data()
    if all_arc_data is None:
        return
    
    # Process each arc
    all_query_results = {}
    
    for arc_index in sorted(all_arc_data.keys()):
        arc_data = all_arc_data[arc_index]
        
        try:
            query_results = run_queries_for_arc(arc_data, arc_index)
            
            all_query_results[arc_index] = {
                'arc_data': arc_data,
                'query_results': query_results
            }
            
            print(f"✓ Completed Arc {arc_index}")
            
        except Exception as e:
            print(f"✗ Error processing Arc {arc_index}: {e}")
            continue
    
    # Save all results to Query_data folder in simulation data format
    script_dir = Path(__file__).parent
    query_output_dir = script_dir / "Query_data"
    
    if all_query_results:
        saved_files = save_query_results(all_query_results, query_output_dir)
        
        print(f"\n=== Query Processing Complete ===")
        print(f"Processed {len(all_query_results)} arcs")
        print(f"Saved {len(saved_files)} individual arc result files")
        print(f"Results directory: {query_output_dir}")
    else:
        print("No query results to save.")

def load_specific_arc(arc_index):
    """Load data for a specific arc only"""
    
    script_dir = Path(__file__).parent
    simulation_data_dir = script_dir / "Simulation_data"  # Still reads from Simulation_data
    
    # Load DA parameters first
    da_params = load_da_params_for_arc(arc_index, simulation_data_dir)
    if da_params is None:
        print(f"Cannot load DA parameters for arc {arc_index}")
        return None
    
    # Initialize DA
    initialize_da_from_params(da_params)
    
    # Find main pickle file for specific arc
    pattern = str(simulation_data_dir / f"arc_{arc_index:03d}_*_complete_results_*.pkl")
    pickle_files = glob.glob(pattern)
    
    if not pickle_files:
        print(f"No data found for arc {arc_index}")
        return None
    
    # Load the most recent file if multiple exist
    pickle_file = max(pickle_files, key=os.path.getctime)
    
    print(f"Loading arc {arc_index} from: {Path(pickle_file).name}")
    
    with open(pickle_file, 'rb') as f:
        arc_data = pickle.load(f)
    
    return arc_data

if __name__ == "__main__":
    # Run all arcs
    main()
    
    # Or run specific arc:
    # arc_data = load_specific_arc(0)
    # if arc_data:
    #     query_results = run_queries_for_arc(arc_data, 0)