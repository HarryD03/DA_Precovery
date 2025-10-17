from astroquery.cadc import Cadc
import numpy as np
from shapely.geometry import Polygon
import os

def query_polygon_images(polygon_vertices, collections=None, certificate_path=None):
    """
    Query CADC database for the number of images within a polygon defined by vertices.
    
    Parameters
    ----------
    polygon_vertices : list of tuples
        List of (RA, DEC) coordinates in degrees (ICRS) defining the polygon vertices
    collections : list of str, optional
        List of telescope collections to search (e.g., ['CFHT', 'HST', 'GEMINI'])
    certificate_path : str, optional
        Path to CADC certificate file for authentication
        
    Returns
    -------
    tuple
        (count, result_table) where count is the number of images and result_table contains the full query results
    """
    
    # Initialize CADC object
    cadc = Cadc()
    
    # Login with certificate if provided
    if certificate_path and os.path.exists(certificate_path):
        try:
            cadc.login(certificate_file=certificate_path)
            print("Successfully logged in to CADC")
        except Exception as e:
            print(f"Warning: Login failed ({e}), continuing with anonymous access")
    else:
        print("No certificate provided or file not found, using anonymous access")
    
    # Convert vertices to proper polygon string format for ADQL
    if isinstance(polygon_vertices[0], (list, tuple)):
        # Already in coordinate pairs
        coord_pairs = [f"{ra} {dec}" for ra, dec in polygon_vertices]
    else:
        # Assume flat list [ra1, dec1, ra2, dec2, ...]
        coord_pairs = []
        for i in range(0, len(polygon_vertices), 2):
            coord_pairs.append(f"{polygon_vertices[i]} {polygon_vertices[i+1]}")
    
    polygon_str = " ".join(coord_pairs)
    
    # Build ADQL query
    query_outline = f"""
    SELECT {num} *
    FROM caom2.Plane AS Plane
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
    WHERE (INTERSECTS( INTERVAL ( {mjd_start} , {mjd_end} ), Plane.time_bounds_samples ) = 1 
        AND CONTAINS( POINT('ICRS', {ra}, {dec}), Plane.position_bounds ) = 1 
        AND LOWER(Plane.energy_bandpassName) LIKE '{filter}%' 
        AND collection = '{collection}'
        AND calibrationLevel >= {cal_level}
        AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
        ORDER BY time_bounds_lower {order}"""
    
    
    """
    WHERE dataproduct_type = 'image'
    AND INTERSECTS(POLYGON('ICRS', {polygon_str}), s_region) = 1
    """
    
    # Add collection filter if specified
    if collections:
        collection_filter = "','".join(collections)
        query += f" AND obs_collection IN ('{collection_filter}')"
    
    try:
        # Execute the query
        print("Executing CADC query...")
        print(f"Query: {query}")
        
        result = cadc.exec_sync(query)
        image_count = len(result)
        
        print(f"Found {image_count} images within the polygon")
        
        return image_count, result
        
    except Exception as e:
        print(f"Query failed: {e}")
        return 0, None

def sort_vertices_ccw(vertices):
    """
    Sort polygon vertices in counter-clockwise order.
    
    Parameters
    ----------
    vertices : list of tuples
        List of (RA, DEC) coordinates in degrees
        
    Returns
    -------
    list of tuples
        Vertices sorted in counter-clockwise order
    """
    vertices = np.array(vertices)
    
    # Calculate centroid
    centroid = np.mean(vertices, axis=0)
    
    # Calculate angles from centroid to each vertex
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    
    # Sort by angle (counter-clockwise)
    sorted_indices = np.argsort(angles)
    
    return vertices[sorted_indices].tolist()

def create_test_polygon():
    """
    Create a test polygon for demonstration purposes.
    
    Returns
    -------
    list of tuples
        Test polygon vertices in (RA, DEC) degrees
    """
    # Example polygon around M31 region
    vertices = [
        (10.5, 41.0),   # Bottom-left
        (11.0, 41.0),   # Bottom-right  
        (11.0, 41.5),   # Top-right
        (10.5, 41.5),   # Top-left
        (10.5, 41.0)    # Close the polygon
    ]
    
    return vertices

def main():
    """
    Main function demonstrating polygon query functionality.
    """
    
    print("CADC Polygon Query Script")
    print("=" * 40)
    
    # Certificate path
    cert_path = 'C:/Users/chezh/OneDrive - Cranfield University/Documents/IRP/05_Thesis_code/utils/cadcproxy.pem'
    
    # Create test polygon
    test_vertices = create_test_polygon()
    print(f"Test polygon vertices: {test_vertices}")
    
    # Sort vertices counter-clockwise (recommended for CADC)
    sorted_vertices = sort_vertices_ccw(test_vertices)
    print(f"Sorted vertices (CCW): {sorted_vertices}")
    
    # Query for images within the polygon
    collections_to_search = ['CFHT', 'HST', 'GEMINI']
    
    print(f"\nSearching collections: {collections_to_search}")
    count, results = query_polygon_images(
        polygon_vertices=sorted_vertices,
        collections=collections_to_search,
        certificate_path=cert_path
    )
    
    if results is not None and count > 0:
        print(f"\nQuery successful! Found {count} images.")
        print("\nSample results:")
        print(f"Columns available: {results.colnames[:10]}...")  # Show first 10 columns
        
        if 'obs_collection' in results.colnames:
            unique_collections = set(results['obs_collection'])
            print(f"Collections found: {unique_collections}")
            
        if 'target_name' in results.colnames:
            unique_targets = set(results['target_name'][:5])  # Show first 5 targets
            print(f"Sample targets: {unique_targets}")
    else:
        print("No images found or query failed.")

    # Test with a smaller, more focused polygon
    print("\n" + "="*40)
    print("Testing with smaller polygon...")
    
    small_vertices = [
        (45.0, 10.0),
        (45.1, 10.0), 
        (45.1, 10.1),
        (45.0, 10.1),
        (45.0, 10.0)
    ]
    
    count2, results2 = query_polygon_images(
        polygon_vertices=small_vertices,
        collections=['CFHT', 'HST'],
        certificate_path=cert_path
    )
    
    print(f"Small polygon test: Found {count2} images")

if __name__ == "__main__":
    main()
