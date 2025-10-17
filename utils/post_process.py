from os import error
import astropy
import numpy as np
from daceypy import ADS, DA, array
import daceypy.op as op
import pandas as pd
import shapely
import utils.time_reference as time_ref
from scipy.spatial import ConvexHull
import alphashape
from utils.optimisealpha import optimizealpha
import astroquery.cadc as CADC
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.time import Time, TimeDelta
import astropy.units as u

def generate_perimeter3D(x_error, y_error, z_error, Ns):
    """
    Generate the boundary of the Orbital Set
    """
    # Define grids for each axis
    xgrid = np.linspace(-1, 1, Ns)
    ygrid = np.linspace(-1, 1, Ns)
    zgrid = np.linspace(-1, 1, Ns)

    # Uncertainties in each direction
    xb = x_error
    yb = y_error
    zb = z_error

    # Create meshgrids for consistent shapes
    Y, Z = np.meshgrid(ygrid, zgrid, indexing='ij')  # Both shape (Ns, Ns)
    X, Z_xz = np.meshgrid(xgrid, zgrid, indexing='ij')  # Both shape (Ns, Ns)  
    X_xy, Y_xy = np.meshgrid(xgrid, ygrid, indexing='ij')  # Both shape (Ns, Ns)

    faces = []
    
    # Only add faces if the corresponding error is non-zero
    if abs(xb) > 1e-12:  # x faces only if x error is non-zero
        # x = -xb face
        face1 = np.stack((
            np.full((Ns, Ns), -xb),   # x = -xb
            yb * Y,                   # y varies
            zb * Z                    # z varies
        ), axis=-1).reshape(-1, 3)
        faces.append(face1)

        # x = +xb face
        face2 = np.stack((
            np.full((Ns, Ns), xb),    # x = +xb
            yb * Y,                   # y varies
            zb * Z                    # z varies
        ), axis=-1).reshape(-1, 3)
        faces.append(face2)

    if abs(yb) > 1e-12:  # y faces only if y error is non-zero
        # y = -yb face
        face3 = np.stack((
            xb * X,                   # x varies
            np.full((Ns, Ns), -yb),   # y = -yb
            zb * Z_xz                 # z varies
        ), axis=-1).reshape(-1, 3)
        faces.append(face3)

        # y = +yb face
        face4 = np.stack((
            xb * X,                   # x varies
            np.full((Ns, Ns), yb),    # y = +yb
            zb * Z_xz                 # z varies
        ), axis=-1).reshape(-1, 3)
        faces.append(face4)

    if abs(zb) > 1e-12:  # z faces only if z error is non-zero
        # z = -zb face
        face5 = np.stack((
            xb * X_xy,                # x varies
            yb * Y_xy,                # y varies
            np.full((Ns, Ns), -zb)    # z = -zb
        ), axis=-1).reshape(-1, 3)
        faces.append(face5)

        # z = +zb face
        face6 = np.stack((
            xb * X_xy,                # x varies
            yb * Y_xy,                # y varies
            np.full((Ns, Ns), zb)     # z = +zb
        ), axis=-1).reshape(-1, 3)
        faces.append(face6)

    # Concatenate all valid faces
    if faces:
        perimeter = np.concatenate(faces)
    else:
        # If all errors are zero, return a single point at origin
        perimeter = np.array([[0.0, 0.0, 0.0]])

    # Remove columns that are all zeros
    non_zero_cols = []
    for i in range(3):
        if not np.allclose(perimeter[:, i], 0.0, atol=1e-12):
            non_zero_cols.append(i)
    
    if non_zero_cols:
        perimeter_filtered = perimeter[:, non_zero_cols]
    else:
        # If all columns are zero, keep at least one dimension
        perimeter_filtered = perimeter[:, [0]]

    # Remove duplicate rows
    perimeter_unique = np.unique(perimeter_filtered, axis=0)

    # Create normalized version
    perimeter_norm = np.zeros_like(perimeter_unique)
    
    # Normalize only non-zero columns
    col_idx = 0
    for i in range(3):
        if i in non_zero_cols:
            if i == 0 and abs(xb) > 1e-12:
                perimeter_norm[:, col_idx] = perimeter_unique[:, col_idx] / xb
            elif i == 1 and abs(yb) > 1e-12:
                perimeter_norm[:, col_idx] = perimeter_unique[:, col_idx] / yb
            elif i == 2 and abs(zb) > 1e-12:
                perimeter_norm[:, col_idx] = perimeter_unique[:, col_idx] / zb
            else:
                perimeter_norm[:, col_idx] = perimeter_unique[:, col_idx]
            col_idx += 1

    return perimeter_norm, perimeter_unique

def gen_grid6D(RA_error, Dec_error, Ns):
    """
    Generate the boundary (perimeter) of a 6D uncertainty box.

    :param RA_error: The 1 sigma measurement precision in Right Ascension
    :param Dec_error: The 1 sigma measurement precision in Declination
    :param Ns: The number of steps for each axis
    :return: perimeter_norm, shape (number_of_perimeter_points, 6)
    """
    # Create grids for each axis
    grids = [np.linspace(-1, 1, Ns) for _ in range(6)]
    bounds = [RA_error, RA_error, RA_error, Dec_error, Dec_error, Dec_error]

    # For each axis, create the two "faces" at -bound and +bound, varying all other axes
    faces = []
    for dim in range(6):
        # Prepare meshgrid for all axes except the fixed one
        mesh_axes = [grids[d] if d != dim else None for d in range(6)]
        mesh_shape = [Ns if d != dim else 1 for d in range(6)]
        for sign in [-1, 1]:
            # Create a mesh for all axes except the fixed one
            mesh = np.meshgrid(*[g if g is not None else [0] for g in mesh_axes], indexing='ij')
            face = np.zeros(mesh[0].shape + (6,))
            for d in range(6):
                if d == dim:
                    face[..., d] = sign * bounds[d]
                else:
                    face[..., d] = bounds[d] * mesh[d]
            faces.append(face.reshape(-1, 6))

    # Concatenate all faces
    perimeter = np.concatenate(faces, axis=0)
    perimeter_norm = np.zeros_like(perimeter)
    for d in range(6):
        perimeter_norm[:, d] = perimeter[:, d] / bounds[d]
    
    return perimeter_norm #(Number of perimeter points, 6) -> Each row is from -1,1. Each collum presents the compoent variation
                        # i.e. if extracted one row, each element is the normalsed z position of a 6D perimeter, with values ranging from [-1,1].

def eval_perimeterADS3D(final_lists: list[list[ADS]], perimeter_norm: np.ndarray, time_vec: np.ndarray, time_idxs: np.ndarray=None, state_dim=6):
    """
        Evaluate the perimeter of an ADS object 

        :params final_lists: The post-propagation ADS object
        :params perimeter_norm: The normalized perimeter points so that the DA region is [-1, 1]. Structure: (Number of perimeter points, Uncertanity dimensions)
        :params time_vec: The time vector associated with the propagation
        :params time_idx: Indicies of Propagation times of interest
        :params state_dim: The state dimension of the problem (default: 6 for orbital mechanics)

        :returns final_map: The propagated evaluated region. 
                            Structure: List[ manifold1 = (Number of perimeter points, state number, subdomain number), manifold2 = (....), .... ]. 
                            len(final_map) = Ts. 

        :returns final_domain: The initial region split into subdomains
                                Structure: List[ domain1 = (Number of perimeter points, state number, subdomain number), domain2 = (....), .... ]. 
                                len(final_map) = Ts.

        :returns time_vec: The time vector associated
                            Structure: List[ Ts[0], Ts[1], ..., Ts[-1] ]
                            len(time_vec) = Ts

    """
    final_map_list = []
    final_domain_list = []
    time_list = []

    if time_idxs == None:
        if isinstance(time_vec, float):
            time_idxs = [0]
        else:
            time_idxs = np.arange(len(time_vec))

    for i in range(len(time_idxs)): # i represents the time instance
        #for ith time instance: 
        final_manifold = np.zeros((perimeter_norm.shape[0], state_dim, len(final_lists[time_idxs[i]])))  #propagated state
        final_domain = np.zeros((perimeter_norm.shape[0], state_dim, len(final_lists[time_idxs[i]])))    #ADS initial state/domain

        time_list.append(time_vec[i]) #append the time instance to the time list
        for j in range(len(final_lists[time_idxs[i]])): # for each ADS subdomain j at time i

            for k in range(perimeter_norm.shape[0]):   # for perimeter point k in the subdomain j within the time instance i
                eval_point = perimeter_norm[k,:]       # Select Perimeter point in 3D uncertainty region [x_norm, y_norm, z_norm]

                final_manifold[k,:,j] = final_lists[time_idxs[i]][j].manifold.eval(eval_point) #from the ADS list extract the perimeter manifold from the jth subdomain at the ith time instance
                final_domain[k,:,j] = final_lists[time_idxs[i]][j].box.eval(eval_point)

        final_map_list.append(final_manifold)
        final_domain_list.append(final_domain)

    return {
        "final_map": final_map_list,
        "final_domain": final_domain_list,
        "time": time_list
    }


def extract_2D_points(manifold, time_idx, component_indices, convert_to_degrees=False):
    """
    Extract 2D points from manifold data for all subdomains at a given time

    :param manifold: Evaluated ADS perimeter
    :param time_idx: Time index to process
    :param component_indices: Indices of the 2D components to extract (e.g., [0, 1] for RA-DEC)
    :param convert_to_degrees: Convert from radians to degrees for RA-DEC coordinates
    :return: Dictionary containing extracted points and metadata
    {
        "all points" : all_points, 
        "unique points" : unique_points, 
        "unique labels" : unique_labels, 
        "subdomain labels" : subdomain_labels}
        
    """
    # Initialize collections
    all_points = []
    subdomain_labels = []
    try:
        subdomains = manifold.shape[2]
    except IndexError:
        subdomains = 1
        manifold = manifold[:,:,np.newaxis]  # Add a new axis for subdomains if not present
    # Extract points from all subdomains
    for subdomain_idx in range(subdomains):
        # Extract 2D points from the specified components
        points_2d = manifold[:, component_indices, subdomain_idx]
        
        # Convert to degrees if specified (typically for RA-DEC coordinates)
        if convert_to_degrees and component_indices == [0, 1]:
            points_2d = np.degrees(points_2d)
        
        # Add points to collection
        all_points.extend(points_2d)
        subdomain_labels.extend([subdomain_idx] * len(points_2d))
    
    # Convert to numpy arrays
    all_points = np.array(all_points)
    subdomain_labels = np.array(subdomain_labels)
    
    # Remove duplicate points - The idea is that the overlapping points are boundaries of the subdomains.
    unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    unique_labels = subdomain_labels[unique_indices]
    
    return {
        "all points" : all_points, 
        "unique points" : unique_points, 
        "unique labels" : unique_labels, 
        "subdomain labels" : subdomain_labels}

def extract_2D_points_RADEC(manifold, time_idx, component_indices, convert_to_degrees=False):
    """
    Extract 2D points from manifold data for all subdomains at a given time

    :param manifold: Evaluated ADS perimeter in Cartesian Heliocentric Ecliptic coordinates
    :param time_idx: Time index to process
    :param component_indices: Indices of the 2D components to extract (e.g., [0, 1] for RA-DEC)
    :param convert_to_degrees: Convert from radians to degrees for RA-DEC coordinates
    :return: Dictionary containing extracted points and metadata
    """
    # Initialize collections
    all_points = []
    subdomain_labels = []
    
    # Extract points from all subdomains
    for subdomain_idx in range(manifold.shape[2]):
        # Extract all 6D points from the subdomain
        points_6d_ecliptic = manifold[:, :, subdomain_idx]  # Shape: (N_points, 6)
        
        # Convert from Heliocentric Ecliptic Cartesian to Heliocentric Equatorial Cartesian
        # Apply rotation matrix for ecliptic -> equatorial transformation
        obliquity = np.deg2rad(-23.43929111)  # J2000.0 obliquity
        rot_matrix = time_ref.ROT1(obliquity)
        
        # Initialize array for equatorial coordinates
        points_6d_equatorial = np.zeros_like(points_6d_ecliptic)
        
        # Transform each point
        for i in range(points_6d_ecliptic.shape[0]):
            # Apply rotation to position (first 3 components)
            points_6d_equatorial[i, :3] = rot_matrix @ points_6d_ecliptic[i, :3]
            # Apply same rotation to velocity (last 3 components)
            points_6d_equatorial[i, 3:] = rot_matrix @ points_6d_ecliptic[i, 3:]
        
        # Convert from Cartesian to Spherical coordinates (RA, DEC, range, RA_dot, DEC_dot, range_dot)
        points_6d_spherical = np.zeros_like(points_6d_equatorial)
        
        for i in range(points_6d_equatorial.shape[0]):
            # Use your time_ref.CC2obs function for the conversion
            points_6d_spherical[i, :] = time_ref.CC2obs(points_6d_equatorial[i, :])
        
        # Extract 2D points from the specified components (typically [0, 1] for RA, DEC)
        points_2d = points_6d_spherical[:, component_indices]
        
        # Convert to degrees if specified (typically for RA-DEC coordinates)
        if convert_to_degrees:
            points_2d = np.degrees(points_2d)
        
        # Add points to collection
        all_points.extend(points_2d)
        subdomain_labels.extend([subdomain_idx] * len(points_2d))
    
    # Convert to numpy arrays
    all_points = np.array(all_points)
    subdomain_labels = np.array(subdomain_labels)
    
    # Remove duplicate points
    unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    unique_labels = subdomain_labels[unique_indices]
    
    return {
        "all points" : all_points, 
        "unique points" : unique_points, 
        "unique labels" : unique_labels, 
        "subdomain labels" : subdomain_labels}

def create_alphashape2D(unique_points, alpha=None):
    """
    Create an bounded Alpha Shape from the perimeter points for a given time index

    :param unique_points: Unique 2D points to create the Alpha Shape
    :param alpha: Alpha value for the Alpha Shape (default: 0.1) if None will autodetermine
    :return: alpha_shape, boundary_coords, actual_alpha
    """

    if len(unique_points) < 3:
        print("Warning: Not enough unique points for alpha shape")
        return
    
    #Create alpha shape from alphashape library
    try: 
        alpha_shape = alphashape.alphashape(unique_points, alpha)
        if hasattr(alpha_shape, 'exterior'):
            #Single Polygon 
            boundary_coords = np.array(alpha_shape.exterior.coords)

        elif hasattr(alpha_shape, 'geoms'):
            # Multiple polygons - take the largest one
            largest_poly = max(alpha_shape.geoms, key=lambda x: x.area if hasattr(x, 'area') else 0)
            if hasattr(largest_poly, 'exterior'):
                boundary_coords = np.array(largest_poly.exterior.coords)
            else:
                boundary_coords = np.array([])
        else:
            #Fallback - use convex hull 
            print("WARNING: ALPHA SHAPE FAILED, using convex hull")
            hull = ConvexHull(unique_points)
            boundary_coords = unique_points[hull.vertices]
            boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
            alpha_shape = None
        
       # Determine the alpha value that was actually used
        if alpha is None:
            alpha_optimal = alphashape.optimizealpha(unique_points)
            actual_alpha = alpha_optimal
        else:
            actual_alpha = alpha
            
    except Exception as e:
        print(f"Warning: Alpha shape computation failed: {e}")
        print("Falling back to convex hull")
        
        # Fallback to convex hull
        hull = ConvexHull(unique_points)
        boundary_coords = unique_points[hull.vertices]
        boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
        alpha_shape = None
        actual_alpha = alpha
    
    return alpha_shape, boundary_coords, actual_alpha

def ADS_HelioCart_2_ECISpherical(final_lists, earthEmphemeris_list, time_idxs):
    """
    Convert Helio-Centric Cartesian coordinates to ECI Spherical coordinates. - Use for Evaluation of Propagation
    :param final_lists: List of lists of ADS objects from propagation. Structure: final_lists[time_idx][subdomain]
    :param earthEphemeris_list: Earth ephemeris data at propagation time conversion. Strucutre: (N_times, 6)
    :param time_idxs: 

    :return: List of Lists of ADS object in ECI spherical coordinates. Structure: spherical_final_lists[time_idx][subdomain]
    """

    spherical_final_lists = []

    if time_idxs is None:
        time_idxs = list(range(len(final_lists)))

    for i in range(len(time_idxs)):
        if time_idxs[i] >= len(final_lists):
            print(f"Warning: time_idx {time_idxs[i]} exceeds final_lists length {len(final_lists)}")
            continue

        spherical_subdomains = []

        #Get Earth state at this time
        if i < len(earthEmphemeris_list):
            earth_ecl_state = earthEmphemeris_list[i,:]
        else:
            print(f"Warning: No Earth position for time index {i}, using zeros")
            earth_ecl_state = np.zeros(len(earthEmphemeris_list[0,:]))
        
        #Process each ADS subdomain at this time i 
        for ads_subdomain in final_lists[i]:
            #Extract the DA manifold and domain (6D state vector)
            manifold_da = ads_subdomain.manifold  # Array of 6 DA objects
            domain_da = ads_subdomain.box         # Array of 6 DA objects
            nsplit_history = ads_subdomain.nsplit 

            #Isolate Position and Velocity
            ECI_manifold = time_ref.Helio2ECIJ2000(manifold_da, earth_ecl_state)
            ECI_domain = time_ref.Helio2ECIJ2000(domain_da, earth_ecl_state)

            #Convert to spherical coordinates
            ECI_manifold_spherical = time_ref.CC2obs(ECI_manifold)
            ECI_domain_spherical = time_ref.CC2obs(ECI_domain)

            spherical_ads = ADS(ECI_domain_spherical, nsplit_history, ECI_manifold_spherical)
            spherical_subdomains.append(spherical_ads)
        

        spherical_final_lists.append(spherical_subdomains)
    
    return spherical_final_lists

def ADS_ECICart_2_Topocentric(final_lists: list[list[ADS]], site_ECI_position: np.ndarray):
    """
    Convert ECI Cartesian coordinates to Cartesian Equatorial Topocentric coordinates.
    """
    topocentric_final_lists = []

    for time_idx, subdomains in enumerate(final_lists):
        topocentric_subdomains = []

        for ads_subdomain in subdomains:
            # Extract the DA manifold and domain (6D state vector)
            manifold_da = ads_subdomain.manifold  # Array of 6 DA objects
            domain_da = ads_subdomain.box         # Array of 6 DA objects
            nsplit_history = ads_subdomain.nsplit 

            # Convert ECI Cartesian to Topocentric
            topocentric_manifold = manifold_da - site_ECI_position
            topocentric_domain = domain_da - site_ECI_position

            topocentric_ads = ADS(topocentric_domain, nsplit_history, topocentric_manifold)
            topocentric_subdomains.append(topocentric_ads)


        topocentric_final_lists.append(topocentric_subdomains)

    return topocentric_final_lists

def ADS_Cart_2_Obs(final_lists):
    """
    Convert ECI Cartesian coordinates to Observational coordinates.
    """
    obs_final_lists = []

    for time_idx, subdomains in enumerate(final_lists):
        obs_subdomains = []

        for ads_subdomain in subdomains:
            # Extract the DA manifold and domain (6D state vector)
            manifold_da = ads_subdomain.manifold  # Array of 6 DA objects
            domain_da = ads_subdomain.box         # Array of 6 DA objects
            nsplit_history = ads_subdomain.nsplit 

            # Convert ECI Cartesian to Observational
            obs_manifold = time_ref.CC2obs(manifold_da)
            obs_domain = time_ref.CC2obs(domain_da)

            obs_ads = ADS(obs_domain, nsplit_history, obs_manifold)
            obs_subdomains.append(obs_ads)

        obs_final_lists.append(obs_subdomains)

    return obs_final_lists

def ADS_Helio2GEO(final_lists, earthEphemeris_list, time_idxs):
    """
    Convert Helio-Centric Cartesian coordinates to Geocentric Cartesian coordinates - use to ensure Apophis is inside 
    :param final_lists: List of lists of ADS objects from propagation. Structure: final_lists[time_idx][subdomain]
    :param earthEphemeris_list: Earth ephemeris data at propagation time conversion. Strucutre: (N_times, 6)
    :param time_idxs: 

    :return: List of Lists of ADS object in Geocentric coordinates. Structure: geo_final_lists[time_idx][subdomain]
    """

    geo_final_lists = []

    if time_idxs is None:
        time_idxs = list(range(len(final_lists)))
    
    for i, time_idx in enumerate(time_idxs):
        if time_idx >= len(final_lists):
            print(f"Warning: time_idx {time_idx} exceeds final_lists length {len(final_lists)}")
            continue

        geo_subdomains = []

        #Get Earth state at this time
        if i < len(earthEphemeris_list):
            earth_ecl_state = earthEphemeris_list[:,i]
        else:
            print(f"Warning: No Earth position for time index {i}, using zeros")
            earth_ecl_state = np.zeros(6)
        
        #Process each ADS subdomain at this time i 
        for ads_subdomain in final_lists[time_idx]:
            #Extract the DA manifold and domain (6D state vector)
            manifold_da = ads_subdomain.manifold  # Array of 6 DA objects
            domain_da = ads_subdomain.box         # Array of 6 DA objects
            nsplit_history = ads_subdomain.nsplit 


            #Translate
            geo_manifold = time_ref.ICRS2ECI(manifold_da, earth_ecl_state)
            geo_domain = time_ref.ICRS2ECI(domain_da, earth_ecl_state)


            geo_ads = ADS(geo_domain, nsplit_history, geo_manifold)
            geo_subdomains.append(geo_ads)

        geo_final_lists.append(geo_subdomains)
    
    return geo_final_lists

def calculate_ads_mean_single_time(ads_list):
    """
    Calculate mean state from a list of ADS domains at a single time step
    
    Parameters:
    -----------
    ads_list : list
        List of ADS objects, each containing a manifold
        
    Returns:
    --------
    numpy.ndarray : Mean state vector (6,) for [x, y, z, vx, vy, vz]
    """
    n_domains = len(ads_list)
    n_states = len(ads_list[0].manifold)  # Should be 6 for position/velocity
    
    # Extract constant parts from each domain
    constant_states = np.zeros((n_domains, n_states))
    
    for i, ads_domain in enumerate(ads_list):
        for j in range(n_states):
            if hasattr(ads_domain.manifold[j], 'cons'):
                constant_states[i, j] = ads_domain.manifold[j].center.cons()
            else:
                constant_states[i, j] = ads_domain.manifold[j].center
    # Calculate mean across domains
    mean_state = np.mean(constant_states, axis=0)
    
    return mean_state

def ADS_ECISpherical_2_ICRS(final_lists, earthEphemeris_list, time_idxs):
    """
    Convert ECI Spherical coordinates to ICRS Spherical coordinates - use to ensure Apophis is inside 
    :param final_lists: List of lists of ADS objects from propagation. Structure: final_lists[time_idx][subdomain]
    :param earthEphemeris_list: Earth ephemeris data at propagation time conversion. Strucutre: (N_times, 6)
    :param time_idxs: 

    :return: List of Lists of ADS object in ICRS coordinates. Structure: icrs_final_lists[time_idx][subdomain]
    """

    spherical_final_lists = []

    if time_idxs is None:
        time_idxs = list(range(len(final_lists)))
    
    for i, time_idx in enumerate(time_idxs):
        if time_idx >= len(final_lists):
            print(f"Warning: time_idx {time_idx} exceeds final_lists length {len(final_lists)}")
            continue

        spherical_subdomains = []

        #Get Earth state at this time
        if i < len(earthEphemeris_list):
            earth_ICRS_spherical = earthEphemeris_list[i,:]
        else:
            print(f"Warning: No Earth position for time index {i}, using zeros")
            earth_ICRS_spherical = np.zeros(6)
        
        #Process each ADS subdomain at this time i 
        for ads_subdomain in final_lists[time_idx]:
            #Extract the DA manifold and domain (6D state vector)
            manifold_da = ads_subdomain.manifold  # Array of 6 DA objects
            domain_da = ads_subdomain.box         # Array of 6 DA objects
            nsplit_history = ads_subdomain.nsplit 


            #Translate 
            ICRS_manifold_spherical = time_ref.ECI2ICRS(manifold_da, earth_ICRS_spherical)
            ICRS_domain_spherical = time_ref.ECI2ICRS(domain_da, earth_ICRS_spherical)

            spherical_ads = ADS(ICRS_domain_spherical, nsplit_history, ICRS_manifold_spherical)
            spherical_subdomains.append(spherical_ads)
        

        spherical_final_lists.append(spherical_subdomains)
    
    return spherical_final_lists

def load_Apophis_Ephemeris(start_time, end_time, step='1d', location='500'):
    """
        Load Apophis 'true' Ephemeris data with AstroPy.
        GCRS/ECI Frame

        :param start_time: Start time (e.g. '2029-04-13')
        :param end_time: End time (e.g., '2029-04-14')
        :param location: Observer Location (e.g., '500' = Geocentric), 
        :param step: Time step for ephemeris (e.g., '1d')
        :return: Table with (ICRS) ephemeris data.
                Call columns: eph['name']
                call row: eph[i]
                call cell: eph['name'][i]
    """
    from astroquery.jplhorizons import Horizons

    if isinstance(start_time, str):
        try:
            apophis_id = '99942'
            obj = Horizons(id=apophis_id,
                        location=location,
                        epochs={'start': start_time,
                                'stop' : end_time,
                                'step' : step})

            # Query State: Obtain ICRS Spherical Coordinates
            eph = obj.ephemerides()                 # ICRS Emphermis Coordinates
            vec = obj.vectors()                     # Heliocentric Vectors

            return eph, vec

        except Exception as e:
            print(f"Error loading Apophis ephemeris: {e}")
            return None
    elif isinstance(start_time, Time):    #step contains all the time instances as a list[Astropy Time]
        year, month, day, hour, minute, second = start_time.ymdhms
        start_time = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02.0f}"
        year, month, day, hour, minute, second = end_time.ymdhms
        end_time = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02.0f}"
        step = "1"
        try:
            apophis_id = '99942'
            obj = Horizons(id=apophis_id,
                        location=location,
                        epochs={'start': start_time,
                                'stop' : end_time,
                                'step' : step})

            # Query State: Obtain ICRS Spherical Coordinates
            eph = obj.ephemerides()                 # ICRS Emphermis Coordinates
            vec = obj.vectors()                     # Heliocentric Vectors

            return eph, vec

        except Exception as e:
            print(f"Error loading Apophis ephemeris: {e}")
            return None

        

def points_in_alphashape(points, alpha_shape):
    """
    Assess whether multiple points are within the alphashape perimeter
    
    :param points: Array of points to test, shape (n_points, 2) or (n_points, 3)
    :param alpha_shape: Alpha shape object from alphashape library
    :return: Boolean array indicating which points are within the alpha shape
    """
    from shapely.geometry import Point, MultiPoint
    import shapely.geometry
    import numpy as np
    
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Create Point objects for all points
    test_points = [Point(pt[0], pt[1]) for pt in points]
    
    results = np.zeros(len(test_points), dtype=bool)
    
    # Handle different alpha shape types
    if isinstance(alpha_shape, shapely.geometry.polygon.Polygon):
        # Single polygon case - vectorized
        for i, test_point in enumerate(test_points):
            results[i] = alpha_shape.contains(test_point) or alpha_shape.touches(test_point)
    
    elif hasattr(alpha_shape, 'geoms'):
        # Multiple polygons case
        for i, test_point in enumerate(test_points):
            for geom in alpha_shape.geoms:
                if isinstance(geom, shapely.geometry.polygon.Polygon):
                    if geom.contains(test_point) or geom.touches(test_point):
                        results[i] = True
                        break
    
    elif hasattr(alpha_shape, 'contains'):
        # Generic shapely geometry
        for i, test_point in enumerate(test_points):
            results[i] = alpha_shape.contains(test_point) or alpha_shape.touches(test_point)
    
    else:
        print(f"Warning: Unsupported alpha shape type: {type(alpha_shape)}")
        results.fill(False)
    
    return results

def CADC_Fixed_Polygon(polygon, start_date, end_date, collection=''):
    """
    Find the Images Within a Fixed Polygon area
    :params Polygon: The polygon defining the RAxDEC area of interest
    :params start_date: The start date for the query. Must be in iso Astropy format
    :params end_date: The end date for the query. Must be in iso Astropy format
    """
    from shapely.ops import orient
    #if collection:
    if collection == None:
        collection_clause = '\n'
    else:
        collection_clause = f"AND collection = '{collection}'"

    #polygon decomposition
    polygon = polygon.buffer(0)
    polygon = orient(polygon, sign=1.0)

    polygon_list = list(polygon.boundary.coords)
    #Eliminate overlapping end coordinate
    if polygon_list[0] == polygon_list[-1]:
        polygon_list = polygon_list[:-1]

    coords_interleaved = [c for pair in polygon_list for c in pair]
    polygon_str = ", ".join(f"{v:.15g}" for v in coords_interleaved)
    query_outline = """SELECT {num}
    productID, position_bounds, publisherID
    FROM caom2.Plane AS Plane
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
    WHERE (INTERSECTS( INTERVAL( {mjd_start}, {mjd_end} ), Plane.time_bounds_samples ) = 1
        AND INTERSECTS(Plane.position_bounds, POLYGON('ICRS', {polygon})) = 1
        {collection_clause}
        AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
        ORDER BY time_bounds_lower {order}
    """

    query_params = {
        'num': '',  # Restricts the number of results (empty string returns all)
        'mjd_start': start_date.mjd,
        'mjd_end': end_date.mjd,
        'polygon': polygon_str,     #Must be in degrees
        'collection_clause': collection_clause,
        'order': 'ASC'
    }

    cadc = CADC.Cadc

    results = cadc.exec_sync(query_outline.format(**query_params))      #Results of query

    
    return results

def CADC_Fixed_Polygon_contains(polygon, start_date, end_date, collection=''):
    """
    Find the Images Within a Fixed Polygon area
    :params Polygon: The polygon defining the RAxDEC area of interest
    :params start_date: The start date for the query. Must be in iso Astropy format
    :params end_date: The end date for the query. Must be in iso Astropy format
    """
    from shapely.ops import orient
    #if collection:
    if collection == None:
        collection_clause = '\n'
    else:
        collection_clause = f"AND collection = '{collection}'"

    #polygon decomposition
    polygon = polygon.buffer(0)
    polygon = orient(polygon, sign=1.0)

    polygon_list = list(polygon.boundary.coords)
    #Eliminate overlapping end coordinate
    if polygon_list[0] == polygon_list[-1]:
        polygon_list = polygon_list[:-1]

    coords_interleaved = [c for pair in polygon_list for c in pair]
    polygon_str = ", ".join(f"{v:.15g}" for v in coords_interleaved)
    query_outline = """SELECT {num}
    productID, position_bounds
    FROM caom2.Plane AS Plane
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
    WHERE (INTERSECTS( INTERVAL( {mjd_start}, {mjd_end} ), Plane.time_bounds_samples ) = 1
        AND CONTAINS(Plane.position_bounds, POLYGON('ICRS', {polygon})) = 1
        AND LOWER(Plane.energy_bandpassName) LIKE '{filter}%' 
        {collection_clause}
        AND calibrationLevel >= {cal_level}
        AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
        ORDER BY time_bounds_lower {order}
    """
    query_params = {
        'num': '',  # Restricts the number of results (empty string returns all)
        'mjd_start': start_date.mjd,
        'mjd_end': end_date.mjd,
        'polygon': polygon_str,     #Must be in degrees
        'filter': 'r',
        'collection_clause': collection_clause,
        'cal_level': 2,
        'order': 'ASC'
    }

    cadc = CADC.Cadc

    results = cadc.exec_sync(query_outline.format(**query_params))      #Results of query

    #Only Care about RA, DEC, Time, image number
    
    return results


def CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection=None):
    """
    Find the Images Within a Fixed Polygon area
    :params Polygon: The polygon defining the RAxDEC area of interest
    :params start_date: The start date for the query. Must be in iso Astropy format
    :params end_date: The end date for the query. Must be in iso Astropy format
    """
    from shapely.ops import orient
    #if collection:
    if collection == None:
        collection_clause = '\n'
    else:
        collection_clause = f"AND collection = '{collection}'"

    #polygon decomposition
    polygon = polygon.buffer(0)
    polygon = orient(polygon, sign=1.0)

    polygon_list = list(polygon.boundary.coords)
    #Eliminate overlapping end coordinate
    if polygon_list[0] == polygon_list[-1]:
        polygon_list = polygon_list[:-1]

    coords_interleaved = [c for pair in polygon_list for c in pair]
    polygon_str = ", ".join(f"{v:.15g}" for v in coords_interleaved)

    #add instrument, time interval, 
    query_outline = """SELECT
    productID,
    Observation.instrument_name,     
    position_bounds,
    Plane.time_bounds_lower AS t_min,
    Plane.time_bounds_upper AS t_max
    FROM caom2.Plane AS Plane
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
    WHERE (INTERSECTS( INTERVAL( {mjd_start}, {mjd_end} ), Plane.time_bounds_samples ) = 1
        AND INTERSECTS(Plane.position_bounds, POLYGON('ICRS', {polygon})) = 1
        {collection_clause}
        AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
    """

    query_params = {
        'mjd_start': start_date.mjd,
        'mjd_end': end_date.mjd,
        'polygon': polygon_str,     #Must be in degrees
        'collection_clause': collection_clause,
    }

    cadc = CADC.Cadc

    job = cadc.create_async(query_outline.format(**query_params))      #Results of query
    job = job.run().wait()
    job.raise_if_error()
    result = job.fetch_result().to_table()

    
    return result

def CADC_Example_point(ra, dec, start_date, end_date, collection):
    
    from astroquery.cadc import Cadc
    
    if collection == None:
        collection_clause = '\n'
    else:
        collection_clause = f"AND collection = '{collection}'"

    # Build the query
    query_outline = """SELECT {num}
    productID, position_bounds
    FROM caom2.Plane AS Plane 
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
    WHERE (INTERSECTS( INTERVAL( {mjd_start}, {mjd_end} ), Plane.time_bounds_samples ) = 1 
        AND CONTAINS( POINT('ICRS', {ra}, {dec}), position_bounds ) = 1 
        AND LOWER(Plane.energy_bandpassName) LIKE '{filter}%' 
        {collection_clause}
        AND calibrationLevel >= {cal_level}
        AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
        ORDER BY time_bounds_lower {order}"""

    # Select the parameters for the ADQL query
    query_params = {
        'num': '',  # Restricts the number of results (empty string returns all)
        'mjd_start': start_date.mjd,
        'mjd_end': end_date.mjd,
        'ra': ra,     #Must be in degrees
        'dec': dec,    
        'filter': 'r',
        'collection_clause': collection_clause,
        'cal_level': 2,
        'order': 'ASC'  # Order the results from oldest to newest
    }

    # Instantiate the CADC module
    cadc = Cadc()

    # Run the query and fetch the results
    results = cadc.exec_sync(query_outline.format(**query_params))

    # Select a subset of columns to preview
    columns_subset = [
        'productID', 'collection', 'energy_bandpassName', 'time_bounds_samples',
        'time_bounds_lower', 'time_exposure'
    ]

    print('Total number of results: {}'.format(len(results)))

    # Showing a sample of the results
    print(results[0:5])
    
    return results

def ESO_Fixed_Polygon_async(polygon, start_date, end_date, collection=None):
    """
    Query ESO TAP with the Polygon - This is broken

    """
    import time
    from typing import Iterable, Tuple, Optional
    import pyvo
    import pandas as pd
    import re
    from shapely.ops import orient
    import pyvo
    
    #if collection:
    if collection == None:
        collection_clause = '\n'
    else:
        collection_clause = f"AND collection = '{collection}'"

    #polygon decomposition
    polygon = polygon.buffer(0)
    polygon = orient(polygon, sign=1.0)

    polygon_list = list(polygon.convex_hull.boundary.coords)
    #Eliminate overlapping end coordinate

    if polygon_list[0] == polygon_list[-1]:
        polygon_list = polygon_list[:-1]

    coords_interleaved = [c for pair in polygon_list for c in pair]
    polygon_str = ", ".join(f"{v:.15g}" for v in coords_interleaved)

    #Build Query
    eso_query = """SELECT
    dp_id, 
    instrument, 
    mjd_obs,
    s_region
    FROM dbo.raw AS R
    WHERE INTERSECTS(R.s_region, POLYGON('ICRS', {polygon})) = 1
    AND R.mjd_obs BETWEEN {mjd_start} AND {mjd_end}
    """
    query_params ={
        'polygon': polygon_str,
        'mjd_start': start_date.mjd,
        'mjd_end': end_date.mjd,
    }

    #Query TAP
    query_full = eso_query.format(**query_params)
    svc = pyvo.dal.TAPService("https://archive.eso.org/tap_obs")
    job = svc.submit_job(query_full, language="ADQL")
    job.run(); job.wait()
    # Short-poll loop (avoids WAIT=-1 long-poll)
    eso_tbl = job.fetch_result().to_table()

    #local filtering: eso_tbl holds the convex_hull boundary coordinates of the real geometry. These are always the largest
    # We need to filter out the false positives
    float_pat = r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    pos_re = re.compile(r'POSITION\s+\w+\s+' + float_pat + r'\s+' + float_pat, re.IGNORECASE)

    filtered_rows = []
    for row in eso_tbl:
        s_region = str(row['s_region'])
        match = pos_re.search(s_region)

        if match:
            ra_str, dec_str = match.groups()
            try:
                ra = float(ra_str)
                dec = float(dec_str)
                
                # Create a point and check if it's inside the original polygon (not convex hull)
                from shapely.geometry import Point
                point = Point(ra, dec)
                
                if polygon.contains(point) or polygon.touches(point):
                    filtered_rows.append(row)
            except ValueError:
                # Skip rows where RA/DEC can't be parsed as floats
                continue

    # Convert filtered rows back to astropy table
    if filtered_rows:
        eso_tbl = Table(rows=filtered_rows, names=eso_tbl.colnames)
    else:
        # Return empty table with same structure if no matches
        eso_tbl = eso_tbl[:0]


    return eso_tbl

def CADC_ESO_combine(cadc_tbl: Table, eso_tbl: Table) -> Table:    
    """
    Concatenate ESO CADC results.
    :param cadc_tbl: The CADC table to concatenate.
    :param eso_tbl: The ESO table to concatenate.
    """
    cadc = cadc_tbl.copy()
    eso  = eso_tbl.copy()

    #Harmonize Table labels
    # Rename to productID
    if "productID" not in eso.colnames and "dp_id" in eso.colnames:
        eso.rename_column("dp_id", "productID")
    if "publisherID" not in eso.colnames and "obs_publisher_did" in eso.colnames:
        eso.rename_column("obs_publisher_did", "publisherID")

    # Footprint - rename to "Position_bounds"
    if "position_bounds" not in eso.colnames and "s_region" in eso.colnames:
        eso.rename_column("s_region", "position_bounds")
    if "position_bounds" not in cadc.colnames and "s_region" in cadc.colnames:
        cadc.rename_column("s_region", "position_bounds")

    # Instrument name - match cadc
    if "instrument_name" not in eso.colnames and "instrument" in eso.colnames:
        eso.rename_column("instrument", "instrument_name")

    # Time columns → t_min / t_max
    if "t_min" not in cadc.colnames and "time_bounds_lower" in cadc.colnames:
        cadc.rename_column("time_bounds_lower", "t_min")    #rename time_bounds_lower to t_min
    if "t_max" not in cadc.colnames and "time_bounds_upper" in cadc.colnames:
        cadc.rename_column("time_bounds_upper", "t_max")    #rename time_bounds_upper to t_max

    # ESO raw often has only mjd_obs; map to both t_min/t_max for stacking
    if "t_min" not in eso.colnames and "mjd_obs" in eso.colnames:
        eso["t_min"] = eso["mjd_obs"]
    if "t_max" not in eso.colnames and "mjd_obs" in eso.colnames:
        eso["t_max"] = eso["mjd_obs"]
        #delete eso["mjd_obs"]
        del eso["mjd_obs"]
    
    

    # Make columns unicode
    def _make_unicode(tbl: Table, cols):
        for c in cols:
            if c in tbl.colnames:
                tbl[c] = tbl[c].astype("U")  # unicode string dtype
    
    _make_unicode(cadc, ["productID", "publisherID", "position_bounds", "collection", "instrument_name"])
    _make_unicode(eso,  ["productID", "publisherID", "position_bounds", "collection", "instrument_name"])

    # create a canonical 'ProductID' column which is case sensitive
    for tbl in (cadc, eso):
        if "ProductID" not in tbl.colnames and "productID" in tbl.colnames:
            tbl["ProductID"] = tbl["productID"].astype("U")
    
    ## Provenance tag
    cadc["tap_source"] = "CADC"
    eso["tap_source"]  = "ESO"

    #Vertical Stack 
    combined = vstack([cadc, eso], join_type="outer", metadata_conflicts="silent")

    #Sort columns by minimum time 
    if "t_min" in combined.colnames:
        combined.sort("t_min")

    return combined

def SSOIS_Query_Apophis(error, start_date, end_date, interval_mjd, search=True):
    """
    Conduct SSOIS query for the asteroid Apophis.
    Options between:
    - Positional uncertainty
    - No positional uncertainty

    :param error: The positional uncertainty in arcseconds. MPC assumes a box independant of time.
    :param start_date: The start date for the query ("YYYY+MM+DD") format - rounded to the nearest day
    :param end_date: The end date for the query ("YYYY+MM+DD") format - rounded to the nearest day (shortest interval = 1 day)
    :param interval_mjd: The end time for the State propagator / ADQL search. 
    """
    from datetime import datetime
    CADC_name = 'Apophis'
    eunits = 'arcseconds'

    #mjd conversion to "YYYY+MM+DD"
    start_date_isot = start_date.isot
    end_date_isot = end_date.isot

    dt = datetime.fromisoformat(start_date_isot)
    start_date_yyyymmdd = dt.strftime('%Y+%m+%d')  # '2004+12+28'
    dt = datetime.fromisoformat(end_date_isot)
    end_date_yyyymmdd = dt.strftime('%Y+%m+%d')  # '2004+12+28'

    assert isinstance(start_date_yyyymmdd, str), "Start date must be a string in the format 'YYYY+MM+DD'"
    assert isinstance(end_date_yyyymmdd, str), "End date must be a string in the format 'YYYY+MM+DD'"

    print('Conducting SSOIS Query...')

    baseurl = 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl'
    url = f"{baseurl}?lang=en;object={CADC_name};search=bynameCADC;epoch1={start_date_yyyymmdd};epoch2={end_date_yyyymmdd};eellipse={error};eunits={eunits};extres=no;xyres=no;format=tsv"

    #Convert to DataFrame for ease 
    data_table = pd.read_csv(url, sep='\t')

    #Obtain the ra and dec of the image closest to the initial time.
    if search:
        if not data_table.empty:
            time_diff = data_table['MJD'] - start_date.mjd
            within_interval = time_diff.abs() <= interval_mjd.mjd

            filtered_images = data_table[within_interval]
            if not filtered_images.empty:
                print(f"Found {len(filtered_images)} images within interval")
                print(f"Time range: MJD {filtered_images['MJD'].min():.3f} to {filtered_images['MJD'].max():.3f}")
                return filtered_images
            else: 
                print("ERROR: No images found within the specified time interval, try increasing the propagation timestep")
                return None
        else:
            print("ERROR: No valid images found.")
            return None
    else: 
        return data_table

def image_match(ssois_table, adql_table, case_insensitive=True, return_astropy=False, adql_id_col="productID", ssois_id_col="Image"):
    """
    Match SSOIS and ADQL images.
    - Match rows with the same 'product ID' (ADQL_table) and the 'Image' cells (ssois_table)

    :param ssois_table: The SSOIS 'truth' table of Apophis
    :param ADQL_table: The ADQL fixed area table of observations
    :param ssois_id_col: The column name in the SSOIS table that contains the image IDs
    :param adql_id_col: The column name in the ADQL table that contains the image IDs
    :param relaxed: Whether to use relaxed matching (root-level matching) or strict matching
    """

    # Convert to pandas
    ssois_df = ssois_table.to_pandas() if isinstance(ssois_table, Table) else ssois_table.copy()
    adql_df  = adql_table.to_pandas()  if isinstance(adql_table,  Table) else adql_table.copy()

    # Pick ADQL ID column
    if adql_id_col is None:
        for c in ("productID", "ProductID", "dp_id"):
            if c in adql_df.columns:
                adql_id_col = c
                break
        if adql_id_col is None:
            raise ValueError("ADQL table must have 'productID' (or 'ProductID'/'dp_id').")

    # Basic cleanup (trim; optional case-fold)
    left  = ssois_df[ssois_id_col].astype(str).str.strip()
    right = adql_df[adql_id_col].astype(str).str.strip()
    
    if case_insensitive:
        left  = left.str.lower()
        right = right.str.lower()

    ssois_df = ssois_df.assign(__key=left)
    adql_df  = adql_df.assign(__key=right)

    # Strict inner join on the exact key
    matched = ssois_df.merge(adql_df, on="__key", how="inner",
                             suffixes=("_ssois", "_adql"))
    matched.drop(columns=["__key"], inplace=True)

    if return_astropy:
        return Table.from_pandas(matched)
    
    return matched


def image_metrics(ADQL, matched_images, truth_images):
    """
    Calculate True Positive, False Positive, and False Negative rates for image matching.
    :param ADQL: The predicted image table of observations of fixed region, fixed time Astropy Table
    :param matched_images: The matched images Astropy Table
    :param truth_images: The ground truth images or control image table Astropy Table
    :return: A dictionary with the metrics
        Total_Images : N_predicted_images
        TP: True Positives
        FP: False Positives
        FN: False Negatives
        precision: Precision    Measures False Alarms
        recall: Recall          Measures sensitivity
        F1: F1
    """
    N_predicted_images = len(ADQL)
    true_positives = len(matched_images)

    false_positives = N_predicted_images - true_positives
    false_negatives = len(truth_images) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    f1_score = 2* (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Total_Images" : N_predicted_images,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score
    }

def full_query(polygon, start_date, end_date):
    """
    Execute the full query: Table Generation -> Image Matching -> Image Metrics
    """
    # Step 1: Table Generation
    CADC = CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"CADC obtained length: {len(CADC)} ")
    
    ESO = ESO_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"ESO obtained length: {len(ESO)} ")
    
    AQDL_combined = CADC_ESO_combine(CADC, ESO)

    #Obtain SSOIS Control
    error = 0
    if abs(start_date.mjd.value - end_date.mjd.value) < 1: 
        SSOIS_date = start_date + 1 * u.day
    else:
        SSOIS_date = end_date
    
    SSOIS = SSOIS_Query_Apophis(error, start_date, SSOIS_date, end_date, search=False)
    print(f"SSOIS obtained length: {len(SSOIS)} ")
    
    assert SSOIS is not None
    SSOIS_tbl = Table.from_pandas(SSOIS)

    # Step 2: Image Matching
    matches = image_match(SSOIS_tbl, AQDL_combined, case_insensitive=True, return_astropy=True)
    print(f"Image matches found: {len(matches)} ")    
    
    # Step 3: Image Metrics
    metrics = image_metrics(AQDL_combined, matches, SSOIS_tbl)
    print(f"Image metrics: {metrics}")

    return metrics

def CADC_sort_CCW(RA, DEC):
    """
        Sort the RA, DEC coordinates so they're CCW 
        
        :param RA: Array-like of Right Ascension coordinates (radians)
        :param DEC: Array-like of Declination coordinates (radians)
        :return: Tuple of (sorted_RA, sorted_DEC) in counter-clockwise order
    """
    ra = np.array(RA)
    dec = np.array(DEC)

    # Calculate centroid
    cx, cy = ra.mean(), dec.mean()

    # Calculate angles from centroid to each point
    angles = np.arctan2(dec - cy, ra - cx)
    
    # Sort indices by angle (counter-clockwise)
    idx = np.argsort(angles)
    
    return ra[idx], dec[idx]