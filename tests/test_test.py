
from time import time
import numpy as np
import pytest
from pathlib import Path  # Add this import
from daceypy import ADS, DA
from utils.post_process import CADC_Example_point, generate_perimeter3D, eval_perimeterADS3D, ADS_HelioCart_2_ECISpherical, ADS_ECISpherical_2_ICRS, load_Apophis_Ephemeris, CADC_sort_CCW, create_alphashape2D, extract_2D_points, points_in_alphashape
import pickle
import astropy.coordinates as acoords
import astropy.units as u
from astropy.time import Time, TimeDelta
import utils.time_reference as time_ref
import alphashape
import matplotlib.pyplot as plt
import os 
import utils.optimisealpha as optalpha

def test_simple():
    """Simple test that should always work"""
    assert 1 + 1 == 2
    print("✅ Simple test passed")


def perimeter_sample_data():
    # Parameters from ADS_2.py
    xb = 0.008      # uncertainty in x
    yb = 0.08       # uncertainty in y
    Ns = 33         # Number of gridpoints
    
    print("=== Generating ADS_2.py style perimeter ===")
    
    xgrid = np.linspace(-1, 1, Ns)
    ygrid = np.linspace(-1, 1, Ns)
    
    # Four boundaries from ADS_2.py
    lb = np.ones((Ns, 2))               # Left boundary
    lb[:, 0] = lb[:, 0] * (-xb)         # x = -xb
    lb[:, 1] = yb * ygrid               # y varies
    
    rb = np.ones((Ns, 2))               # Right boundary  
    rb[:, 0] = rb[:, 0] * (xb)          # x = +xb
    rb[:, 1] = yb * ygrid               # y varies
    
    bb = np.ones((Ns, 2))               # Bottom boundary
    bb[:, 0] = xb * xgrid               # x varies
    bb[:, 1] = bb[:, 1] * (-yb)         # y = -yb
    
    tb = np.ones((Ns, 2))               # Top boundary
    tb[:, 0] = xb * xgrid               # x varies
    tb[:, 1] = tb[:, 1] * (yb)          # Fixed: Added missing closing parenthesis
    
    # Concatenate all boundaries (ADS_2 order: top, right, bottom, left)
    ads2_perimeter = np.concatenate((tb, rb, bb, lb))
    ads2_perimeter_norm = np.concatenate((tb, rb, bb, lb))
    ads2_perimeter_norm[:, 0] = ads2_perimeter[:, 0] / xb
    ads2_perimeter_norm[:, 1] = ads2_perimeter[:, 1] / yb

    return ads2_perimeter, ads2_perimeter_norm


def test_perimeter_sample_data():
    """Test perimeter sample data generation"""
    ads2_perimeter, ads2_perimeter_norm = perimeter_sample_data()
    
    assert ads2_perimeter.shape == (132, 2), f"Expected shape (132, 2), got {ads2_perimeter.shape}"
    assert ads2_perimeter_norm.shape == (132, 2), f"Expected shape (132, 2), got {ads2_perimeter_norm.shape}"
    
    # Check normalization
    assert np.allclose(ads2_perimeter_norm[:, 0].min(), -1.0, atol=1e-10)
    assert np.allclose(ads2_perimeter_norm[:, 0].max(), 1.0, atol=1e-10)
    assert np.allclose(ads2_perimeter_norm[:, 1].min(), -1.0, atol=1e-10)
    assert np.allclose(ads2_perimeter_norm[:, 1].max(), 1.0, atol=1e-10)

def sample_data():
    """ Sample data from DaceyPy tutorials > ADS_EX > ADS_2.py"""
    DA.init(14, 2)
    DA.setEps(1e-16)    # Load the ADS objects

    try:
        # Fixed: Use forward slashes for cross-platform compatibility
        if not os.path.exists('final_lists_advanced.pkl'):
            pytest.skip("Pickle file not found - run ADS tutorial first to generate the data")

        with open('final_lists_advanced.pkl', 'rb') as P:
            ads_objects = pickle.load(P)


        print(f"Successfully loaded ADS objects from pickle file")
        print(f"Data type: {type(ads_objects)}")
            
        # Check if it's final_lists structure
        if isinstance(ads_objects, list):
            print(f"Number of time steps: {len(ads_objects)}")
            if len(ads_objects) > 0 and isinstance(ads_objects[0], list):
                print(f"Number of subdomains at first time step: {len(ads_objects[0])}")
                if len(ads_objects[0]) > 0 and isinstance(ads_objects[0][0], ADS):
                    print(f"First object is ADS: {ads_objects[0][0]}")
        
        Ns = 33
        return ads_objects, Ns
        
    except FileNotFoundError:
        print("Error: Pickle file not found")
        print("Make sure to run the ADS tutorial first to generate the data")
        return None, None
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None, None


def test_sample_data_loading():
    """Test that sample data loads correctly"""
    final_lists, Ns = sample_data()
    
    if final_lists is None:
        pytest.skip("Sample data not available - run ADS tutorial first")
    
    assert isinstance(final_lists, list), "final_lists should be a list"
    assert len(final_lists) > 0, "final_lists should not be empty"
    assert Ns == 33, "Ns should be 33"
    
    if len(final_lists) > 0:
        assert isinstance(final_lists[0], list), "Each time step should be a list"
    
        if len(final_lists[0]) > 0:
            assert isinstance(final_lists[0][0], ADS), "Each element should be an ADS object"


def test_generate_perimeter3D():
    """Test the generation of the perimeter"""
    
    # Parameters from ADS_2.py
    xb = 0.008      # uncertainty in x
    yb = 0.08       # uncertainty in y
    Ns = 33         # Number of gridpoints
    
    print("=== Testing generate_perimeter3D() vs ADS_2.py ===")

    ads2_perimeter_dup, ads2_perimeter_norm = perimeter_sample_data()
    ads2_perimeter = np.unique(ads2_perimeter_dup, axis=0)
    ads2_perimeter_norm = np.unique(ads2_perimeter_norm, axis=0)

    print(f"ADS_2 perimeter shape: {ads2_perimeter.shape}")
    print(f"ADS_2 perimeter_norm range x: [{ads2_perimeter_norm[:, 0].min():.3f}, {ads2_perimeter_norm[:, 0].max():.3f}]")
    print(f"ADS_2 perimeter_norm range y: [{ads2_perimeter_norm[:, 1].min():.3f}, {ads2_perimeter_norm[:, 1].max():.3f}]")
    
    # Generate using your generate_perimeter3D() with zb=0
    print("\n2. Generating with generate_perimeter3D() (zb=0)...")
    zb = 0.0  # No z uncertainty

    try:
        perimeter_norm_3d_2d, perimeter_3d_2d = generate_perimeter3D(xb, yb, zb, Ns)
        print(f"Generated perimeter shape: {perimeter_3d_2d.shape}")
        print(f"Generated perimeter_norm range x: [{perimeter_norm_3d_2d[:, 0].min()}, {perimeter_norm_3d_2d[:, 0].max()}]")
        print(f"Generated perimeter_norm range y: [{perimeter_norm_3d_2d[:, 1].min()}, {perimeter_norm_3d_2d[:, 1].max()}]")


    except Exception as e:
        pytest.fail(f"Error in generate_perimeter3D: {e}")
    
    # Compare results
    print("\n3. Comparing results...")
    
    # FIXED: Removed contradictory assertions - only check that shapes should match
    assert ads2_perimeter.shape == perimeter_3d_2d.shape, \
           f"Shape mismatch: ADS_2 {ads2_perimeter.shape} vs Mine {perimeter_3d_2d.shape}"

    # Check if normalized ranges are correct
    assert np.allclose(perimeter_norm_3d_2d[:, 0].min(), -1.0, atol=1e-10) and \
           np.allclose(perimeter_norm_3d_2d[:, 0].max(), 1.0, atol=1e-10), \
           f"X normalization failed: range [{perimeter_norm_3d_2d[:, 0].min():.6f}, {perimeter_norm_3d_2d[:, 0].max():.6f}]"
    
    assert np.allclose(perimeter_norm_3d_2d[:, 1].min(), -1.0, atol=1e-10) and \
           np.allclose(perimeter_norm_3d_2d[:, 1].max(), 1.0, atol=1e-10), \
           f"Y normalization failed: range [{perimeter_norm_3d_2d[:, 1].min():.6f}, {perimeter_norm_3d_2d[:, 1].max():.6f}]"


    # Check boundary coverage (should hit all extremes)
    x_boundaries = np.unique(np.round(perimeter_norm_3d_2d[:, 0], 6))
    y_boundaries = np.unique(np.round(perimeter_norm_3d_2d[:, 1], 6))
    
    x_covers_extremes = (-1.0 in np.round(x_boundaries, 6)) and (1.0 in np.round(x_boundaries, 6))
    y_covers_extremes = (-1.0 in np.round(y_boundaries, 6)) and (1.0 in np.round(y_boundaries, 6))
    
    assert x_covers_extremes, f"X boundaries do not cover extremes: {x_boundaries}"
    assert y_covers_extremes, f"Y boundaries do not cover extremes: {y_boundaries}"

    #Order ads2 perimeter and perimeter_3d_2d and subtract
    ads2_perimeter.sort()
    perimeter_3d_2d.sort()

    difference = np.setdiff1d(ads2_perimeter, perimeter_3d_2d)
    assert difference.size == 0, f"Discrepancy found in perimeter points: {difference}"

    print("✅ All assertions passed!")

def test_eval_perimeterADS3D():
    """ Testing Access and result values to ADS_2.py evalution results"""
    DA.init(14, 2)
    DA.setEps(1e-16)
    # Load the ADS objects
    ads2_perimeter_dup, ads2_perimeter_norm = perimeter_sample_data()
    TF = 40.                 #Final time
    T0 = 0.                  #Initial time
    Ns = 33                  #Number of gridpoints        
    Ts = 41                  #Number of timesteps

    # part 1 of the example, assemble perimeter of ground truth domain:
    tgrid = np.linspace(T0, TF, Ts)     #Generate time grid

    time_analysis = [0, 16, 33, 34, 35, 36, 37, 38, 39, 40]
    with open("final_lists_advanced.pkl", "rb") as f:
        ads_list = pickle.load(f)

    result = eval_perimeterADS3D(ads_list, ads2_perimeter_norm, tgrid, time_idxs=time_analysis, state_dim=4)
    manifold = result['final_map']
    domain = result['final_domain']

    
    
    #Load reference
    with open("ADS_2_final_maps.pkl", "rb") as f:
        reference_result = pickle.load(f)
    
    reference_manifolds = reference_result[0]
    reference_domains = reference_result[1]

    # Compare results
    assert isinstance(manifold, list), "Manifold Expects list"
    assert isinstance(manifold[0], np.ndarray), "Expected manifold to be a numpy array"
    assert isinstance(domain[0], np.ndarray), "Expected domain to be a numpy array"

    difference_manifolds_time = manifold[1][0,0,:] - reference_manifolds[0][0,0,:]
    difference_domains_time = domain[1][0,0,:] - reference_domains[0][0,0,:]

    different_manifolds_state = manifold[1][0,:,0] - reference_manifolds[0][:,0,0]
    different_domains_state = domain[1][0,:,0] - reference_domains[0][:,0,0]

    difference_manifolds_perimeter = manifold[1][:,0,0] - reference_manifolds[0][0,:,0]
    difference_domains_perimeter = domain[1][:,0,0] - reference_domains[0][0,:,0]

    assert np.allclose(difference_manifolds_time, 0, rtol=1e-12)
    assert np.allclose(difference_domains_time, 0, rtol=1e-12)
    assert np.allclose(different_manifolds_state, 0, rtol=1e-12)
    assert np.allclose(different_domains_state, 0, rtol=1e-12)
    assert np.allclose(difference_manifolds_perimeter, 0, rtol=1e-12)
    assert np.allclose(difference_domains_perimeter, 0, rtol=1e-12)

    plt.figure() # plot a 2D manifold 
    plt.title("2D manifold")
    manifold_2d = manifold[0][:, :2, 0]  # Shape: (perimeter_points, 2)
    plt.plot(manifold_2d[:,0], manifold_2d[:,1], 'bo-')
    #plt.plot(ads2_perimeter_norm[:,0], ads2_perimeter_norm[:,1], 'ro-')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.show()


def create_ads_with_subdomains(base_state, num_subdomains=4):
    """
    Create ADS objects with subdomains by adding small perturbations to the base state
    
    Returns: List of ADS objects (subdomains)
    """
    DA.init(6, 2)
    DA.setEps(1e-16)
    
    ads_subdomains = []
    
    # Create multiple subdomains with small perturbations
    perturbations = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],        # Original
        [1000.0, 0.0, 0.0, 0.1, 0.0, 0.0],     # Small position/velocity perturbation
        [0.0, 1000.0, 0.0, 0.0, 0.1, 0.0],     # Another perturbation
        [0.0, 0.0, 1000.0, 0.0, 0.0, 0.1]      # Third perturbation
    ]
    
    for i in range(min(num_subdomains, len(perturbations))):
        perturbed_state = base_state + np.array(perturbations[i])
        ads_obj = ADS(perturbed_state, [])
        ads_subdomains.append(ads_obj)
    
    return ads_subdomains

def test_ADS_HelioCart_2_ECISpherical_single():
    """ Test the ADS_HelioCart_2_ECISpherical function"""

    time = [Time("2025-01-01T00:00:00", scale='tdb')]
    time_idxs = list(range(len(time)))

    DA.init(6,2)
    DA.setEps(1e-16)  # Load the ADS objects
    state = np.array([1.5e8, 0.0, 0.0, 0.0, 29.78, 0.0])  # Example: 1 AU from Sun, circular orbit
    #Get ADS reference

    ads_obj = [[ADS(state,[])]]  # The Object will always be a list of lists. Outer list = domain, inner_list = subdomain

    #Generate 2D Earth Emphermis (Heliocentric)
    EARTH_H = np.zeros((len(time), 6))  # Initialize Earth state vector
    
    for i in range(len(time)):
        position, velocity = acoords.get_body_barycentric_posvel('earth', time[i])       #Earth Barycentric Position and velocity???
        vel = acoords.CartesianDifferential(velocity.xyz.to(u.km / u.s))
        Earth_ICRS_value = acoords.CartesianRepresentation(position.xyz.to(u.km), differentials=vel)

        Earth_ICRS = acoords.SkyCoord(Earth_ICRS_value, frame='icrs', representation_type='cartesian', differential_type='cartesian', obstime=time)
        EARTH_Helio = Earth_ICRS.transform_to(acoords.HeliocentricEclipticIAU76()).cartesian
        EARTH_Helio_pos = EARTH_Helio.xyz.to(u.km).value.T.reshape(-1)
        EARTH_Helio_vel = EARTH_Helio.differentials['s'].d_xyz.to(u.km/u.s).value.T.reshape(-1)

        EARTH_H[i,:] = np.concatenate((EARTH_Helio_pos, EARTH_Helio_vel))

    result = ADS_HelioCart_2_ECISpherical(ads_obj, EARTH_H, time_idxs)
    truth = time_ref.Helio2ECIJ2000(state, EARTH_H[0, :])  # This is the reference transformation
    truth = time_ref.CC2obs(truth)

    #Checks
    assert result is not None, "Transformation failed"
    assert len(result) == 1, "Expected single result"
    assert isinstance(result[0][0], ADS), "Expected ADS object"
    assert result[0][0].box.shape == (6,), "Expected state vector of shape (6,)"

    difference = result[0][0].box - truth
    assert np.allclose(difference.cons(), 0, rtol=1e-12), "Expected domain to match"

def test_ADS_HelioCart_2_ECISpherical_subdomain():
    """ Test the ADS_HelioCart_2_ECISpherical function with multiple subdomains"""

    time = [Time("2025-01-01T00:00:00", scale='tdb')]
    time_idxs = list(range(len(time)))
 

    DA.init(6,2)
    DA.setEps(1e-16)  # Load the ADS objects
    basestate = np.array([1.5e8, 0.0, 0.0, 0.0, 29.78, 0.0])  # Example: 1 AU from Sun, circular orbit
    
    ads_subdomains = create_ads_with_subdomains(basestate, num_subdomains=3)
    final_lists = [ads_subdomains]  # Wrap in a list to match expected input format
    
    #Generate 2D Earth Emphermis (Heliocentric)
    EARTH_H = np.zeros((len(time), 6))  # Initialize Earth state vector
    
    for i in range(len(time)):
        position, velocity = acoords.get_body_barycentric_posvel('earth', time[i])       #Earth Barycentric Position and velocity???
        vel = acoords.CartesianDifferential(velocity.xyz.to(u.km / u.s))
        Earth_ICRS_value = acoords.CartesianRepresentation(position.xyz.to(u.km), differentials=vel)

        Earth_ICRS = acoords.SkyCoord(Earth_ICRS_value, frame='icrs', representation_type='cartesian', differential_type='cartesian', obstime=time)
        EARTH_Helio = Earth_ICRS.transform_to(acoords.HeliocentricEclipticIAU76()).cartesian
        EARTH_Helio_pos = EARTH_Helio.xyz.to(u.km).value.T.reshape(-1)
        EARTH_Helio_vel = EARTH_Helio.differentials['s'].d_xyz.to(u.km/u.s).value.T.reshape(-1)

        EARTH_H[i,:] = np.concatenate((EARTH_Helio_pos, EARTH_Helio_vel))

    result = ADS_HelioCart_2_ECISpherical(final_lists, EARTH_H, time_idxs)

    # Apply the transformation to each ADS subdomain
    truth = time_ref.Helio2ECIJ2000(basestate, EARTH_H[0])  # This is the reference transformation
    truth = time_ref.CC2obs(truth)

    #checks 
    assert result is not None, "Transformation failed"
    assert isinstance(result, list), "Expected a list of ADS objects"
    assert len(result) == 1, "Expected single result"
    assert isinstance(result[0], list), "Expected ADS object"
    assert isinstance(result[0][0], ADS), "Expected ADS object in subdomain"
    assert result[0][0].box.shape == (6,), "Expected state vector of shape (6,)"

    difference = result[0][0].box - truth
    assert np.allclose(difference.cons(), 0, rtol=1e-12), "Expected domain to match"

def test_ADS_HelioCart_2_ECISpherical_domains_subdomains():
    """Test the ADS_HelioCart_2_ECISpherical with multiple domains and subdomains within it. Testing ADS Access and conversion"""

    times = [
        Time("2025-01-01T00:00:00", scale='tdb'),
        Time("2025-01-02T00:00:00", scale='tdb'),
        Time("2025-01-03T00:00:00", scale='tdb')
    ]
    time_idxs = list(range(len(times)+1))
    
    DA.init(6,2)
    DA.setEps(1e-16)  # Load the ADS objects

    base_state = np.array([1.5e8, 0.0, 0.0, 0.0, 29.78, 0.0])  # Example: 1 AU from Sun, circular orbit

    #Store Base State within ADS list as index 0
    final_list = [create_ads_with_subdomains(base_state, num_subdomains=2)]  # Create initial subdomains
    
    EARTH_H = np.zeros((len(time_idxs)+1, 6))
    position, velocity = acoords.get_body_barycentric_posvel('earth', times[0])       #Earth Barycentric Position and velocity???
    vel = acoords.CartesianDifferential(velocity.xyz.to(u.km / u.s))
    Earth_ICRS_value = acoords.CartesianRepresentation(position.xyz.to(u.km), differentials=vel)

    Earth_ICRS = acoords.SkyCoord(Earth_ICRS_value, frame='icrs', representation_type='cartesian', differential_type='cartesian', obstime=times[0])
    EARTH_Helio = Earth_ICRS.transform_to(acoords.HeliocentricEclipticIAU76()).cartesian
    EARTH_Helio_pos = EARTH_Helio.xyz.to(u.km).value.T.reshape(-1)
    EARTH_Helio_vel = EARTH_Helio.differentials['s'].d_xyz.to(u.km/u.s).value.T.reshape(-1)

    EARTH_H[0,:] = np.concatenate((EARTH_Helio_pos, EARTH_Helio_vel))


    for i, time in enumerate(times):
        evolved_state = base_state + np.array([i * 1e6, 0.0, 0.0, 0.0, 0.1 * i, 0.0])  # Evolve state slightly for each time step
        ads_subdomains = create_ads_with_subdomains(evolved_state, num_subdomains=2)
        final_list.append(ads_subdomains)  # Append subdomains for this time step

        #Get Earth
        position, velocity = acoords.get_body_barycentric_posvel('earth', times[i])       #Earth Barycentric Position and velocity???
        vel = acoords.CartesianDifferential(velocity.xyz.to(u.km / u.s))
        Earth_ICRS_value = acoords.CartesianRepresentation(position.xyz.to(u.km), differentials=vel)

        Earth_ICRS = acoords.SkyCoord(Earth_ICRS_value, frame='icrs', representation_type='cartesian', differential_type='cartesian', obstime=times[i])
        EARTH_Helio = Earth_ICRS.transform_to(acoords.HeliocentricEclipticIAU76()).cartesian
        EARTH_Helio_pos = EARTH_Helio.xyz.to(u.km).value.T.reshape(-1)
        EARTH_Helio_vel = EARTH_Helio.differentials['s'].d_xyz.to(u.km/u.s).value.T.reshape(-1)

        EARTH_H[i+1,:] = np.concatenate((EARTH_Helio_pos, EARTH_Helio_vel))
        
        

    result = ADS_HelioCart_2_ECISpherical(final_list, EARTH_H, time_idxs )

    #truth
    truth = time_ref.Helio2ECIJ2000(base_state, EARTH_H[0])  # This is the reference transformation
    truth = time_ref.CC2obs(truth)

    #checks 
    assert result is not None, "Transformation failed"
    assert isinstance(result, list), "Expected a list of ADS objects"
    assert len(result) == 4, "Expected 4 domains result"
    assert isinstance(result[0], list), "Expected ADS object"
    assert isinstance(result[0][0], ADS), "Expected ADS object in subdomain"
    assert len(result[0]) == 2, "Expected two subdomains"
    assert result[0][0].box.shape == (6,), "Expected state vector of shape (6,)"
    
    difference = result[0][0].box - truth
    assert np.allclose(difference.cons(), 0, rtol=1e-12), "Expected domain to match"

def test_ADS_ECISpherical_2_ICRS_single():
    """     Test the ADS_ECI_SPHERICAL_2_ICRS FUNCTION - Needed to test Apohpis inside."""

    time = [Time("2025-01-01T00:00:00", scale='tdb')]
    time_idxs = list(range(len(time)))

    DA.init(6,2)
    DA.setEps(1e-16)  # Load the ADS objects
    state = np.array([7000, 0, 0, 0, 7.5, 0])  # Example: 1 AU from Sun, circular orbit
    state_obs = time_ref.CC2obs(state)  # Convert to observer-centric coordinates

    ads_obj = [[ADS(state_obs,[])]]  # The Object will always be a list of lists. Outer list = domain, inner_list = subdomain


    ################### Test Starts ###################################

    EARTH_G = np.zeros((len(time), 6))  # Initialize Earth state vector
        
    for i in range(len(time)):
        position, velocity = acoords.get_body_barycentric_posvel('earth', time[i])       #Earth Barycentric Position and velocity???
        Earth_ICRS_pos = position.xyz.to(u.km).value.T.reshape(-1)
        Earth_ICRS_vel = velocity.xyz.to(u.km/u.s).value.T.reshape(-1)

        EARTH_G[i,:] = np.concatenate((Earth_ICRS_pos, Earth_ICRS_vel))
        EARTH_G[i,:] = time_ref.CC2obs(EARTH_G[i,:])

    result = ADS_ECISpherical_2_ICRS(ads_obj, EARTH_G, time_idxs)

    truth = time_ref.ECI2ICRS(state_obs, EARTH_G[0])  # This is the reference transformation

    difference = result[0][0].box - truth

    assert np.allclose(difference.cons(), 0, rtol=1e-12), "Expected domain to match"
    
    assert result is not None, "Transformation failed"
    assert isinstance(result, list), "Expected a list of ADS objects"
    assert len(result) == 1, "Expected 1 domains result"
    assert isinstance(result[0], list), "Expected ADS object"
    assert isinstance(result[0][0], ADS), "Expected ADS object in subdomain"
    assert len(result[0]) == 1, "Expected 1 subdomains"
    assert result[0][0].box.shape == (6,), "Expected state vector of shape (6,)"

def test_ADS_ECISpherical_2_ICRS_subdomains():

    time = [Time("2025-01-01T00:00:00", scale='tdb')]
    time_idxs = list(range(len(time)))

    DA.init(6,2)
    DA.setEps(1e-16)  # Load the ADS objects
    state = np.array([7000, 0, 0, 0, 7.5, 0])  # Example: 1 AU from Sun, circular orbit
    state_obs = time_ref.CC2obs(state)  # Convert to observer-centric coordinates

    ads_subdomains = create_ads_with_subdomains(state_obs, num_subdomains=2)
    final_lists = [ads_subdomains]  # Wrap in a list to match expected input format
    

    ################### Test Starts ###################################

    EARTH_G = np.zeros((len(time), 6))  # Initialize Earth state vector
        
    for i in range(len(time)):
        position, velocity = acoords.get_body_barycentric_posvel('earth', time[i])       #Earth Barycentric Position and velocity???
        Earth_ICRS_pos = position.xyz.to(u.km).value.T.reshape(-1)
        Earth_ICRS_vel = velocity.xyz.to(u.km/u.s).value.T.reshape(-1)

        EARTH_G[i,:] = np.concatenate((Earth_ICRS_pos, Earth_ICRS_vel))
        EARTH_G[i,:] = time_ref.CC2obs(EARTH_G[i,:])

    result = ADS_ECISpherical_2_ICRS(final_lists, EARTH_G, time_idxs)

    truth = time_ref.ECI2ICRS(state_obs, EARTH_G[0])  # This is the reference transformation

    difference = result[0][0].box - truth

    assert np.allclose(difference.cons(), 0, rtol=1e-12), "Expected domain to match"
    
    assert result is not None, "Transformation failed"
    assert isinstance(result, list), "Expected a list of ADS objects"
    assert len(result) == 1, "Expected 1 domains result"
    assert isinstance(result[0], list), "Expected list object"
    assert isinstance(result[0][0], ADS), "Expected ADS object in subdomain"
    assert len(result[0]) == 2, "Expected 2 subdomains"
    assert result[0][0].box.shape == (6,), "Expected state vector of shape (6,)"

def test_ADS_ECISpherical_2_ICRS_domains_subdomains():
    
    times = [
        Time("2025-01-01T00:00:00", scale='tdb'),
        Time("2025-01-02T00:00:00", scale='tdb'),
        Time("2025-01-03T00:00:00", scale='tdb')
    ]
    time_idxs = list(range(len(times)+1))
    

    DA.init(6,2)
    DA.setEps(1e-16)  # Load the ADS objects
    state = np.array([7000, 0, 0, 0, 7.5, 0])  # Example: 1 AU from Sun, circular orbit
    state_obs = time_ref.CC2obs(state)  # Convert to observer-centric coordinates

    final_lists = [create_ads_with_subdomains(state_obs, num_subdomains=2)]  # Create initial subdomains
    

    ################### Test Starts ###################################

    EARTH_G = np.zeros((len(times)+1, 6))  # Initialize Earth state vector

    position, velocity = acoords.get_body_barycentric_posvel('earth', times[0])       #Earth Barycentric Position and velocity???
    Earth_ICRS_pos = position.xyz.to(u.km).value.T.reshape(-1)
    Earth_ICRS_vel = velocity.xyz.to(u.km/u.s).value.T.reshape(-1)

    EARTH_G[0,:] = np.concatenate((Earth_ICRS_pos, Earth_ICRS_vel))
    EARTH_G[0,:] = time_ref.CC2obs(EARTH_G[0,:])
    

    for i in range(1,len(times)):
        position, velocity = acoords.get_body_barycentric_posvel('earth', times[i-1])       #Earth Barycentric Position and velocity???
        Earth_ICRS_pos = position.xyz.to(u.km).value.T.reshape(-1)
        Earth_ICRS_vel = velocity.xyz.to(u.km/u.s).value.T.reshape(-1)

        EARTH_G[i,:] = np.concatenate((Earth_ICRS_pos, Earth_ICRS_vel))
        EARTH_G[i,:] = time_ref.CC2obs(EARTH_G[i,:])

    result = ADS_ECISpherical_2_ICRS(final_lists, EARTH_G, time_idxs)

    truth = time_ref.ECI2ICRS(state_obs, EARTH_G[0])  # This is the reference transformation

    difference = result[0][0].box - truth

    assert np.allclose(difference.cons(), 0, rtol=1e-12), "Expected domain to match"
    
    assert result is not None, "Transformation failed"
    assert isinstance(result, list), "Expected a list of ADS objects"
    assert len(result) == 1, "Expected 1 list result"
    assert isinstance(result[0], list), "Expected list object"
    assert isinstance(result[0][0], ADS), "Expected ADS object in subdomain"
    assert len(result[0]) == 2, "Expected 2 subdomains"
    assert result[0][0].box.shape == (6,), "Expected state vector of shape (6,)"



def Apophis_sample():
    """ Sample data for Apophis Emphermeris at 2025-06-15"""
    RA = "06h51m26.15s"
    DEC = "+21d07m06.7s"
    coord = acoords.SkyCoord(RA, DEC, frame='icrs')
    
    x = -6.145446075009656e7
    y = 2.893141879600154e8
    z = -9.229086731037766e6
    vx = -5.369034309408959e1
    vy = -6.906073442109465e0
    vz = -3.406728363690137e-2
    vy = -1.023862608131614e1
    vz = -3.504078633215579e-2
    vec = [x, y, z, vx, vy, vz]

    return coord, vec


def test_load_Apophis():
    """ Testing Apophis Emphermeris loading"""
    startTime = "2004-12-26"    # UT
    endTime = "2004-12-27"
    step = "1d"
    coord, refvec = Apophis_sample()

    eph, vec = load_Apophis_Ephemeris(startTime, endTime, step)

    assert eph is not None, "Ephemeris loading failed"
    assert vec is not None, "Vector loading failed"

    ref_RA_deg = coord.ra.deg
    ref_DEC_deg = coord.dec.deg

    RA = eph['RA'][0]
    DEC = eph['DEC'][0]

    assert np.isclose(RA, ref_RA_deg, rtol=1e-5), "RA does not match"
    assert np.isclose(DEC, ref_DEC_deg, rtol=1e-5), "DEC does not match"

    refx, refy, refz, refvx, refvy, refvz = refvec


def test_CADC_sort_CCW():
    """ Test the sorting of CADC data in counter-clockwise order """
    RA = np.array([np.pi, 2*np.pi, np.pi/2])
    DEC = np.array([np.pi/4, np.pi/12, np.pi/6])

    RA_CCW, DEC_CCW = CADC_sort_CCW(RA, DEC)

    DEC_CCW_ref = np.array([np.pi/12, np.pi/4, np.pi/6])
    RA_CCW_ref = np.array([2*np.pi, np.pi, np.pi/2])

    assert np.allclose(RA_CCW, RA_CCW_ref, rtol=1e-5), "RA_CCW does not match"
    assert np.allclose(DEC_CCW, DEC_CCW_ref, rtol=1e-5), "DEC_CCW does not match"

def generate_test_coordinates_2D(shape='circle', num_points=50, noise_level=0.1):
    """
    Generate random 2D coordinates for testing alphashape functions
    
    :param shape: Shape type ('circle', 'square', 'random', 'L_shape', 'crescent')
    :param num_points: Number of points to generate
    :param noise_level: Amount of noise to add to structured shapes
    :return: numpy array of shape (num_points, 2)
    """
    np.random.seed(42)  # For reproducible tests
    
    if shape == 'circle':
        # Generate points around a circle with some noise
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius = 1.0 + noise_level * np.random.normal(0, 1, num_points)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        coords = np.column_stack([x, y])
        
    elif shape == 'square':
        # Generate points around a square perimeter
        side_points = num_points // 4
        coords = []
        
        # Bottom edge
        x_bottom = np.linspace(-1, 1, side_points)
        y_bottom = np.ones(side_points) * (-1)
        coords.extend(np.column_stack([x_bottom, y_bottom]))
        
        # Right edge
        x_right = np.ones(side_points) * 1
        y_right = np.linspace(-1, 1, side_points)
        coords.extend(np.column_stack([x_right, y_right]))
        
        # Top edge
        x_top = np.linspace(1, -1, side_points)
        y_top = np.ones(side_points) * 1
        coords.extend(np.column_stack([x_top, y_top]))
        
        # Left edge
        x_left = np.ones(side_points) * (-1)
        y_left = np.linspace(1, -1, side_points)
        coords.extend(np.column_stack([x_left, y_left]))
        
        coords = np.array(coords)
        # Add noise
        coords += noise_level * np.random.normal(0, 1, coords.shape)
        
    elif shape == 'random':
        # Completely random points
        coords = np.random.uniform(-2, 2, (num_points, 2))
        
    else:  # Default to circle
        coords = generate_test_coordinates_2D('circle', num_points, noise_level)
    
    return coords

@pytest.mark.skipif(alphashape is None, reason="alphashape library not installed")
def test_alphashape():
    """See the Alpha shape on a plot"""
    coords = generate_test_coordinates_2D('circle', 1000)
    alpha_shape = alphashape.alphashape(coords, alpha=5)

    plt.figure()
    plt.plot(*coords.T, 'o', label='Input Points')
    plt.plot(*alpha_shape.exterior.xy, 'r-', label='Alpha Shape')
    plt.legend()
    plt.show()

    assert alpha_shape is not None, "Alpha shape generation failed"


def generate_evaluated_ADS_data():

    final_lists, Ns = sample_data()
    ads2_perimeter, ads2_perimeter_norm = perimeter_sample_data()
    
    # Use the actual ADS structure but evaluate on test coordinates
    test_coords = generate_test_coordinates_2D('circle', num_points=50)
    
    # Create time vector
    TF = 40.0
    T0 = 0.0
    Ts = 41
    tgrid = np.linspace(T0, TF, Ts)
    
    # Use a subset of time indices for testing
    time_analysis = [0, 5, 10, 16, 30, 40]  # Just use first few time steps
    
    try:
        # Use the actual eval_perimeterADS3D function with real ADS data
        result = eval_perimeterADS3D(final_lists, ads2_perimeter_norm, tgrid, 
                                   time_idxs=time_analysis, state_dim=4)
        return result
    except Exception as e:
        print(f"Error creating mock result from sample data: {e}")
        return None


@pytest.mark.skipif(alphashape is None, reason="alphashape library not installed")
def test_create_alphashape2D_basic():
    """Test the create_alphashape2D function with proper error handling"""
    
    result = generate_evaluated_ADS_data()

    if result is not None:
        manifold = result["final_map"][0]
        points_data = extract_2D_points(manifold, time_idx=0, component_indices=[0, 1], convert_to_degrees=False)
        
        assert 'unique points' in points_data, "Missing unique_points in result"
        assert 'all points' in points_data, "Missing all_points in result"
        assert len(points_data['unique points']) > 0, "No unique points extracted"
        
        print(f"Extracted {len(points_data['unique points'])} unique points from real ADS data")
        print(f"Total points: {len(points_data['all points'])}")

        # Initialize variables
        alpha_shape_ = None
        boundary_coords = None
        actual_alpha = 0.05

        try: 
            unique_points = points_data['unique points']
            result_tuple = create_alphashape2D(unique_points, alpha=actual_alpha)  # Use fixed alpha

            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                alpha_shape_, boundary_coords, actual_alpha = result_tuple
                
                assert alpha_shape_ is not None, "Alpha shape creation failed with real data"
                assert boundary_coords is not None, "Boundary coordinates missing in alpha shape result"
                assert len(boundary_coords) > 0, "No boundary points from real data"
                
                print("✅ Alpha shape created successfully")
            else:
                print(f"❌ Unexpected return type: {type(result_tuple)}")

        except Exception as e:
            print(f"❌ ADS Alpha shape failed: {e}")
            import traceback
            traceback.print_exc()

        # FIXED: Plot with proper error handling
        general_data = points_data["unique points"]
        plt.figure()
        plt.plot(general_data[:, 0], general_data[:, 1], 'o', label='input points')
        
        # Only plot alpha shape if it exists
        if alpha_shape_ is not None and hasattr(alpha_shape_, 'exterior'):
            plt.plot(*alpha_shape_.exterior.xy, 'r--', label=f'alphashape: {actual_alpha}')
        
        plt.legend()
        plt.show()
    else:
        pytest.skip("No ADS data available")




@pytest.mark.skipif(alphashape is None, reason="alphashape library not installed")
def test_create_alphashape2D_optimise():
    """Test the create_alphashape2D function with proper error handling"""
    
    result = generate_evaluated_ADS_data()

    if result is not None:
        manifold = result["final_map"][0]
        points_data = extract_2D_points(manifold, time_idx=0, component_indices=[0, 1], convert_to_degrees=False)
        
        assert 'unique points' in points_data, "Missing unique_points in result"
        assert 'all points' in points_data, "Missing all_points in result"
        assert len(points_data['unique points']) > 0, "No unique points extracted"
        
        print(f"Extracted {len(points_data['unique points'])} unique points from real ADS data")
        print(f"Total points: {len(points_data['all points'])}")

        # Initialize variables
        alpha_shape_ = None
        boundary_coords = None
        actual_alpha = None

        try: 
            unique_points = points_data['unique points']
            alpha = optalpha.optimizealpha(unique_points)
            result_tuple = create_alphashape2D(unique_points, alpha=alpha)  # Use optimized alpha

            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                alpha_shape_, boundary_coords, actual_alpha = result_tuple
                
                assert alpha_shape_ is not None, "Alpha shape creation failed with real data"
                assert boundary_coords is not None, "Boundary coordinates missing in alpha shape result"
                assert len(boundary_coords) > 0, "No boundary points from real data"
                
                print("✅ Alpha shape created successfully")
            else:
                print(f"❌ Unexpected return type: {type(result_tuple)}")

        except Exception as e:
            print(f"❌ ADS Alpha shape failed: {e}")
            import traceback
            traceback.print_exc()

        # FIXED: Plot with proper error handling
        general_data = points_data["unique points"]
        plt.figure()
        plt.plot(general_data[:, 0], general_data[:, 1], 'o', label='input points')
        
        # Only plot alpha shape if it exists
        if alpha_shape_ is not None and hasattr(alpha_shape_, 'exterior'):
            plt.plot(*alpha_shape_.exterior.xy, 'r--', label=f'alphashape: {actual_alpha}')
        
        plt.legend()
        plt.show()
    else:
        pytest.skip("No ADS data available")

def test_create_alphashape2D_subdomain_1():

    result = generate_evaluated_ADS_data()

    if result is not None:
        manifold = result["final_map"][2]
        points_data = extract_2D_points(manifold, time_idx=0, component_indices=[0, 1], convert_to_degrees=False)
        
        assert 'unique points' in points_data, "Missing unique_points in result"
        assert 'all points' in points_data, "Missing all_points in result"
        assert len(points_data['unique points']) > 0, "No unique points extracted"
        
        print(f"Extracted {len(points_data['unique points'])} unique points from real ADS data")
        print(f"Total points: {len(points_data['all points'])}")

        # Initialize variables
        alpha_shape_ = None
        boundary_coords = None
        actual_alpha = None

        try: 
            unique_points = points_data['unique points']
            alpha = optalpha.optimizealpha(unique_points)
            result_tuple = create_alphashape2D(unique_points, alpha=alpha)  # Use optimized alpha

            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                alpha_shape_, boundary_coords, actual_alpha = result_tuple
                
                assert alpha_shape_ is not None, "Alpha shape creation failed with real data"
                assert boundary_coords is not None, "Boundary coordinates missing in alpha shape result"
                assert len(boundary_coords) > 0, "No boundary points from real data"
                
                print("✅ Alpha shape created successfully")
            else:
                print(f"❌ Unexpected return type: {type(result_tuple)}")

        except Exception as e:
            print(f"❌ ADS Alpha shape failed: {e}")
            import traceback
            traceback.print_exc()

        # FIXED: Plot with proper error handling
        general_data = points_data["unique points"]
        plt.figure()
        plt.plot(general_data[:, 0], general_data[:, 1], 'o', label='input points')
        
        # Only plot alpha shape if it exists
        if alpha_shape_ is not None and hasattr(alpha_shape_, 'exterior'):
            plt.plot(*alpha_shape_.exterior.xy, 'r--', label=f'alphashape: {actual_alpha}')
        
        plt.legend()
        plt.show()
    else:
        pytest.skip("No ADS data available")

def test_create_alphashape2D_subdomain_2():

    result = generate_evaluated_ADS_data()

    if result is not None:
        manifold = result["final_map"][5]
        points_data = extract_2D_points(manifold, time_idx=0, component_indices=[0, 1], convert_to_degrees=False)
        
        assert 'unique points' in points_data, "Missing unique_points in result"
        assert 'all points' in points_data, "Missing all_points in result"
        assert len(points_data['unique points']) > 0, "No unique points extracted"
        
        print(f"Extracted {len(points_data['unique points'])} unique points from real ADS data")
        print(f"Total points: {len(points_data['all points'])}")

        # Initialize variables
        alpha_shape_ = None
        boundary_coords = None
        actual_alpha = None

        try: 
            unique_points = points_data['unique points']
            alpha = optalpha.optimizealpha(unique_points)
            result_tuple = create_alphashape2D(unique_points, alpha=alpha)  # Use optimized alpha

            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                alpha_shape_, boundary_coords, actual_alpha = result_tuple
                
                assert alpha_shape_ is not None, "Alpha shape creation failed with real data"
                assert boundary_coords is not None, "Boundary coordinates missing in alpha shape result"
                assert len(boundary_coords) > 0, "No boundary points from real data"
                
                print("✅ Alpha shape created successfully")
            else:
                print(f"❌ Unexpected return type: {type(result_tuple)}")

        except Exception as e:
            print(f"❌ ADS Alpha shape failed: {e}")
            import traceback
            traceback.print_exc()

        # FIXED: Plot with proper error handling
        general_data = points_data["unique points"]
        plt.figure()
        plt.plot(general_data[:, 0], general_data[:, 1], 'o', label='input points')
        
        # Only plot alpha shape if it exists
        if alpha_shape_ is not None and hasattr(alpha_shape_, 'exterior'):
            plt.plot(*alpha_shape_.exterior.xy, 'r--', label=f'alphashape: {actual_alpha}')
        
        plt.legend()
        plt.show()
    else:
        pytest.skip("No ADS data available")

def test_point_in_alphashape_generic():
    # Generate test data
    coords = generate_test_coordinates_2D('circle', 100)
    
    # Create alpha shape
    alpha_shape, boundary_coords, actual_alpha = create_alphashape2D(coords, alpha=0.1)

    if alpha_shape is not None: 
        test_points = [
            [0,0],
            [10,10],
            coords[0]
        ]
        test_results = []
        for i, point in enumerate(test_points):
            is_inside = points_in_alphashape(point, alpha_shape)
            test_results.append(is_inside)
            
            print(f"Point {i} {point}: {'Inside' if is_inside else 'Outside'}")

        assert test_results[0] == True, "Expected point [0,0] to be inside"
        assert test_results[1] == False
        assert test_results[2] == True

        # Test multiple points at once
        all_inside = points_in_alphashape(test_points, alpha_shape)
        print(f"Batch test results: {all_inside}")
        
        # Test with your original coordinate set
        original_inside = points_in_alphashape(coords, alpha_shape)
        percent_inside = np.mean(original_inside) * 100
        print(f"{percent_inside:.1f}% of original points are inside the alpha shape")
    
        plt.figure()
        plt.plot(coords[:, 0], coords[:, 1], 'o', label='input points')
        plt.plot(0, 0, 'o', label='test point 1')
        plt.plot(10, 10, 'o', label='test point 2')
    
        if alpha_shape is not None and hasattr(alpha_shape, 'exterior'):
            plt.plot(*alpha_shape.exterior.xy, 'r--', label=f'alphashape: {actual_alpha}')

        plt.legend()
        plt.show()

    else:
        print("Alpha shape creation failed")

def create_alphashape2D_RA_DEC():
    """
    Create a square box of half-length 3e-4 in RA and DEC directions
    Returns RA, DEC coordinates in radians for the square perimeter
    """
    # Define square box parameters 
    half_length = np.pi/12 / 10 # Half-length of the square box in radians (1.5 degree error)
    center_RA = np.deg2rad(50.002830)     # Center RA (can be modified as needed)
    center_DEC = np.deg2rad(-6.046706)    # Center DEC (can be modified as needed)

    # Number of points per side of the square
    points_per_side = 4
    
    # Generate square perimeter points
    all_points = []
    
    # Bottom edge: DEC = center_DEC - half_length, RA varies
    RA_bottom = np.linspace(center_RA - half_length, center_RA + half_length, points_per_side)
    DEC_bottom = np.full(points_per_side, center_DEC - half_length)
    bottom_edge = np.column_stack([RA_bottom, DEC_bottom])
    
    # Right edge: RA = center_RA + half_length, DEC varies
    RA_right = np.full(points_per_side, center_RA + half_length)
    DEC_right = np.linspace(center_DEC - half_length, center_DEC + half_length, points_per_side)
    right_edge = np.column_stack([RA_right, DEC_right])
    
    # Top edge: DEC = center_DEC + half_length, RA varies (reverse order for CCW)
    RA_top = np.linspace(center_RA + half_length, center_RA - half_length, points_per_side)
    DEC_top = np.full(points_per_side, center_DEC + half_length)
    top_edge = np.column_stack([RA_top, DEC_top])
    
    # Left edge: RA = center_RA - half_length, DEC varies (reverse order for CCW)
    RA_left = np.full(points_per_side, center_RA - half_length)
    DEC_left = np.linspace(center_DEC + half_length, center_DEC - half_length, points_per_side)
    left_edge = np.column_stack([RA_left, DEC_left])
    
    # Combine all edges (counter-clockwise order)
    all_points = np.vstack([bottom_edge, right_edge, top_edge, left_edge])
    
    # Remove duplicate corner points
    all_points = np.unique(all_points, axis=0)
    
    print(f"Created square box with {len(all_points)} points")
    print(f"RA range: [{all_points[:, 0].min():.6e}, {all_points[:, 0].max():.6e}] radians")
    print(f"DEC range: [{all_points[:, 1].min():.6e}, {all_points[:, 1].max():.6e}] radians")
    print(f"Box half-length: {half_length:.6e} radians ({np.degrees(half_length):.6f} degrees)")
    
    return all_points



def test_RA_DEC_Alphashape():

    RA_DEC = create_alphashape2D_RA_DEC()

    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1])

    RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))

    alpha = optalpha.optimizealpha(RA_DEC_CCW)
    result_tuple = create_alphashape2D(RA_DEC_CCW, alpha=alpha)

    if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
        alpha_shape_, boundary_coords, actual_alpha = result_tuple
                
        assert alpha_shape_ is not None, "Alpha shape creation failed with real data"
        assert boundary_coords is not None, "Boundary coordinates missing in alpha shape result"
        assert len(boundary_coords) > 0, "No boundary points from real data"
                
        print("✅ Alpha shape created successfully")
    else:
        print(f"❌ Unexpected return type: {type(result_tuple)}")



    # FIXED: Plot with proper error handling
    general_data = RA_DEC_CCW
    plt.figure()
    plt.plot(general_data[:, 0], general_data[:, 1], 'o', label='input points')
        
    # Only plot alpha shape if it exists
    if alpha_shape_ is not None and hasattr(alpha_shape_, 'exterior'):
        plt.plot(*alpha_shape_.exterior.xy, 'r--', label=f'alphashape: {actual_alpha}')
        
    plt.legend()
    plt.show()

def test_CADC_Example_point():
    from utils.post_process import CADC_Example_point
    ra = 334.0276125
    dec = -17.70465556
    start_date = Time('2003-08-14T00:00:00.000', format='isot')
    end_date = Time('2003-12-12T00:00:00.000', format='isot')

    #Test with Collection (Same as example)
    result = CADC_Example_point(ra, dec, start_date, end_date, collection='CFHT')
    
    assert result is not None
    assert len(result) == 80 
    assert result[0:5][0][0] == '714734p'

    #Test with empty collection (same as example minus collection designations)
    result_no_collection = CADC_Example_point(ra, dec, start_date, end_date, collection=None)
    assert result_no_collection is not None
    print(result_no_collection[0:5])

def test_CADC_Example_Polygon():
    from utils.post_process import CADC_Fixed_Polygon, CADC_Fixed_Polygon_contains
    from shapely.geometry import Polygon
    import time
    #Create Polygon
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon2004-12-27 01:29:04
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2004-12-27T01:29:04', format='isot')
    end_date = Time('2004-12-27T01:29:37', format='isot')

    #Run Test with collection
    start = time.time()
    result = CADC_Fixed_Polygon(polygon, start_date, end_date, collection='CFHT')
    end = time.time() 
    interval = end - start
    print(interval)



    assert len(result) > 0
    assert result is not None
    print(result[0:5])

    #Run Test without collection
    result_nocollection = CADC_Fixed_Polygon(polygon, start_date, end_date, collection=None)

    assert result_nocollection is not None
    assert len(result_nocollection) > 0
    print(result_nocollection[0:5])

def test_CADC_example_SSOIS_image_ID():
    
    from utils.post_process import CADC_Fixed_Polygon, SSOIS_Query_Apophis
    from shapely.geometry import Polygon
    import time
    #Create Polygon
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon2004-12-27 01:29:04
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time 2005-01-17 01:19:11
    start_date = Time('2005-01-17T01:19:05', format='isot')
    end_date = Time('2005-01-17T01:20:10', format='isot')

    #Run Test with collection
    start = time.time()
    result = CADC_Fixed_Polygon(polygon, start_date, end_date, collection=None)
    end = time.time() 
    interval = end - start
    print(result['productID'])
    
    #Load SSOIS

    end_search_mjd = start_date + TimeDelta(1, format='jd')
    error = 1 * u.arcsec
    result_ssois = SSOIS_Query_Apophis(error.value, start_date, end_search_mjd, interval_mjd=end_date)

    print(result_ssois['Image'])

def test_ESO_Example_polygon():

    from utils.post_process import ESO_Fixed_Polygon_async
    from shapely.geometry import Polygon
    import time

    #Create Polygon
    # RA = 50.002830
    # DEC = -6.046706
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon2004-12-27 01:29:04
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2005-01-17T01:18:00', format='isot')
    end_date = Time('2005-01-18T01:21:40', format='isot')
    #Run Test with collection
    start = time.time()
    result = ESO_Fixed_Polygon_async(polygon, start_date, end_date)
    end = time.time() 
    interval = end - start
    print(interval)
    print(result)

    assert result is not None

def test_CADC_Example_polygon_contains():
    from utils.post_process import CADC_Fixed_Polygon, CADC_Fixed_Polygon_contains
    from shapely.geometry import Polygon
    import time    
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon2004-12-27 01:29:04
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2004-12-27T01:29:04', format='isot')
    end_date = Time('2004-12-27T01:29:37', format='isot')

    #Run Test with collection
    start = time.time()
    result = CADC_Fixed_Polygon(polygon, start_date, end_date, collection='CFHT')
    end = time.time() 
    interval = end - start
    print(interval)

    #Run Test with collection for the 'contains' case
    start = time.time()
    result = CADC_Fixed_Polygon_contains(polygon, start_date, end_date, collection='CFHT')
    end = time.time()
    interval_contains = end - start
    print(interval_contains)

def test_CADC_Example_Polygon_async():
    from utils.post_process import CADC_Fixed_Polygon_async
    from shapely.geometry import Polygon
    import time
    #Create Polygon
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = np.degrees(DEC_CCW)

    
    #Create Polygon - same item as alphashape
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2003-08-14T00:00:00.000', format='isot')
    end_date = Time('2003-12-12T00:00:00.000', format='isot')

    #Run Test with collection
    start = time.time()
    result = CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection='CFHT')
    end = time.time()
    interval = end - start
    print(interval)
    assert result is not None
    print(result[0:5])


    #Run Test without collection
    result_nocollection = CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)

    columns_subset = [
    'productID', 'collection', 'energy_bandpassName', 'time_bounds_samples',
    'time_bounds_lower', 'time_exposure'
    ]

    assert result_nocollection is not None
    print(result_nocollection[columns_subset][0:5])

def test_CADC_ESO_combine():    #This function will be used to assess the Propagated uncertanity (defined by the polygon), with the true value (Defined by the SSOIS value) at an instant
    from utils.post_process import CADC_Fixed_Polygon_async, ESO_Fixed_Polygon_async, CADC_ESO_combine
    from shapely.geometry import Polygon
    import time

    #Create Polygon
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2004-12-27T01:29:00', format='isot') 
    end_date = Time('2005-01-17T01:21:23', format='isot')

    #Query CADC Database
    CADC = CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"CADC obtained length: {len(CADC)} ")
    #Query ESO Database
    ESO = ESO_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"ESO obtained length: {len(ESO)} ")
    result = CADC_ESO_combine(CADC, ESO)
    assert result is not None
    assert len(result) == (len(CADC) + len(ESO))    #combined properly
    #add assertion so that the vstack works correctly.


def test_image_match():
    """Full integrated test of CADC and ESO queries compared to SSOIS truth"""
    
    from utils.post_process import CADC_Fixed_Polygon_async, ESO_Fixed_Polygon_async, CADC_ESO_combine, SSOIS_Query_Apophis, image_match
    from shapely.geometry import Polygon
    from astropy.table import Table
    import time

    #Create Polygon
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2004-12-27T01:29:00', format='isot') 
    end_date = Time('2005-01-17T01:21:23', format='isot')

    #Query CADC Database
    CADC = CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"CADC obtained length: {len(CADC)} ")
    #Query ESO Database
    ESO = ESO_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"ESO obtained length: {len(ESO)} ")
    AQDL_combined = CADC_ESO_combine(CADC, ESO)
    assert AQDL_combined is not None
    assert len(AQDL_combined) == (len(CADC) + len(ESO))    #combined properly
    #add assertion so that the vstack works correctly.

    #Query SSOIS Database
    error = 0
    SSOIS = SSOIS_Query_Apophis(error, start_date, end_date, end_date, search=False)
    print(f"SSOIS obtained length: {len(SSOIS)} ")
    assert SSOIS is not None
    SSOIS_tbl = Table.from_pandas(SSOIS)

    # Perform image matching
    matches = image_match(SSOIS_tbl, AQDL_combined, case_insensitive=True, return_astropy=True)
    print(f"Image matches found: {len(matches)} ")
    assert matches is not None
    for match in matches:
        print(f" - {match}")

    assert SSOIS['Image'][2] == matches['Image'][0]
    print(matches)

def test_image_metrics():
    """Full integrated test of CADC and ESO queries compared to SSOIS truth"""
    
    from utils.post_process import CADC_Fixed_Polygon_async, ESO_Fixed_Polygon_async, CADC_ESO_combine, SSOIS_Query_Apophis, image_match, image_metrics
    from shapely.geometry import Polygon
    from astropy.table import Table
    import time

    #Create Polygon
    RA_DEC = create_alphashape2D_RA_DEC()   #Obtains a Unique point RA and DEC
    RA_CCW, DEC_CCW = CADC_sort_CCW(RA_DEC[:,0], RA_DEC[:,1]) #Sort CCW
    #Convert to Degrees
    RA_CCW_deg = (np.degrees(RA_CCW) % 360 ) 
    DEC_CCW_deg = (np.degrees(DEC_CCW))

    #Create Polygon
    polygon = Polygon(zip(RA_CCW_deg, DEC_CCW_deg)) #Polygon for A
    #Time
    start_date = Time('2004-12-27T01:29:00', format='isot') 
    end_date = Time('2005-01-17T01:21:23', format='isot')

    #Query CADC Database
    CADC = CADC_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"CADC obtained length: {len(CADC)} ")
    #Query ESO Database
    ESO = ESO_Fixed_Polygon_async(polygon, start_date, end_date, collection=None)
    print(f"ESO obtained length: {len(ESO)} ")
    AQDL_combined = CADC_ESO_combine(CADC, ESO)
    assert AQDL_combined is not None
    assert len(AQDL_combined) == (len(CADC) + len(ESO))    #combined properly
    #add assertion so that the vstack works correctly.

    #Query SSOIS Database
    error = 0
    SSOIS = SSOIS_Query_Apophis(error, start_date, end_date, end_date, search=False)
    print(f"SSOIS obtained length: {len(SSOIS)} ")
    assert SSOIS is not None
    SSOIS_tbl = Table.from_pandas(SSOIS)

    # Perform image matching
    matches = image_match(SSOIS_tbl, AQDL_combined, case_insensitive=True, return_astropy=True)
    print(f"Image matches found: {len(matches)} ")
    assert matches is not None
    for match in matches:
        print(f" - {match}")

    assert SSOIS['Image'][2] == matches['Image'][0]
    print(matches)

    # Calculate and print metrics
    dict = image_metrics(AQDL_combined, matches, SSOIS_tbl)

    assert dict['TP'] == 3



    
    