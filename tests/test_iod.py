import numpy as np
import pytest
from utils.iod import Guass_8th_seed, f_g_series, DAIOD_1, DAIOD_2, DAIOD_1Scipy, DAIOD_1_debugSciPy,DAIOD_1Scipy_invert, DAIOD_2, DAIOD_full, DAIOD_ADS_full
import utils.post_process as post
import numpy as np
from typing import Union
from daceypy import DA, array, ADS
import daceypy.op as op
from numpy.typing import NDArray
from utils.lambert_izzo import lambert_izzo
from utils.poli_izzo import izzo
from utils.iod import newton_nominal_DAVec, Implicit_solver_DAVec
import utils.time_reference as time_ref
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, ITRS, GCRS
from astropy.time import Time
import astropy.units as u
import utils.Classical_IOD as PWIOD

def sample_data_Earth():
    """
        Generate Guass Solution for DAIOD (ECI)
    """
    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          # Observer position [x components at all instances]
        [3430.2, 3460.1, 3490.1],          # Observer position [y components at all instances]
        [4078.5, 4078.5, 4078.5]           # Observer position [z components at all instances]
    ])

    obs_dir = np.array([            # Direction vectors of the observations
        [0.71643, 0.56897, 0.41841],            # Direction vector for x components
        [0.68074, 0.79531, 0.87007],            # Direction vector for y components
        [-0.15270,-0.20917,-0.26059]            # Direction vector for z components
    ])

    t = np.array([0.0, 118.10, 237.58])


    # Should not raise and should return two lists of length 3
    position, ranges, range_mag, v_2 = Guass_8th_seed(pos_obs, obs_dir, t, mu=3.986e5)
    return range_mag, obs_dir, t, pos_obs, position, v_2

def perimeter_sample_data_6D_systematic(ang_sigma, grid_size=3):
    """
    Alternative: Generate 6D perimeter with systematic face sampling.
    Uses a regular grid on each face for more systematic coverage.
    
    Parameters:
    ang_sigma: Angular uncertainty (same for all measurements)
    grid_size: Number of points along each dimension for face grids
    
    Returns:
    ads2_perimeter_norm: 2D perimeter (for compatibility)
    ads6_perimeter_norm: 6D perimeter with systematic face sampling
    """
    all_points = []
    
    # Generate vertices (corners of hypercube)
    print("Generating 6D hypercube vertices...")
    for i in range(2**6):  # 64 vertices
        vertex = []
        for j in range(6):
            if (i >> j) & 1:
                vertex.append(1.0)
            else:
                vertex.append(-1.0)
        all_points.append(vertex)
    
    print(f"Generated {len(all_points)} vertices")
    
    # Generate face centers and edge midpoints
    print("Generating 6D hypercube faces...")
    # Face centers (one dimension fixed, others at 0)
    for fixed_dim in range(6):
        for fixed_value in [-1.0, 1.0]:
            face_center = [0.0] * 6
            face_center[fixed_dim] = fixed_value
            all_points.append(face_center)
    
    # Edge centers (two dimensions fixed, others at 0)
    for dim1 in range(6):
        for dim2 in range(dim1 + 1, 6):
            for val1 in [-1.0, 1.0]:
                for val2 in [-1.0, 1.0]:
                    edge_center = [0.0] * 6
                    edge_center[dim1] = val1
                    edge_center[dim2] = val2
                    all_points.append(edge_center)
    
    # Systematic grid on each face
    if grid_size > 2:
        grid_points = np.linspace(-1, 1, grid_size)
        
        for fixed_dim in range(6):  # Which dimension to fix
            for fixed_value in [-1.0, 1.0]:  # Fix at boundary
                other_dims = [d for d in range(6) if d != fixed_dim]
                
                # Generate systematic grid on this face
                # Sample every other dimension in a grid pattern
                for i in range(min(4, len(other_dims))):  # Limit to avoid explosion
                    for grid_val in grid_points[1:-1]:  # Exclude boundaries (already have vertices)
                        face_point = [0.0] * 6    
                        face_point[fixed_dim] = fixed_value
                        face_point[other_dims[i]] = grid_val
                        all_points.append(face_point)
    
    ads6_perimeter_norm = np.array(all_points)
    
    # Remove duplicates and sort
    ads6_perimeter_norm = np.unique(ads6_perimeter_norm, axis=0)
    
    # Generate 2D version for compatibility
    ads2_perimeter_norm = ads6_perimeter_norm[:, [0, 3]]  # Take first RA and first DEC
    
    print(f"Generated 6D hypercube with {ads6_perimeter_norm.shape[0]} total points")
    
    return ads2_perimeter_norm, ads6_perimeter_norm
def getTFRM_pirovano():
    """GET TGRM location ECI"""

    # Example: LLA coordinates
    lat = 42.0516* u.deg       # latitude
    lon = 0.7293 * u.deg      # longitude
    alt = 1.622 * u.km         # altitude

    # Time of conversion (important because Earth rotates!)
    t0 = Time("2016-01-12T21:36:00", scale="utc")
    t1 = Time("2016-01-12T21:40:00", scale="utc")
    t2 = Time("2016-01-12T21:44:00", scale="utc")
    
    t = [t0, t1, t2]
    eci_pos = np.zeros((3,3))
    eci_vel = np.zeros((3,3))
    for i in range(3):
        # Step 1: Define location
        loc = EarthLocation(lat=lat, lon=lon, height=alt)

        # Step 2: Represent as ITRS coordinates
        itrs = loc.get_itrs(obstime=t[i])

        # Step 3: Convert ITRS → GCRS (ECI)
        eci_pos[:,i] = itrs.transform_to(GCRS(obstime=t[i])).cartesian.xyz.to(u.km)

        if hasattr(GCRS.cartesian, 'differentials') and GCRS.cartesian.differentials:
            eci_vel[:,i] = GCRS.cartesian.differentials['s'].d_xyz.to(u.km/u.s)
        else:
            dt = 1.0  # seconds
            t_plus = Time(t[i].jd + dt/86400, format='jd', scale='utc')
            itrs_plus = loc.get_itrs(obstime=t_plus)
            gcrs_plus = itrs_plus.transform_to(GCRS(obstime=t_plus))
            eci_pos_plus = gcrs_plus.cartesian.xyz.to(u.km).value
            eci_vel[:,i] = (eci_pos_plus - eci_pos[:,i]) / dt

    return eci_pos, eci_vel, t

def Observation_data_pirovano():
    """Generate NORAD 36830"""

    ang_sigma = 1/3 * (1/3600) * (np.pi/180)  # ≈ 1.61e-6 radians
    sma = 42164.8  # km
    e = 0.000327
    i = np.deg2rad(0.057)  # rad
    RAAN = np.deg2rad(214.869)  # rad
    w = np.deg2rad(78.136)  # rad
    nu_0 = np.deg2rad(122.0860609)  # rad
    nu_1 = np.deg2rad(122.3360609)
    nu_2 = np.deg2rad(122.5860609)

    X_0 = [sma, e, i, RAAN, w, nu_0]
    X_1 = [sma, e, i, RAAN, w, nu_1]
    X_2 = [sma, e, i, RAAN, w, nu_2]

    r_CC_0, v_CC_0 = time_ref.COE2CC(X_0, mu=3.986e5)
    X_CC_0 = np.concatenate((r_CC_0, v_CC_0))
    r_CC_1, v_CC_1 = time_ref.COE2CC(X_1, mu=3.986e5)
    X_CC_1 = np.concatenate((r_CC_1, v_CC_1))
    r_CC_2, v_CC_2 = time_ref.COE2CC(X_2, mu=3.986e5)
    X_CC_2 = np.concatenate((r_CC_2, v_CC_2))

    X_CC = np.column_stack((X_CC_0, X_CC_1, X_CC_2))

    eci_observation_positions, eci_observation_velocities, t = getTFRM_pirovano()
    relative_pos = X_CC[:3,:] - eci_observation_positions 
    
    RA = np.zeros(3)
    DEC = np.zeros(3)
    range_val = np.zeros(3)

    for i in range(3):
        x, y, z = relative_pos[:,i]
        range_val[i] = np.linalg.norm(relative_pos[:,i])
        RA[i] = np.arctan2(y, x)
        DEC[i] = np.arcsin(z / range_val[i])
    
    return RA, DEC, ang_sigma, eci_observation_positions, X_CC, t

def load_Maria_Observatory():
    """Load observation data for Maria Observatory"""
    t0 = Time("2009-09-26T02:21:55", scale="utc")
    t1 = Time("2009-09-26T04:21:55", scale="utc")
    t2 = Time("2009-09-27T02:21:55", scale="utc")

    t = [t0, t1, t2]
    L = 1.10536
    d_r = 0.747550
    h_e = 0.662053
    R_earth = 6378.137
    # Load Earth Equatorial Heliocentric position
    d_r_km = d_r *R_earth
    h_e_km = h_e *R_earth
    lat_rad = np.arctan2(h_e, d_r)
    lat = lat_rad * u.rad
    # Define Observer location on Earth 
    lon = L * u.rad
    r_total = np.sqrt(d_r_km**2 + h_e_km**2)
    altitude = (r_total - R_earth) * u.km
    
    print(f"Observatory coordinates:")
    print(f"  Latitude: {np.degrees(lat_rad):.4f}°")
    print(f"  Longitude: {np.degrees(L):.4f}°")
    print(f"  Altitude: {altitude.value:.3f} km")
    
    # Define observer location on Earth
    maria_location = EarthLocation(lat=lat, lon=lon, height=altitude)
    
    # Convert Observer location -> ECI -> Heliocentric
    helio_pos = np.zeros((3, 3))
    helio_vel = np.zeros((3, 3))
    
    for i in range(3):
        # Step 1: Get Earth location in ITRS
        itrs = maria_location.get_itrs(obstime=t[i])        #itrs is a geocentric Earth Fixed System - Needs to change.
        
        # Step 2: Convert ITRS -> GCRS (ECI)  
        gcrs = itrs.transform_to(GCRS(obstime=t[i]))
        earth_observer_eci = gcrs.cartesian.xyz.to(u.km).value
        earth_observer_ecl = time_ref.ROT1(np.deg2rad(23.4)) @ earth_observer_eci
        # Step 3: Get Earth's heliocentric position at observation time
        from astropy.coordinates import get_body_barycentric
        earth_helio = get_body_barycentric('earth', t[i])
        earth_helio_pos_ICRS = earth_helio.xyz.to(u.km).value
    
        earth_helio_pos_ecl = time_ref.ROT1(np.deg2rad(23.4)) @ earth_helio_pos_ICRS

        # Step 4: Observer heliocentric position = Earth heliocentric + Observer Heliocentric
        helio_pos[:, i] = earth_helio_pos_ecl + earth_observer_ecl
        
        # Calculate heliocentric velocity (numerical differentiation)
        dt = 60.0  # 60 seconds for velocity calculation
        t_plus = Time(t[i].jd + dt/86400, format='jd', scale='utc')
        
        # Earth position at t + dt
        earth_helio_plus = get_body_barycentric('earth', t_plus)
        earth_helio_pos_plus = earth_helio_plus.xyz.to(u.km).value
        # Observer ECI at t + dt
        itrs_plus = maria_location.get_itrs(obstime=t_plus)
        gcrs_plus = itrs_plus.transform_to(GCRS(obstime=t_plus))
        earth_observer_eci_plus = gcrs_plus.cartesian.xyz.to(u.km).value
        
        # Observer heliocentric position at t + dt
        helio_pos_plus = earth_helio_pos_plus + earth_observer_eci_plus
        
        # Calculate velocity
        helio_vel[:, i] = (helio_pos_plus - helio_pos[:, i]) / dt
    
    return helio_pos, helio_vel, t


def load_K09S19T_horizons():
    """Load K09S19T asteroid data via JPL Horizons using astroquery"""
    
    from astroquery.jplhorizons import Horizons
    from astropy.time import Time
    import astropy.units as u
    
    # Observation times (example)
    t0 = Time("2009-09-26T02:21:55", scale="utc")
    t1 = Time("2009-09-26T04:21:55", scale="utc")
    t2 = Time("2009-09-27T02:21:55", scale="utc")

    times = [t0, t1, t2]
    RA_rad = np.zeros(3)
    DEC_rad = np.zeros(3)
    X_ref = np.zeros((6,3))

    # Query JPL Horizons for K09S19T
    # '500' is the geocenter, change to your observer location code if needed

    
    # Get ephemeris data
    RA_1_hours= 5 + 39/60 + 51.01/3600
    DEC_1_deg = 31 + 58/60 + 45.9/3600

    RA_2_hours = 5 + 39/60 + 37.97/3600
    DEC_2_deg = 32 + 2/60 + 09.0/3600

    RA_3_hours = 5 + 37/60 + 16.34/3600
    DEC_3_deg = 32 + 38/60 + 14.2/3600

    RA_hours = np.array([RA_1_hours, RA_2_hours, RA_3_hours])
    DEC_deg = np.array([DEC_1_deg, DEC_2_deg, DEC_3_deg])

    X = 1.700292505832195e6 
    Y = 2.227812927476276e7 
    Z = 3.408339649882400e6
    VX = 2.929622303285276
    VY = 6.749889371084431
    VZ = 4.267827176156860

    X_ref_epoch = np.array([X, Y, Z, VX, VY, VZ])  # Heliocentric Ecliptic Asteroid State Vector

    # Convert to radians
    for i in range(3):
        RA_rad[i] = (RA_hours[i] * 15 * u.deg).to(u.rad).value  # 15 deg/hour
        DEC_rad[i] = (DEC_deg[i] * u.deg).to(u.rad).value

    ang_sigma = 1.454e-6 / 3  # radians (as in your code)

    return RA_rad, DEC_rad, ang_sigma, times, X_ref_epoch


    

def test_Armillien_DAIOD_full():
    """Testing DAIOD_full with Armillien data"""

    RA_rad, DEC_rad, ang_sigma, observation_times, X_ref = load_K09S19T_horizons()
    position_observer, velocity_observer, observation_times = load_Maria_Observatory()
    Obs_perimeter, Obs_perimeter_norm = perimeter_sample_data_6D_independent(ang_sigma)

    from astropy.coordinates import get_body_barycentric
    earth_helio = get_body_barycentric('earth', observation_times[1])
    earth_helio_pos_ICRS = earth_helio.xyz.to(u.km).value
    earth_CC_epoch = time_ref.ROT1(23.4) @ earth_helio_pos_ICRS

    mu = 1.327e11 # kms-1

    observation_times_seconds = np.array([observation_times[i].jd for i in range(3)]) * 86400  # Convert to seconds

    DA.init(6,6)
    X0_CC_epoch = DAIOD_full(RA_rad, DEC_rad, ang_sigma, ang_sigma, position_observer, observation_times_seconds, mu, 6, prograde=True)

    # Further processing or assertions can be done here
    assert X0_CC_epoch is not None, "ERROR: DAIOD_full did not return a valid state vector"
    assert X0_CC_epoch.shape == (6,), "ERROR: DAIOD_full did not return a state vector of correct shape"
    assert X0_CC_epoch.cons() == X_ref

    #convert Heliocentric Ecliptic State Vector to ECI Equatorial. Then Equatorial ECI to Topocentric
    X0_CC_ECI_epoch = time_ref.Helio2ECIJ2000(X0_CC_epoch, earth_CC_epoch)
    X0_Obs_ECI = time_ref.CC2obs(X0_CC_ECI_epoch)

    assert X0_Obs_ECI[0].cons() == RA_rad[1]
    assert X0_Obs_ECI[1].cons() == DEC_rad[1]

    #Evaluate DA polynominal and plot (RA vs DEC)
    perimeter_list = [Obs_perimeter_norm[i, :].tolist() for i in range(Obs_perimeter_norm.shape[0])]


    #prepare output array
    n_perimeter_points = Obs_perimeter_norm.shape[0]
    n_observables = len(X0_Obs_ECI)
    Obs_relative_perimeter = np.zeros((n_perimeter_points, n_observables))

    post.eval_perimeterADS3D
    #Loop through each perimeter point
    for i in range(n_perimeter_points):
        eval_point = Obs_perimeter_norm[i, :].tolist()
        evaulated_obs = X0_Obs_ECI.eval(eval_point)
        Obs_relative_perimeter[i, :] = evaulated_obs

        if i % 10 == 0:
            print(f"Evaluated {i+1}/{n_perimeter_points} perimeter points")

    #Plotting
    plt.figure()
    plt.plot(Obs_relative_perimeter[:,0], Obs_relative_perimeter[:,1],'b-')
    plt.xlabel("RA (rad)")
    plt.ylabel("DEC (rad)")
    plt.title("Observation Perimeter")
    plt.grid()
    plt.show()


def test_DAIOD_ADS_full():
    """Test DAIOD+ADS with Cutis example"""

    ang_sigma = 1/3 * (1/3600) * (np.pi/180)  # ≈ 1.61e-6 radians
    Obs_perimeter_norm = post.gen_grid6D(ang_sigma, 3)
    mu = 3.986e5
    observation_times_seconds = np.array([0, 118.10, 237.58])  # Convert to seconds
    RA_rad = np.array([np.deg2rad(43.537), np.deg2rad(54.420), np.deg2rad(64.318)]) #Topocentric
    DEC_rad = np.array([np.deg2rad(-8.7833), np.deg2rad(-12.074), np.deg2rad(-15.105)])
    pos_obs = np.array([[3489.8, 3460.1, 3429.9],
                        [3430.2, 3460.1, 3490.1],
                        [4078.5, 4078.5, 4078.5]])
    earth_rot = np.array([0, 0, 7.2921159e-5])  # rad/s
    vel_obs = np.cross(earth_rot, pos_obs[:,1])  # km/s

    X_ECI_observation_epoch = np.concatenate((pos_obs[:,1], vel_obs))
    
    #DAIOD
    DA.init(6,6)
    X0_CC_ECI_epoch = DAIOD_ADS_full(RA_rad, DEC_rad, ang_sigma, ang_sigma, pos_obs, observation_times_seconds, mu, 6, prograde=True)
    
    assert X0_CC_ECI_epoch is not None, "ERROR: DAIOD_ADS_full did not return a valid state vector"

    X0_CC_ECI_epoch_list = [X0_CC_ECI_epoch]
    #convert ADS objects
    X0_topo_cart = post.ADS_ECICart_2_Topocentric(X0_CC_ECI_epoch_list, X_ECI_observation_epoch)
    X0_obs = post.ADS_Cart_2_Obs(X0_topo_cart)
    #Evaluate ADS objects
    results = post.eval_perimeterADS3D(X0_obs, Obs_perimeter_norm, time_vec=[observation_times_seconds[1]], time_idxs=[0], state_dim=6)

    Obs = results['final_map'][0] 
    RA = Obs[:,0,:].reshape(-1)
    DEC = Obs[:,1,:].reshape(-1)
    range = Obs[:,2,:].reshape(-1)
    range_rate = Obs[:,5,:].reshape(-1)
    
    #plot
    plt.figure(1)
    plt.plot(RA, DEC,'bo')
    plt.xlabel("RA (rad)")
    plt.ylabel("DEC (rad)")
    plt.title("Observation Perimeter")
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(range, range_rate,'bo')
    plt.xlabel("Range (km)")
    plt.ylabel("Range-Rate (km/s)")
    plt.title("Observation Perimeter")
    plt.grid()
    plt.show()


def test_DAIOD_full():
    """ Testing DAIOD_full() function with Cutis Example"""

    ang_sigma = 1/3 * (1/3600) * (np.pi/180)  # ≈ 1.61e-6 radians
    Obs_perimeter_norm = post.gen_grid6D(ang_sigma, 3)
    mu = 3.986e5
    observation_times_seconds = np.array([0, 118.10, 237.58])  # Convert to seconds
    RA_rad = np.array([np.deg2rad(43.537), np.deg2rad(54.420), np.deg2rad(64.318)]) #Topocentric
    DEC_rad = np.array([np.deg2rad(-8.7833), np.deg2rad(-12.074), np.deg2rad(-15.105)])
    pos_obs = np.array([[3489.8, 3460.1, 3429.9],
                        [3430.2, 3460.1, 3490.1],
                        [4078.5, 4078.5, 4078.5]])
    earth_rot = np.array([0, 0, 7.2921159e-5])  # rad/s
    vel_obs = np.cross(earth_rot, pos_obs[:,1])  # km/s

    X_ECI_observation_epoch = np.concatenate((pos_obs[:,1], vel_obs))

    DA.init(6,6)
    X0_CC_ECI_epoch = DAIOD_full(RA_rad, DEC_rad, ang_sigma, ang_sigma, pos_obs, observation_times_seconds, mu, 6, prograde=True)

    # Further processing or assertions can be done here
    #X0_relative_pos = X0_CC_epoch[:3] - eci_observation_positions[:,1] #check X0_CC_epoch
    #X0_relative_vel = X0_CC_epoch[3:] - eci_observation_velocities[:,1]
    #X0_relative = X0_relative_pos.concat(X0_relative_vel)
    # Convert from ECI Cartesian to Spherical to compare with Pirovano
    X0_CC_topo = X0_CC_ECI_epoch - X_ECI_observation_epoch

    Obs_relative = time_ref.CC2obs(X0_CC_topo) # The Constant RA and DEC are different by 1e-2 rad - This is seen from Earth Centre.
    perimeter_list = [Obs_perimeter_norm[i, :].tolist() for i in range(Obs_perimeter_norm.shape[0])]

    #Evaluation of DA objects
    Obs_relative_cons = Obs_relative.cons()

    #prepare output array
    n_perimeter_points = Obs_perimeter_norm.shape[0]
    n_observables = len(Obs_relative)
    Obs_relative_perimeter = np.zeros((n_perimeter_points, n_observables))
    #Loop through each perimeter point
    for i in range(n_perimeter_points):
        eval_point = Obs_perimeter_norm[i, :].tolist()  #TODO: Problem is here (Not evaluating properly to form a box)
        evaluated_obs = Obs_relative.eval(eval_point)
        Obs_relative_perimeter[i, :] = evaluated_obs

        if i % 10 == 0:
            print(f"Evaluated {i+1}/{n_perimeter_points} perimeter points")

    #assertions
    #assert np.allclose(X0_CC_epoch.cons(), X_CC[:, 1], rtol=1e-2), "ERROR: DAIOD_full did not return expected state vector"
    RA = np.unique(Obs_relative_perimeter[:,0])
    DEC = np.unique(Obs_relative_perimeter[:,1])
    X, Y = np.meshgrid(RA, DEC)
    #plotting - Almost Need to 
    plt.figure(1)
    plt.scatter(Obs_relative_perimeter[:,0], Obs_relative_perimeter[:,1])
    plt.plot(Obs_relative.cons()[0], Obs_relative.cons()[1],'ro') #Plot nominal

    if n_perimeter_points >= 64:
        plt.plot(Obs_relative_perimeter[:64,0], Obs_relative_perimeter[:64,1], 'ro', markersize=4, label='Vertices')
    
    plt.xlabel("RA (Rad)")
    plt.ylabel("DEC (Rad)")
    plt.title("RA and DEC Observation Perimeter")
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(Obs_relative_perimeter[:,2], Obs_relative_perimeter[:,5],'bo')
    plt.plot(Obs_relative.cons()[2], Obs_relative.cons()[5],'ro') #Plot nominal
    plt.xlabel("Range (km)")
    plt.ylabel("Range-Rate (km/s)")
    plt.title("Range and Range-Rate Observation Perimeter")
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(Obs_relative_perimeter[:,1], Obs_relative_perimeter[:,4],'bo')
    plt.plot(Obs_relative.cons()[1], Obs_relative.cons()[4],'ro') #Plot nominal
    plt.xlabel("DEC (Rad)")
    plt.ylabel("DEC-rate (Rad/s)")
    plt.title("DEC and Range-Rate Observation Perimeter")
    plt.grid()
    plt.show()

    plt.figure(4)
    plt.plot(Obs_relative_perimeter[:,0], Obs_relative_perimeter[:,3],'bo')
    plt.plot(Obs_relative.cons()[0], Obs_relative.cons()[3],'ro') #Plot nominal
    plt.xlabel("RA (Rad)")
    plt.ylabel("RA-rate (Rad/s)")
    plt.title("RA and RA-rate Observation Perimeter")
    plt.grid()
    plt.show()
    print("DAIOD_full test completed successfully.")

def test_DAIOD_full_vs_PW():
    """ Testing DAIOD Algorithm vs IOD + perturbations """
    
    ang_sigma = 1/3 * (1/3600) * (np.pi/180)  # ≈ 1.61e-6 radians
    Obs_perimeter_norm = post.gen_grid6D(ang_sigma, 3)
    mu = 3.986e5
    observation_times_seconds = np.array([0, 118.10, 237.58])  # Convert to seconds
    RA_rad = np.array([np.deg2rad(43.537), np.deg2rad(54.420), np.deg2rad(64.318)]) #Topocentric
    DEC_rad = np.array([np.deg2rad(-8.7833), np.deg2rad(-12.074), np.deg2rad(-15.105)])
    pos_obs = np.array([[3489.8, 3460.1, 3429.9],
                        [3430.2, 3460.1, 3490.1],
                        [4078.5, 4078.5, 4078.5]])
    earth_rot = np.array([0, 0, 7.2921159e-5])  # rad/s
    vel_obs = np.cross(earth_rot, pos_obs[:,1])  # km/s

    X_ECI_observation_epoch = np.concatenate((pos_obs[:,1], vel_obs))
    order = 15
    #DAIOD
    DA.init(order,6)
    X0_CC_ECI_epoch = DAIOD_full(RA_rad, DEC_rad, ang_sigma, ang_sigma, pos_obs, observation_times_seconds, mu, order, prograde=True)

    #PW Monte Carlo xyz
    results = PWIOD.monte_carlo_gauss_PWiod(10000, pos_obs, RA_rad, DEC_rad, observation_times_seconds, ang_sigma, mu=3.986e5)

    # Further processing or assertions can be done here
    #X0_relative_pos = X0_CC_epoch[:3] - eci_observation_positions[:,1] #check X0_CC_epoch
    #X0_relative_vel = X0_CC_epoch[3:] - eci_observation_velocities[:,1]
    #X0_relative = X0_relative_pos.concat(X0_relative_vel)
    # Convert from ECI Cartesian to Spherical to compare with Pirovano
    X0_CC_topo = X0_CC_ECI_epoch - X_ECI_observation_epoch
    results_CC_topo = results - X_ECI_observation_epoch

    Obs_relative = time_ref.CC2obs(X0_CC_topo) # The Constant RA and DEC are different by 1e-2 rad - This is seen from Earth Centre.

    results_obs = np.zeros_like(results_CC_topo)
    for i in range(results_CC_topo.shape[0]):
        results_obs[i,:] = time_ref.CC2obs(results_CC_topo[i,:])

    #Evaluation of DA objects
    Obs_relative_cons = Obs_relative.cons()

    #prepare output array
    n_perimeter_points = Obs_perimeter_norm.shape[0]
    n_observables = len(Obs_relative)
    Obs_relative_perimeter = np.zeros((n_perimeter_points, n_observables))
    #Loop through each perimeter point
    for i in range(n_perimeter_points):
        eval_point = Obs_perimeter_norm[i, :].tolist()  #TODO: Problem is here (Not evaluating properly to form a box)
        evaluated_obs = Obs_relative.eval(eval_point)
        Obs_relative_perimeter[i, :] = evaluated_obs

        if i % 10 == 0:
            print(f"Evaluated {i+1}/{n_perimeter_points} perimeter points")


    #Plots for comparison
    #RA vs DEC 
    fig, ax = plt.subplots()
    ax.scatter(Obs_relative_perimeter[:,0], Obs_relative_perimeter[:,1], color='blue', label='DAIOD', alpha=0.5)
    ax.scatter(results_obs[:,0], results_obs[:,1], color='red', label='PW Gauss IOD', alpha=0.5)
    ax.set_xlabel("RA (Rad)")
    ax.set_ylabel("DEC (Rad)")
    ax.set_title("RA and DEC Observation Perimeter")
    ax.grid()
    ax.legend()
    plt.show()

    plt.figure(2)
    plt.scatter(Obs_relative_perimeter[:,2], Obs_relative_perimeter[:,5], color='blue')
    plt.scatter(results_obs[:,2], results_obs[:,5], color='red')
    plt.xlabel("Range")
    plt.ylabel("Range-Rate")
    plt.title("Range and Range-Rate Observation Perimeter")
    plt.grid()
    plt.legend(["DAIOD", "PW Gauss IOD"])
    plt.show()

def test_Lambert_Guass():
    """
        Testing Gauss + DAIOD for a Gauss compliant observations (Earth)
    """
    _, _, t, _, r_vec, v_2 = sample_data_Earth()

    # Pass Gauss solution into Lambert
    r1 = r_vec[:, 0]
    r2 = r_vec[:, 1]
    dt = t[1] - t[0]  # Time difference between first two observations
    mu = 3.986e5

    v1, v2 = izzo(mu, r1, r2, dt, 0, prograde=True, lowpath=False, numiter=20, rtol=1e-9)
    velocities = lambert_izzo(r1, r2, dt, mu=3.986e5, multi_revs=0, prograde=True)
    solution = velocities[0]
    
    v2_mine = solution[:,1]
    print(f"The Gauss Solution {v_2}")
    print(f"The Poliastro Lambert Solution {v2}")
    print(f"My solution {v2_mine}")

    assert np.allclose(v2, v2_mine)

def test_Lamber_Guass_DA():
    _, _, t, _, r_vec, v_2 = sample_data_Earth()

    # Pass Gauss solution into Lambert
    r1 = r_vec[:, 0]
    r2 = r_vec[:, 1]
    dt = t[1] - t[0]  # Time difference between first two observations
    mu = 3.986e5

    v1, v2 = izzo(mu, r1, r2, dt, 0, prograde=True, lowpath=False, numiter=20, rtol=1e-9)

    DA.init(4,6)
    r1 = array([r1[i] + DA(i+1) for i in range(3)])
    r2 = array([r2[i] + DA(i+4) for i in range(3)])

    velocities = lambert_izzo(r1, r2, dt, mu=3.986e5, multi_revs=0, prograde=True)

    solution = velocities[0]
    
    v2_mine = solution[:,1]
    print(f"The Gauss Solution {v_2}")
    print(f"The Poliastro Lambert Solution {v2}")
    print(f"My solution {v2_mine.cons()}")

    assert np.allclose(v2, v2_mine.cons())

def test_DAIOD_1_Gauss_Lambert_Earth():
    """
        Testing Gauss + DAIOD for a Lambert compliant observations (Heliocentric)
    """

    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth() 

    range_mag_L1, J_dv = DAIOD_1Scipy_invert(range_mag, obs_dir, t, 4, pos_obs, mu=3.986e5, tol=1e-3)

    assert range_mag_L1.cons().shape == range_mag.shape
    assert J_dv.shape == np.zeros((3,3)), "Jacobian needs to be 3x3"
    print(f"The Guass Range Magnitude:\n{range_mag}")
    print(f"The Refined Range Magnitude:\n{range_mag_L1}")

def test_DAIOD_1_Gauss_Lambert_Earth_Scipy():
    """
    Testing DAIODD_1 with the Scipy solver to get the initial condition
    """
    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth() 

    range_mag_L1 = DAIOD_1Scipy(range_mag, obs_dir, t, 4, pos_obs, mu=3.986e5, tol=1e-6)

    #Jacobian Testing - Compare range_mag_L1 jacobian with Scipy Version 
    range_mag_float, range_mag_float_JAC = DAIOD_1_debugSciPy(range_mag, obs_dir, t, 4, pos_obs, mu=3.986e5, tol=1e-6)

    DA_JAC = array([range_mag_L1.deriv(i+1) for i in range(3)])
    scipy_JAC = range_mag_float_JAC
    
    # Jacobian
    assert np.allclose(DA_JAC, scipy_JAC, rtol=1e-5), "Numerical and DA derived Jacobian must be same"
    # Shape Assertion
    assert range_mag_L1.cons().shape == range_mag.shape
    # New Range should be close to Gauss range
    assert np.allclose(range_mag_L1.cons(), range_mag, rtol=1e-1), "ERROR: Range has diverged from Gauss solution"
    assert np.allclose(range_mag, range_mag_float, "Scipy Nominal Result Inconsistent")





def test_DAIOD_case_2():
    """
        Integrated Test of the DAIOD Algorithm: Case 2 = Iterative Improvement of range_mag with Angle Variables
    """
    range_mag, obs_dir, t, pos_obs, position, v_2 = sample_data_Earth() 

    RA_nom = np.array([0.759857367, 0.949802813, 1.12254676])
    DEC_nom = np.array([-0.15329974, -0.210726107, -0.263633267])
    RA_sigma = 1e-3
    DEC_sigma = 1e-4

    range_mag_DA_drange, J_dv = DAIOD_1Scipy_invert(range_mag, obs_dir, t, 4, pos_obs, mu=3.986e5, tol=1e-3)
    
    DA.init(4,6)
    dRA = (3*RA_sigma*array.identity(3))
    dDEC = array([3*DEC_sigma*DA(i+1) for i in range(3,6)])
    RA = RA_nom + dRA
    DEC = DEC_nom + dDEC

    range_mag_DA_dangles = DAIOD_2(range_mag_DA_drange, RA, DEC, t, 4, pos_obs, mu=3.986e5, J=J_dv)

    assert range_mag_DA_dangles.cons().shape == range_mag.shape, "Range Magnitude shape mismatch"
    assert np.allclose(range_mag_DA_dangles.cons(), range_mag_DA_drange.cons(), rtol=1e-2), "ERROR: Range has diverged from Gauss solution"
    assert np.allclose(range_mag_DA_dangles.cons(), range_mag, rtol=1e-2), "ERROR: Range has diverged from Gauss solution significantly"



# Textbook Example - Performance test
def test_Guass_8th_seed_basic():
    # Use simple, linearly independent vectors for observer positions and directions
    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          # Observer position [x components at all instances]
        [3430.2, 3460.1, 3490.1],          # Observer position [y components at all instances]
        [4078.5, 4078.5, 4078.5]           # Observer position [z components at all instances]
    ])

    obs_dir = np.array([            # Direction vectors of the observations
        [0.71643, 0.56897, 0.41841],            # Direction vector for x components
        [0.68074, 0.79531, 0.87007],            # Direction vector for y components
        [-0.15270,-0.20917,-0.26059]            # Direction vector for z components
    ])

    t = np.array([0.0, 118.10, 237.58])


    # Should not raise and should return two lists of length 3
    position, ranges, range_mag, v_2 = Guass_8th_seed(pos_obs, obs_dir, t, mu=3.986e5)
    
    r1_textbook_ex = np.array([
        6096.9,
        5907.5,
        3522.9])
    
    r2_textbook_ex = np.array([
        5659.1,
        6533.8,
        3270.1])
    
    r3_textbook_ex = np.array([
        5169.1,
        7107.0,
        2995.3])
    
    assert isinstance(position, np.ndarray)
    assert isinstance(ranges, np.ndarray)
    assert len(position) == 3
    assert len(ranges) == 3
    assert np.isclose(np.linalg.norm(ranges[:,2]), range_mag[2])

    assert np.allclose(position[:,1], r2_textbook_ex, rtol=1e-3)
    assert np.allclose(position[:,0], r1_textbook_ex, rtol=1e-3)
    assert np.allclose(position[:,2], r3_textbook_ex, rtol=1e-3)

def test_Guass_Arc_length():
    """
    Try and Break Guass through arc length
    """
    import astropy.coordinates as acoords
    import astropy.units as u
    import astropy.constants as ac
    from astropy.time import Time, TimeDelta

    #Vary observation arc times
    dt = np.linspace(0.01, 10, 100) # days
    dt_astropy = TimeDelta(dt, format='jd')
    obs_time0 = Time('2005-01-17T01:19:10.000', format='isot', scale='ut1')
    Guass_key = []
    for j in range(len(dt)):
        print(j)
        obs_times = Time([obs_time0 - dt_astropy[j], obs_time0, obs_time0 + dt_astropy[j]])

        #THIS IS CORRECT
        mu = ac.G * ac.M_sun
        mu = mu.to('km**3 / s**2').value
        epochs = Time(obs_times, scale='tdb')
        acoords.solar_system_ephemeris.set("builtin")
        #Get ICRS Equatorial position of the Earth at each observation epoch
        pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]          # pv[i] = (pos, vel)
        #Translate to Heliocentric Equatorial Frame
        pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
        pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]  # (pos, vel) relative to Sun
        for i, (p_helio, v_helio) in enumerate(pv_earth_helio):
            #Maintain Equaotorial Frame
            pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))  # [x, y, z, vx, vy, vz]

        pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T  # km [3xN]

        #Get RA, DEC
        obs_times_yyyymmdd = obs_times.to_value('datetime').astype('datetime64[D]')
        interval_1 = obs_times_yyyymmdd[1] - obs_times_yyyymmdd[0]
        interval_2 = obs_times_yyyymmdd[2] - obs_times_yyyymmdd[1]

        interval1_days = TimeDelta(interval_1.astype(int), format='jd')
        interval2_days = TimeDelta(interval_2.astype(int), format='jd')

        if interval1_days.value <= 90: 
            interval1 = f"{int(interval1_days.value)}d"
        else:
            raise ValueError("Interval too long")
        
        if interval2_days.value <= 90:
            interval2 = f"{int(interval2_days.value)}d"
        else:
            raise ValueError("Interval too long")


        eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')
        _, vec = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1', location='500@10')

        _, vec2 = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1', location='500@10')
        eph2, _  = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')

        ra = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad).value
        dec = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad).value
        apophis_pos = ((np.concatenate([vec['x'], vec2['x'][1:]]) , np.concatenate([vec['y'], vec2['y'][1:]]), np.concatenate([vec['z'], vec2['z'][1:]])) * u.AU).to(u.km)
        apophis_vel = ((np.concatenate([vec['vx'], vec2['vx'][1:]]), np.concatenate([vec['vy'], vec2['vy'][1:]]), np.concatenate([vec['vz'], vec2['vz'][1:]])) * (u.AU/u.d)).to(u.km/u.s)
        apophis_vec = np.concatenate([apophis_pos.value, apophis_vel.value], axis=0) #km

        obs_dir = time_ref.create_da_los_vectors(ra, dec)
        position, range_vec, range_mag, v_2 = Guass_8th_seed(pos_obs, obs_dir, (obs_times.mjd * u.day).to(u.s).value, mu)

        #checks - If not Nan: convert from CC2COE then compare with known values. Ensure its in the ballpark doesn't have to be perfect.
        Apophis_truth_GEO_COE = time_ref.CC2COE(apophis_vec[:3,1], apophis_vec[3:,1], mu)
        if not (np.isnan(position).any() or np.isnan(v_2).any() or np.isnan(range_mag).any() or np.isnan(range_vec).any()): #obtain root perfectly
            # 1 Outcome: Only 1 root
            # 2 Outcome: Multiple roots pass
            #Assess the epoch State
            COE = time_ref.CC2COE(position[:,1], v_2, mu=mu)# Take middle observation as reference
            a, e, i, raan, argp, nu = COE
            a_truth, e_truth, i_truth, raan_truth, argp_truth, nu_truth = Apophis_truth_GEO_COE
            try:
                assert np.isclose(a, a_truth, rtol=1e-1)  # Semi-major axis should be within 0.01 AU
                assert np.isclose(e, e_truth, atol=5e-1)
                assert np.isclose(np.rad2deg(i), np.rad2deg(i_truth), atol=2) # 2 degrees
                assert np.isclose(np.rad2deg(raan), np.rad2deg(raan_truth), atol=2)
                assert np.isclose(np.rad2deg(argp), np.rad2deg(argp_truth), atol=2)
                

                for k in range(len(range_mag)):
                    range_unit = range_vec[:,k] / range_mag[k]
                    assert np.isclose(np.linalg.norm(range_unit), 1, rtol=1e-1)
                
                Guass_key.append(True)

            except AssertionError as e:
                print(f"Assertion error: {e}, Gauss Converged but not in tolerance")
                Guass_key.append(True)

        else:
            print("All roots failed: Testing if NaN exisits throughout")
            try:
                assert np.isnan(position).all()
                assert np.isnan(v_2).all()
                assert np.isnan(range_vec).all()
                Guass_key.append(True)

            except AssertionError as e:
                print(f"Assertion error: {e}, Gauss Failed but NaN does not exist throughout")
                Guass_key.append(False)


    assert all(Guass_key) is True, "Gauss Failed at some point"

def test_DAIOD_Arc_length():
    """Error Handling of DAIOD 1,2,3 with varying arc length"""
    import astropy.coordinates as acoords
    import astropy.units as u
    import astropy.constants as ac
    from astropy.time import Time, TimeDelta

    #Vary observation arc times
    dt = np.linspace(0.01, 10, 100) # days
    dt_astropy = TimeDelta(dt, format='jd')
    obs_time0 = Time('2005-01-17T01:19:10.000', format='isot', scale='ut1')
    Guass_key = []

    for j in range(len(dt)): # The Arc length
        print(j)
        obs_times = Time([obs_time0 - dt_astropy[j], obs_time0, obs_time0 + dt_astropy[j]])

        #THIS IS CORRECT
        mu = ac.G * ac.M_sun
        mu = mu.to('km**3 / s**2').value
        epochs = Time(obs_times, scale='tdb')
        acoords.solar_system_ephemeris.set("builtin")
        #Get ICRS Equatorial position of the Earth at each observation epoch
        pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]          # pv[i] = (pos, vel)
        #Translate to Heliocentric Equatorial Frame
        pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
        pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]  # (pos, vel) relative to Sun
        for i, (p_helio, v_helio) in enumerate(pv_earth_helio):
            #Maintain Equaotorial Frame
            pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))  # [x, y, z, vx, vy, vz]

        pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T  # km [3xN]

        #Get RA, DEC
        obs_times_yyyymmdd = obs_times.to_value('datetime').astype('datetime64[D]')
        interval_1 = obs_times_yyyymmdd[1] - obs_times_yyyymmdd[0]
        interval_2 = obs_times_yyyymmdd[2] - obs_times_yyyymmdd[1]

        interval1_days = TimeDelta(interval_1.astype(int), format='jd')
        interval2_days = TimeDelta(interval_2.astype(int), format='jd')

        if interval1_days.value <= 90: 
            interval1 = f"{int(interval1_days.value)}d"
        else:
            raise ValueError("Interval too long")
        
        if interval2_days.value <= 90:
            interval2 = f"{int(interval2_days.value)}d"
        else:
            raise ValueError("Interval too long")


        eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')
        _, vec = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1', location='500@10')

        _, vec2 = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1', location='500@10')
        eph2, _  = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')

        ra = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad).value
        dec = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad).value
        apophis_pos = ((np.concatenate([vec['x'], vec2['x'][1:]]) , np.concatenate([vec['y'], vec2['y'][1:]]), np.concatenate([vec['z'], vec2['z'][1:]])) * u.AU).to(u.km)
        apophis_vel = ((np.concatenate([vec['vx'], vec2['vx'][1:]]), np.concatenate([vec['vy'], vec2['vy'][1:]]), np.concatenate([vec['vz'], vec2['vz'][1:]])) * (u.AU/u.d)).to(u.km/u.s)
        apophis_vec = np.concatenate([apophis_pos.value, apophis_vel.value], axis=0) #km

        ra_sigma = (eph['RA_3sigma'][1] * u.arcsec).to(u.rad).value/3
        dec_sigma = (eph['DEC_3sigma'][1] * u.arcsec).to(u.rad).value/3

        if j == 4:
            print(j)
        X0_Obj_ECI_DAIOD_ADS = DAIOD_ADS_full(ra, dec, ra_sigma, dec_sigma, pos_obs, (obs_times.mjd * u.day).to(u.s).value, mu, 4, prograde=True)
        if isinstance(X0_Obj_ECI_DAIOD_ADS[0], ADS):
            X0_domain = X0_Obj_ECI_DAIOD_ADS[0].manifold.cons()
        else:
            X0_domain = X0_Obj_ECI_DAIOD_ADS

        if not np.isnan(X0_domain).any(): #test 1 subdomain
            # A output is generated (although we don't know how good)
            try: # Compare the Cartesian vectors (position and velocity)
                X_vec = X0_domain
                r_norm = np.linalg.norm(X_vec[:3])
                v_norm = np.linalg.norm(X_vec[3:])

                #check mags - light tolerance
                assert np.isclose(r_norm, np.linalg.norm(apophis_vec[:,1][:3]), rtol = 1e-1)
                assert np.isclose(v_norm, np.linalg.norm(apophis_vec[:,1][3:]), rtol=5e-1)
                Guass_key.append(True)

            except AssertionError as e: #No where close to true value
                print(f"FAILED {e}: Large discrepancy between predicted and true state")
                Guass_key.append(False)

        else:
            #NaNs are produced
            try: # assert all nans are produced by DAIOD if wrong
                X_vec = X0_domain
                assert(np.isnan(X_vec).all())
                Guass_key.append(True)
            except AssertionError as e: 
                print("Failed ")
                Guass_key.append(False)
    

    assert all(Guass_key) is True
    


# Functional Example Test
def test_Guass_8th_seed_invalid_coeffs():
    # Use vectors that will produce NaN or inf in coefficients
    pos_obs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    obs_dir = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    t = np.array([0.0, 0.0, 0.0])

    with pytest.raises(AssertionError):
        Guass_8th_seed(pos_obs, obs_dir, t)

def test_Guass_8th_seed_no_real_roots():
    # Use vectors that will produce no positive real roots
    pos_obs = np.array([
        [-1.0e8, 0.0, 0.0],
        [0.0, -1.0e8, 0.0],
        [0.0, 0.0, -1.0e8]
    ])
    obs_dir = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    t = np.array([0.0, 1.0e4, 2.0e4])

    with pytest.raises(AssertionError):
        Guass_8th_seed(pos_obs, obs_dir, t)


# Functional tests 
def test_f_g_series_v0_zero_vector():
    # Provide a v0 that is a zero vector - see if works with no velcoity
    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.zeros(3)  # Zero vector
    dt = 1000.0
    f_order = 4
    g_order = 4
    f, g = f_g_series(r0, dt, f_order, g_order, v0=v0)
    # Should still return valid f and g values
    assert isinstance(f, float)
    assert isinstance(g, float)

def test_f_g_series_v0_nonzero_vector():
    # Provide a nonzero v0
    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.array([1.0e3, 2.0e3, 3.0e3])  # Nonzero vector
    dt = 1000.0
    f_order = 5
    g_order = 5
    f, g = f_g_series(r0, dt, f_order, g_order, v0=v0)
    assert isinstance(f, Union[float,int])
    assert isinstance(g, Union[float,int])

def test_f_g_series_v0_invalid_shape():
    # Provide a v0 that is not a 3D vector

    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.array([1.0, 2.0])  # Invalid shape
    dt = 1000.0
    with pytest.raises(AssertionError):
        f_g_series(r0, dt, 3, 3, v0=v0)

def test_f_g_series_IOD():
    # Provide no v0 vector - see if v0 is None part works.
    r0 = np.array([1.0e8, 0.0, 0.0])
    dt = 1000.0
    f_order = 2
    g_order = 3
    f, g = f_g_series(r0, dt, f_order, g_order)

    # Should still return valid f and g values
    assert isinstance(f, float)
    assert isinstance(g, float)

# ---------------------------------------------------------------------
#Performance tests (textbook examples)

def test_f_g_series_IOD_typical():
    # Textbook example with typical values - Orbital mechancis for Engineering Students Ex: 5: Guass example
    # IOD as no v0 is provided
    r0 = np.array([5659.1, 6533.8, 3270.1])
    dt = -118.10
    f_order = 2
    g_order = 3
    f, g = f_g_series(r0, dt, f_order, g_order, mu=3.986e5)
    # Should return valid f and g values
    # Replace 1.234 with the textbook value you want to check against
    f_textbook_value = 0.99648
    g_textbook_value = -117.97
    assert np.isclose(f, f_textbook_value, rtol=1e-3), f"Expected {f_textbook_value}, got {f}"
    assert np.isclose(g, g_textbook_value, rtol=1e-3), f"Expected {g_textbook_value}, got {g}"