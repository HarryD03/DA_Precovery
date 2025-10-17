import numpy as np
from typing import Union
from daceypy import DA, array, ADS
import daceypy.op as op
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve      # or numpy.linalg for tiny systems
import scipy.linalg as la
import utils.time_reference as time_ref
from utils.lambert_izzo import lambert_izzo

def DAIODfunc(domain: ADS, observer_position: NDArray, time_observation_seconds: NDArray, mu: float, order: int, prograde: bool) -> ADS:

    RA_DA = domain.box[:3]
    DEC_DA = domain.box[3:]
    RA_rad = RA_DA.cons()
    DEC_rad = DEC_DA.cons()
    # Unit Vector
    i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad) 

    # Guass seed - seed for newton raphson
    positions, ranges, range_mags, v_2 = Guass_8th_seed(observer_position, i_rho, time_observation_seconds, mu)
    
    if not (np.isnan(positions).any() or np.isnan(ranges).any() or np.isnan(range_mags).any() or np.isnan(v_2).any()):
        print(f"GAUSS passed:")
    else:
        print(f"GAUSS failed:")
        X = []
        X.append(list(positions[:,1]))
        X.append(list(v_2))
        return np.array(X).flatten()

    #DAIOD_1 - DA inversion for range_mag_L1
    range_mag_L1, Jacobian_dv = DAIOD_1Scipy_invert(range_mags, i_rho, time_observation_seconds, order, observer_position, mu,tol=1e-9, prograde_bool=prograde)
    
    #DAIOD2
    range_mag_DAangles = DAIOD_2(range_mag_L1, RA_DA, DEC_DA, time_observation_seconds, order, r_obs_heliocentric=observer_position, mu=mu, J=Jacobian_dv, prograde_bool=prograde)

    #DAIOD_3 - Full state definition
    i_rho_DA = array(time_ref.create_da_los_vectors(RA_DA, DEC_DA)) 
    range_vec = i_rho_DA * range_mag_DAangles
    pos_vec = range_vec + observer_position    
        
    dt1 = time_observation_seconds[1] - time_observation_seconds[0]
    dt2 = time_observation_seconds[2] - time_observation_seconds[1]

    vel = []
    velocities1 = lambert_izzo(pos_vec[:,0], pos_vec[:,1], dt1, mu, 0, prograde=prograde)
    vel.append(velocities1[0])  # Unpack the first solution

    velocities2 = lambert_izzo(pos_vec[:,1], pos_vec[:,2], dt2, mu, 0, prograde=prograde)
    vel.append(velocities2[0])  # Unpack the first solution

    v1 = velocities1[0][:,0]
    v2 = velocities1[0][:,1]
    v3 = velocities2[0][:,1]     #Velocity Vector (CC) Heliocentric Ecliptic
        
    vel_vec = v1.concat(v2).concat(v3)  #Concatenate the velocity vectors

    if vel_vec.shape != (3,3):
        print(f"ERROR: Velocity vector shape mismatch\nreshaping...\n")
        vel_vec = np.array(vel_vec, dtype=object)
        vel_vec = vel_vec.reshape((3,3)).T
        vel_vec = array(vel_vec)

    X0_CC = pos_vec.concat(vel_vec)   # NEO Cartesian State at all positions
    X0_CC_epoch = X0_CC[:,1].copy()

    return ADS(domain.box, domain.nsplit, X0_CC_epoch)

def DAIOD_ADS_full(RA_rad, DEC_rad, RA_sigma_rad, DEC_sigma_rad, observer_position, time_observation_seconds, mu, order, prograde=False):
    """
    Full Differential Algebraic Implicit Orbit Determination (DAIOD) function:
    - Conduct DA initialization outside of the function
    - All coordinate transformations outside function (must be consistent and inertial)
    :params RA_rad: Right Ascension of Observations. Structure (1x3)
    :params DEC_rad: Declination of Observations. Structure (1x3)
    :params RA_sigma_rad: Uncertainty in Right Ascension. Structure (1)
    :params DEC_sigma_rad: Uncertainty in Declination. Structure (1)
    :params observer_position: Position of the observer. Structure (3x3)
    :params time_observation_seconds: Time of observations. Structure (1x3)
    :params mu: Gravitational parameter. Scalar
    :params order: Order of the DA method. Scalar
    :params prograde_bool: Prograde flag. Boolean

    :return X0_CC_epoch: DA Initial state vector at epoch. Taylor Map wrt RA/DEC. Structure (6x1) 
    """

    assert RA_rad.shape == (3,), "RA_rad must be a 1D array with 3 elements"
    assert DEC_rad.shape == (3,), "DEC_rad must be a 1D array with 3 elements"

    DA.init(order, 6)

    # Unit Vector
    i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad) 

    # Guass seed - gate for error handling
    positions, ranges, range_mags, v_2 = Guass_8th_seed(observer_position, i_rho, time_observation_seconds, mu)

    if not (np.isnan(positions).any() or np.isnan(ranges).any() or np.isnan(range_mags).any() or np.isnan(v_2).any()):
        print(f"GAUSS passed:")
    else:
        print(f"GAUSS failed:")
        X = []
        X.append(list(positions[:,1]))
        X.append(list(v_2))
        return np.array(X).flatten()

    if range_mags.shape != (3,):
        print(f"ERROR: Range mags shape mismatch\nreshaping...\n")
        range_mags = range_mags.reshape(-1)

    #DAIOD_2 - DA expansion as angles
    # Create NORMALIZED domain variables (this is key!)
    RA_DA = array([RA_rad[i] + 3*RA_sigma_rad*DA(i + 1) for i in range(3)])
    DEC_DA = array([DEC_rad[i] + 3*DEC_sigma_rad*DA(i + 4) for i in range(3)])
    
    # Set tolerances to be a fraction of the maximum variation
    pos_tol = 1e-3  # Pirovano Defined 1km
    vel_tol = 1e-3  # Priovano Defined 1m/s

    nsplits_limit = 10  # Pirovano Defined 10 splits max

    domain0 = RA_DA.concat(DEC_DA)

    init_domain = ADS(domain0, [])
    init_list = [init_domain]
    tol = np.array([pos_tol, pos_tol, pos_tol, vel_tol, vel_tol, vel_tol])
    #Conduct ADS
    final_list = ADS.eval(init_list, tol, nsplits_limit, lambda domain: DAIODfunc(domain, observer_position, time_observation_seconds, mu, order, prograde))

    print(f"ADS completed. Number of final domains: {len(final_list)}")

    X0_CC_epoch_ADS = final_list

    return X0_CC_epoch_ADS 

def DAIOD_full(RA_rad, DEC_rad, RA_sigma_rad, DEC_sigma_rad, observer_position, time_observation_seconds, mu, order, prograde=False):
    """
    Full Differential Algebraic Implicit Orbit Determination (DAIOD) function:
    - Conduct DA initialization outside of the function
    - All coordinate transformations outside function (must be consistent and inertial)
    :params RA_rad: Right Ascension of Observations. Structure (1x3)
    :params DEC_rad: Declination of Observations. Structure (1x3)
    :params RA_sigma_rad: Uncertainty in Right Ascension. Structure (1)
    :params DEC_sigma_rad: Uncertainty in Declination. Structure (1)
    :params observer_position: Position of the observer. Structure (3x3)
    :params time_observation_seconds: Time of observations. Structure (1x3)
    :params mu: Gravitational parameter. Scalar
    :params order: Order of the DA method. Scalar
    :params prograde_bool: Prograde flag. Boolean

    :return X0_CC_epoch: DA Initial state vector at epoch. Taylor Map wrt RA/DEC. Structure (6x1) 
    """

    assert RA_rad.shape == (3,), "RA_rad must be a 1D array with 3 elements"
    assert DEC_rad.shape == (3,), "DEC_rad must be a 1D array with 3 elements"

    # Unit Vector
    i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad) 

    # Guass seed - seed for newton raphson
    positions, ranges, range_mags, v_2 = Guass_8th_seed(observer_position, i_rho, time_observation_seconds, mu)
    
    if not (np.isnan(positions).any() or np.isnan(ranges).any() or np.isnan(range_mags).any() or np.isnan(v_2).any()):
        print(f"GAUSS passed:")
    else:
        print(f"GAUSS failed:")
        X = []
        X.append(list(positions[:,1]))
        X.append(list(v_2))
        return np.array(X).flatten()

    #error clause
    if not (np.isnan(positions).any() or np.isnan(ranges).any() or np.isnan(range_mags).any() or np.isnan(v_2).any()):
        print(f"GAUSS passed:")
        print(f"  positions: {positions}")
        print(f"  ranges: {ranges}")
        print(f"  range_mags: {range_mags}")
        print(f"  v_2: {v_2}")
    else:
        print(f"GAUSS failed:")
        print(f"  positions: {positions}")
        print(f"  ranges: {ranges}")
        print(f"  range_mags: {range_mags}")
        print(f"  v_2: {v_2}")
        return None




    #DAIOD_1 - DA inversion for range_mag_L1
    range_mag_L1, Jacobian_dv = DAIOD_1Scipy_invert(range_mags, i_rho, time_observation_seconds, order, observer_position, mu,tol=1e-6, prograde_bool=prograde)
    

    #DAIOD_2 - DA expansion as angles
    RA_DA = array([RA_rad[i] + 3*RA_sigma_rad*DA(i + 1) for i in range(3)])
    DEC_DA = array([DEC_rad[i] + 3*DEC_sigma_rad*DA(i + 4) for i in range(3)])    # Scale RA and DEC DA part so that when evaluated its within the [-1,1] range
    range_mag_DAangles = DAIOD_2(range_mag_L1, RA_DA, DEC_DA, time_observation_seconds, order, observer_position, mu, Jacobian_dv, prograde_bool=prograde)

    #DAIOD_3 - Full state definition
    i_rho_DA = array(time_ref.create_da_los_vectors(RA_DA, DEC_DA)) 
    range_vec = i_rho_DA * range_mag_DAangles
    pos_vec = range_vec + observer_position    
    
    dt1 = time_observation_seconds[1] - time_observation_seconds[0]
    dt2 = time_observation_seconds[2] - time_observation_seconds[1]

    vel = []
    velocities1 = lambert_izzo(pos_vec[:,0], pos_vec[:,1], dt1, mu, 0, prograde=prograde)
    vel.append(velocities1[0])  # Unpack the first solution

    velocities2 = lambert_izzo(pos_vec[:,1], pos_vec[:,2], dt2, mu, 0, prograde=prograde)
    vel.append(velocities2[0])  # Unpack the first solution

    v1 = velocities1[0][:,0]
    v2 = velocities1[0][:,1]
    v3 = velocities2[0][:,1]     #Velocity Vector (CC) Heliocentric Ecliptic
    
    vel_vec = v1.concat(v2).concat(v3)  #Concatenate the velocity vectors

    if vel_vec.shape != (3,3):
        print(f"ERROR: Velocity vector shape mismatch\nreshaping...\n")
        vel_vec = np.array(vel_vec, dtype=object)
        vel_vec = vel_vec.reshape((3,3)).T
        vel_vec = array(vel_vec)

    X0_CC = pos_vec.concat(vel_vec)   # NEO Cartesian State at all positions
    X0_CC_epoch = X0_CC[:,1].copy()

    return X0_CC_epoch 





def DAIOD(RA: Union[NDArray,array], DEC: Union[NDArray,array], range_mag: Union[NDArray,array], r_obs_heliocentric: NDArray, t_obs_s: NDArray, mu):

    def f(range_mag):                                       #deltV = residual + M(dranges)
    
        """
        f(range_vec) returns the velocity difference between the second and first velocity estimates
        for the central observation. The velocity difference is calculated by first calculating the
        positions of the observations using the range vector and the observer position. The positions
        are then used to calculate the velocities between each observation via the Izzo solution to
        Lambert's problem. The velocity difference is then calculated as the difference between the
        second and first velocity estimates. This function is used in the Newton method to find the
        root of the velocity difference.
        """
        range_vec = np.zeros_like(i_rho)
        for i in range(0, len(range_mag)):
            range_vec[:,i] = op.dot(range_mag[i], i_rho[:,i])

        r_vec = np.zeros_like(range_vec)
        for i in range(0, len(range_vec)):                  #define posiiton vector for lamber_izzo
            r_vec[:,i] = range_vec[:,i] + r_obs_heliocentric[:,i]

        vel = []
        for i in range(0, len(r_vec) - 1):                  #calculate the velocities via lamerts problem 
            velocities = lambert_izzo(r_vec[:,i], r_vec[:,i+1], t_obs_s[i+1] - t_obs_s[i], mu, 0, cw=False)

            #unpack solutions 
            solution = velocities[0]

            v1 = solution[:,i]
            v2 = solution[:,i+1]

            print(f"v1: {v1.cons()}\n")                 #debugging purposes
            print(f"v2: {v2.cons()}\n")                 #debugging purposes

            vel.append(v1)
            vel.append(v2)

        # Centre posiitons should have zero velocity difference
        v2_plus = vel[2]                
        v2_minus = vel[1]

        DV = (v2_plus - v2_minus)

        return DV                           #DV = [dv_i, dv_j, dv_k] + M(dranges)

    def f1(range_vec):
        """
        Delta V Function for Angle Variables
        """

        r_vec = np.zeros_like(range_vec)
        for i in range(0, len(range_vec)):
            r_vec[:,i] = range_vec[:,i] + r_obs_heliocentric[:,i]

        vel = []
        for i in range(0, len(r_vec) - 1):                  #calculate the velocities via lamerts problem 
            velocities = lambert_izzo(r_vec[:,i], r_vec[:,i+1], t_obs_s[i+1] - t_obs_s[i], mu, 0, cw=False)

            #unpack solutions 
            solution = velocities[0]

            v1 = solution[:,i]
            v2 = solution[:,i+1]

            print(f"v1: {v1.cons()}\n")                 #debugging purposes
            print(f"v2: {v2.cons()}\n")                 #debugging purposes

            vel.append(v1)
            vel.append(v2)

        # Centre posiitons should have zero velocity difference
        v2_plus = vel[2]                
        v2_minus = vel[1]

        DV = (v2_plus - v2_minus)

        return DV

    if not isinstance(RA, array) and not isinstance(DEC, array): #Case 1: Obtain DA Range Vector
        
        assert(isinstance(range_mag, np.ndarray)), "Range Vector is not Initialised in DA"
        range_mag_guass = range_mag
        tol = 1e-8

        p0 = np.zeros_like(range_mag_guass)                                        
        range_mag_L1 = newton_nominal_DAVec(range_mag_guass, p0, f, tol, 100)      #Obtain Range so that f(Range(0)) = 0
        
        p = array([p0[i] + DA(i+1) for i in range(len(range_mag))])                #Define Range polynmoinal map
        range_mag_L1 = newton_nominal_DAVec(range_mag_guass, p, f, tol, 100)    

        return range_mag_L1        #DA output


    if isinstance(RA, array) and isinstance(DEC, array):        #Case 2: DAIOD with Range DA Variables dependant on angle DA.
        assert(isinstance(range_mag, array)), "Range Vector must be a DA variable"
        
        i_rho = time_ref.create_da_los_vectors(RA, DEC)          #returns 3x3 matrix   
                      #Line of sight unit vector
        range_vec0 = op.dot(range_mag.cons(), i_rho)             #Taylor Polynomial Map
        range_vec = Implicit_solver_DAVec(range_vec0, 0, f1, 6, x0DA=False, DAIOD=True, JacDAIOD=op.dot(range_mag.linear(), i_rho.cons()))
        r_vec = np.zeros_like(range_vec)

        for i in range(0, len(range_vec)):
            r_vec[:,i] = range_vec[:,i] + r_obs_heliocentric[:,i]

    velocities = []
    velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, cw=True)
    solution = velocities[0]

    v1 = solution[:,0]
    v2 = solution[:,1]

    print(f"v1: {v1.cons()}\n")                 #debugging purposes
    print(f"v2: {v2.cons()}\n")                 #debugging purposes

    rv = array([r_vec[:,1], v2])               #Define Epoch state of arc - DA parts correspond to observation uncertanity 

    return rv

def DAIOD_1(range_mag_guass: float, i_rho: np.ndarray, t_obs_s: np.ndarray, order, r_obs_heliocentric: np.ndarray = np.zeros((3, 3)), mu: float=1.32712e11, tol: float=1e-9):
    """
        Part 1 of the DAIOD Algorithm: Define the ranges such that DV(range_mag(p)) = 0

        :param range_mag_guass: Range magnitude. See Guass_8th_seed() [km]
        :param i_rho: Line of sight unit vector. See create_da_los_vectors() [-]
        :param t_obs_s: The time of the observation. [s]
        :param r_obs_heliocentric: Position of the observer. Assumed heliocentric plane, centre of Earth.  [km]
        :param mu: The standard gravitational parameter. Assumed Sun Orbiting [km^3/s^2]
        :param tol: The tolerance for the Newton method. [km]

        :return range_mag_L1: Range Taylor Polynomial Map. range_mag_L1 + variations  [km]
    """

    def f(x, p):                                      #deltV = residual + M(dranges)

        """
        f(range_vec) returns the velocity difference between the second and first velocity estimates
        for the central observation. The velocity difference is calculated by first calculating the
        positions of the observations using the range vector and the observer position. The positions
        are then used to calculate the velocities between each observation via the Izzo solution to
        Lambert's problem. The velocity difference is then calculated as the difference between the
        second and first velocity estimates. This function is used in the Newton method to find the
        root of the velocity difference.
        """
        range_mag =  x + p                          #range_mag = Nominal + Perturbation
        
        range_vec = range_mag * i_rho
                         #define posiiton vector for lamber_izzo
        r_vec = range_vec + r_obs_heliocentric

    
        #calculate the velocities via lamerts problem 
        velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, prograde=True)
        v2_minus = velocities[0][:,1] #unpack solutions 

        velocities = lambert_izzo(r_vec[:,1], r_vec[:,2], t_obs_s[2] - t_obs_s[1], mu, 0, prograde=True)
        # Centre posiitons should have zero velocity difference
        v2_plus = velocities[0][:,0] #unpack solutions

        DV = (v2_minus - v2_plus)

        return DV                                      #DV = [dv_i, dv_j, dv_k] + M(dranges)

    assert(isinstance(range_mag_guass, np.ndarray)), "Guass Range must be in Floats"
    
    p0 = np.zeros_like(range_mag_guass) 
    assert order >= 4, "Order must be 4 for the Householder Iteration to work" 

    DA.init(order, 3)                                      
    range_mag_L1 = newton_nominal_DAVec(range_mag_guass, p0, f, order, tol, 100)      #Obtain Range so that f(Range(0)) = 0
    
    DA.init(order, 6)
    p = array([p0[i] + DA(i+1) for i in range(len(range_mag_guass))])          #Define Range polynmoinal map

    range_mag_L1 = Implicit_solver_DAVec(range_mag_L1, p, f, 3)

    return range_mag_L1        #DA output

def DAIOD_1_debugSciPy(range_mag_guass: float, i_rho: np.ndarray, t_obs_s: np.ndarray, order, r_obs_heliocentric: np.ndarray = np.zeros((3, 3)), mu: float=1.32712e11, tol: float=1e-9):

    def f(x, p):                                      #deltV = residual + M(dranges)

        """
        f(range_vec) returns the velocity difference between the second and first velocity estimates
        for the central observation. The velocity difference is calculated by first calculating the
        positions of the observations using the range vector and the observer position. The positions
        are then used to calculate the velocities between each observation via the Izzo solution to
        Lambert's problem. The velocity difference is then calculated as the difference between the
        second and first velocity estimates. This function is used in the Newton method to find the
        root of the velocity difference.
        """
        range_mag =  x + p                          #range_mag = Nominal + Perturbation
            
        range_vec = range_mag * i_rho
                            #define posiiton vector for lamber_izzo
        r_vec = range_vec + r_obs_heliocentric

        
        #calculate the velocities via lamerts problem 
        velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, prograde=True)
        v2_minus = velocities[0][:,1] #unpack solutions 

        velocities = lambert_izzo(r_vec[:,1], r_vec[:,2], t_obs_s[2] - t_obs_s[1], mu, 0, prograde=True)
        # Centre posiitons should have zero velocity difference
        v2_plus = velocities[0][:,0] #unpack solutions

        DV = (v2_minus - v2_plus)

        return DV                           #DV = [dv_i, dv_j, dv_k] + M(dranges)
    
    assert(isinstance(range_mag_guass, np.ndarray)), "Guass Range must be in Floats"
    
    # Create a wrapper function for scipy.optimize.fsolve (takes only x as input)
    def f_scipy(x):
        """Wrapper function for scipy that takes only x as input"""
        p_zero = np.zeros_like(x)
        result = f(x, p_zero)
        return result.cons() if hasattr(result, 'cons') else result
    
    # Use scipy.optimize.fsolve for robust root finding
    from scipy.optimize import fsolve
    
    print(f"Initial guess for fsolve: {range_mag_guass}")

    range_mag_L1_nom = fsolve(f_scipy, range_mag_guass, xtol=tol)
    print(f"fsolve converged to: {range_mag_L1_nom}")

    # Convert to DA array to maintain compatibility with rest of code
    DA.init(order, 6)
    p0 = np.zeros_like(range_mag_guass)
        # Create DA array from the solved nominal values

    p0 = np.zeros_like(range_mag_guass)  # Define p0 for the DA part
    p = array([p0[i] + DA(i+1) for i in range(len(range_mag_guass))])          #Define Range polynmoinal map

    #Generate Numerical Jacobian
    from scipy.optimize import approx_fprime
    
    def f_component(range_mag, component_idx):
        """Return specific component of velocity difference for finite difference"""
        p_zero = np.zeros_like(range_mag)
        dv = f(range_mag, p_zero)
        dv_result = dv.cons() if hasattr(dv, 'cons') else dv
        return dv_result[component_idx]
    
    # Compute finite difference Jacobian
    n_vars = len(range_mag_guass)  # Number of range variables
    fd_jacobian = np.zeros((3, n_vars))  # 3 velocity components, n_vars range components
    h = 1e-8  # Step size for finite differences
    
    for i in range(3):  # velocity components [dvx, dvy, dvz]
        def func_i(range_mag):
            return f_component(range_mag, i)
        fd_jacobian[i, :] = approx_fprime(range_mag_L1_nom, func_i, h)
    
    print(f"\n=== FINITE DIFFERENCE JACOBIAN ===")
    print("∂[dvx, dvy, dvz]/∂[range1, range2, range3] =")
    print(fd_jacobian)
    
    return range_mag_L1_nom, fd_jacobian        #DA output


import numpy as np

def DV_from_ranges_float(x, i_rho, t_obs_s, r_obs_heliocentric, mu, prograde_bool):
    """
    Pure-float DV(x): 3-vector. Uses the same model as in DAIOD_1Scipy_invert.f
    x: (3,) float array of range magnitudes [km]
    """
    # Build positions
    range_vec = (x.reshape(3, 1) * i_rho)              # (3x3)
    r_vec = range_vec + r_obs_heliocentric             # (3x3)

    # Lambert solves
    v12m = lambert_izzo(r_vec[:, 0], r_vec[:, 1], t_obs_s[1] - t_obs_s[0], mu, 0, prograde=prograde_bool)[0]
    v12p = lambert_izzo(r_vec[:, 1], r_vec[:, 2], t_obs_s[2] - t_obs_s[1], mu, 0, prograde=prograde_bool)[0]

    DV = v12m[:, 1] - v12p[:, 0]                       # (3,)
    # Ensure DV is a flat float array
    return np.asarray(DV, dtype=float).reshape(3,)

def finite_diff_Jx_DAIOD1(x_nom,
                          i_rho,
                          t_obs_s,
                          r_obs_heliocentric,
                          mu,
                          prograde_bool=False,
                          rel_step=1e-6,
                          abs_step_min=1e-3,
                          scheme="central",
                          step_sweep=None):
    """
    Finite-difference Jacobian Jx = d(DV)/d(x) at x_nom.
    x_nom: (3,) floats [km]
    i_rho: (3x3) LOS unit vectors per epoch
    t_obs_s: (3,) times [s]
    r_obs_heliocentric: (3x3) observer position per epoch [km]
    returns: (3x3) Jacobian matrix
    """
    x_nom = np.asarray(x_nom, dtype=float).reshape(3,)
    # Base evaluation
    DV0 = DV_from_ranges_float(x_nom, i_rho, t_obs_s, r_obs_heliocentric, mu, prograde_bool)

    # Choose step sizes per component
    h = np.maximum(rel_step * np.abs(x_nom), abs_step_min)

    def column_by_fd(j, h_j):
        e = np.zeros_like(x_nom); e[j] = 1.0
        if scheme == "central":
            xp = x_nom + h_j * e
            xm = x_nom - h_j * e
            DVp = DV_from_ranges_float(xp, i_rho, t_obs_s, r_obs_heliocentric, mu, prograde_bool)
            DVm = DV_from_ranges_float(xm, i_rho, t_obs_s, r_obs_heliocentric, mu, prograde_bool)
            return (DVp - DVm) / (2.0 * h_j)
        elif scheme == "forward":
            xp = x_nom + h_j * e
            DVp = DV_from_ranges_float(xp, i_rho, t_obs_s, r_obs_heliocentric, mu, prograde_bool)
            return (DVp - DV0) / h_j
        else:
            raise ValueError("scheme must be 'central' or 'forward'")

    # Single-step Jacobian
    Jx = np.column_stack([column_by_fd(j, h[j]) for j in range(3)])

    # Optional step sweep to check stability (returns dict of {step: J})
    sweep = None
    if step_sweep is not None:
        sweep = {}
        for s in step_sweep:
            h_s = np.maximum(s * np.abs(x_nom), abs_step_min)
            J_s = np.column_stack([column_by_fd(j, h_s[j]) for j in range(3)])
            sweep[s] = J_s

    return Jx, DV0, sweep

def compare_fd_vs_DA(J_fd, J_da, label_fd="FD", label_da="DA"):
    diff = J_fd - J_da
    def frob(A): return np.sqrt(np.sum(A*A))
    print(f"{label_fd} Frobenius: {frob(J_fd):.6e}   {label_da} Frobenius: {frob(J_da):.6e}")
    print(f"‖{label_fd}-{label_da}‖_F : {frob(diff):.6e}")
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(np.abs(J_fd) > 0, np.abs(diff)/np.abs(J_fd), 0.0)
    print(f"Max abs diff : {np.max(np.abs(diff)):.6e}")
    print(f"Max rel diff : {np.max(rel):.6e}")
    return diff


def DAIOD_1Scipy_invert(range_mag_guass: float, i_rho: np.ndarray, t_obs_s: np.ndarray, order, r_obs_heliocentric, mu: float, tol: float=1e-9, prograde_bool: bool=False):
    """
        Part 1 of the DAIOD Algorithm: Define the ranges such that DV(range_mag(p)) = 0

        :param range_mag_guass: Range magnitude. See Guass_8th_seed() [km]
        :param i_rho: Line of sight unit vector. See create_da_los_vectors() [-]
        :param t_obs_s: The time of the observation. [s]
        :param r_obs_heliocentric: Position of the observer. Assumed heliocentric plane, centre of Earth.  [km]
        :param mu: The standard gravitational parameter. Assumed Sun Orbiting [km^3/s^2]
        :param tol: The tolerance for the Newton method. [km]

        :return range_mag_L1: Range Taylor Polynomial Map. range_mag_L1 + variations  [km]
    """

    def f(x, p):                                      #deltV = residual + M(dranges)

        """
        f(range_vec) returns the velocity difference between the second and first velocity estimates
        for the central observation. The velocity difference is calculated by first calculating the
        positions of the observations using the range vector and the observer position. The positions
        are then used to calculate the velocities between each observation via the Izzo solution to
        Lambert's problem. The velocity difference is then calculated as the difference between the
        second and first velocity estimates. This function is used in the Newton method to find the
        root of the velocity difference.
        """
        range_mag =  x + p                          #range_mag = Nominal + Perturbation
        
        range_vec = range_mag * i_rho
                         #define posiiton vector for lamber_izzo
        r_vec = range_vec + r_obs_heliocentric

    
        #calculate the velocities via lamerts problem 
        velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, prograde=prograde_bool)
        v2_minus = velocities[0][:,1] #unpack solutions 

        velocities = lambert_izzo(r_vec[:,1], r_vec[:,2], t_obs_s[2] - t_obs_s[1], mu, 0, prograde=prograde_bool)
        # Centre posiitons should have zero velocity difference
        v2_plus = velocities[0][:,0] #unpack solutions

        DV = v2_minus - v2_plus

        return DV                           #DV = [dv_i, dv_j, dv_k] + M(dranges)
    
    assert(isinstance(range_mag_guass, np.ndarray)), "Guass Range must be in Floats"
    
    # Create a wrapper function for scipy.optimize.fsolve (takes only x as input)
    def f_scipy(x):
        """Wrapper function for scipy that takes only x as input"""
        p_zero = np.zeros_like(x)
        result = f(x, p_zero)
        return result.cons() if hasattr(result, 'cons') else result
    
    # Use scipy.optimize.fsolve for robust root finding
    from scipy.optimize import fsolve, root

    print(f"Initial guess for fsolve: {range_mag_guass}")
 
    range_mag_L1_nom = fsolve(f_scipy, range_mag_guass, xtol=tol)
    if np.any(range_mag_L1_nom < 0) < 0:
        raise ValueError("fsolve converged to a negative range value, which is non-physical. Change Prograde Orbit condition")
    
    if order is None:
        return range_mag_L1_nom
    # Convert to DA array to maintain compatibility with rest of code

    
    DA.init(order, 6)

    i_rho = array(i_rho)
    r_obs_heliocentric = array(r_obs_heliocentric)


    p0 = np.zeros_like(range_mag_guass)
    x0 = range_mag_L1_nom * np.ones(3)
    dx = array.identity(3)
    dp = array([DA(i+4) for i in range(3)])
    x = x0 + dx
    p = p0 + dp

    #Complete Inversion Described by Pirovano
    
    F = f(x, p)          #Generate Map

    F_aug = F.concat(p)
    Finv = F_aug.invert()                   #Inversion of Map 
    
    rhs = array.zeros(6)
    rhs[3:] = p
    sol = Finv.eval(rhs) # x in terms of f and p coefficents
    
    dx_polys = sol[:3]  #Extract dx
    if np.any(dx_polys.cons(), axis=0):
        dx_polys -= dx_polys.cons()  #Ensure zero constant term

    x_DA = x0 + dx_polys  #Update x with dx

    F_DA = f(x_DA, p)      #F = 0 at the solution point

    Jac_full = F_DA.linear()   # This is too small. f(x_DA, p) = 0

    mask = np.any(np.abs(Jac_full) > 0, axis=0)  # Mask for non-zero columns
    idx = np.where(mask)[0]  # Get indices of non-zero columns
    Jac_nonzero = Jac_full[:, idx]  # Extract non-zero columns

    # Check with Finite Differences
    J_fd, DV0, sweep = finite_diff_Jx_DAIOD1(range_mag_L1_nom,
                                             i_rho.cons(),
                                             t_obs_s,
                                             r_obs_heliocentric.cons(),
                                             mu,
                                             prograde_bool=prograde_bool,
                                             rel_step=1e-6,
                                             abs_step_min=1e-3,
                                             scheme="central",
                                             step_sweep=[1e-5, 1e-6, 1e-7])
    
    diff = compare_fd_vs_DA(J_fd, Jac_nonzero)
    print(diff)
    return x_DA, F.linear()[:,:3]       #DA output

def DAIOD_1Scipy(range_mag_guass: float, i_rho: np.ndarray, t_obs_s: np.ndarray, order, r_obs_heliocentric: np.ndarray = np.zeros((3, 3)), mu: float=1.32712e11, tol: float=1e-9):
    """
        Part 1 of the DAIOD Algorithm: Define the ranges such that DV(range_mag(p)) = 0

        :param range_mag_guass: Range magnitude. See Guass_8th_seed() [km]
        :param i_rho: Line of sight unit vector. See create_da_los_vectors() [-]
        :param t_obs_s: The time of the observation. [s]
        :param r_obs_heliocentric: Position of the observer. Assumed heliocentric plane, centre of Earth.  [km]
        :param mu: The standard gravitational parameter. Assumed Sun Orbiting [km^3/s^2]
        :param tol: The tolerance for the Newton method. [km]

        :return range_mag_L1: Range Taylor Polynomial Map. range_mag_L1 + variations  [km]
    """

    def f(x, p):                                      #deltV = residual + M(dranges)

        """
        f(range_vec) returns the velocity difference between the second and first velocity estimates
        for the central observation. The velocity difference is calculated by first calculating the
        positions of the observations using the range vector and the observer position. The positions
        are then used to calculate the velocities between each observation via the Izzo solution to
        Lambert's problem. The velocity difference is then calculated as the difference between the
        second and first velocity estimates. This function is used in the Newton method to find the
        root of the velocity difference.
        """
        range_mag =  x + p                          #range_mag = Nominal + Perturbation
        
        range_vec = range_mag * i_rho
                         #define posiiton vector for lamber_izzo
        r_vec = range_vec + r_obs_heliocentric

    
        #calculate the velocities via lamerts problem 
        velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, prograde=True)
        v2_minus = velocities[0][:,1] #unpack solutions 

        velocities = lambert_izzo(r_vec[:,1], r_vec[:,2], t_obs_s[2] - t_obs_s[1], mu, 0, prograde=True)
        # Centre posiitons should have zero velocity difference
        v2_plus = velocities[0][:,0] #unpack solutions

        DV = (v2_minus - v2_plus)

        return DV                           #DV = [dv_i, dv_j, dv_k] + M(dranges)
    
    assert(isinstance(range_mag_guass, np.ndarray)), "Guass Range must be in Floats"
    
    # Create a wrapper function for scipy.optimize.fsolve (takes only x as input)
    def f_scipy(x):
        """Wrapper function for scipy that takes only x as input"""
        p_zero = np.zeros_like(x)
        result = f(x, p_zero)
        return result.cons() if hasattr(result, 'cons') else result
    
    # Use scipy.optimize.fsolve for robust root finding
    from scipy.optimize import fsolve
    
    print(f"Initial guess for fsolve: {range_mag_guass}")

    range_mag_L1_nom = fsolve(f_scipy, range_mag_guass, xtol=tol)
    print(f"fsolve converged to: {range_mag_L1_nom}")

    # Convert to DA array to maintain compatibility with rest of code
    DA.init(order, 3)
    p0 = np.zeros_like(range_mag_guass)
        # Create DA array from the solved nominal values

    p0 = np.zeros_like(range_mag_guass)  # Define p0 for the DA part
    p = array([p0[i] + DA(i+1) for i in range(len(range_mag_guass))])          #Define Range polynmoinal map

    range_mag_L1, J_L1 = Implicit_solver_DAVec(range_mag_L1_nom, p, f, 3, x0DA=False)

    assert np.allclose(range_mag_L1.cons(), range_mag_L1_nom, rtol=1e-6), "DAIOD_1 failed to converge to x where f(x) = 0"
    return range_mag_L1, J_L1         #DA output


def DAIOD_2(range_mag: array, RA, DEC, t_obs_s: array, order, r_obs_heliocentric: np.ndarray = np.zeros((3, 3)), mu: float=1.32712e11, J=0, prograde_bool: bool=False):
    """
        Part 2 of DAIOD Algorithm: Define the range variation in terms of anglular variations

    """
    def f1(x, p):
        """
        Delta V Function for Angle Variables
        :params x: Range Magnitude
        :params p: RA and DEC array
        """
        RA = p[:3]
        DEC = p[3:]
        
        i_rho = array(time_ref.create_da_los_vectors(RA, DEC))

        range_mag = x

        range_vec = range_mag * i_rho

        r_vec = range_vec + r_obs_heliocentric

        
        #calculate the velocities via lamerts problem 
        velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, prograde=prograde_bool)
        v2_minus = velocities[0][:,1] #unpack solutions 

        velocities = lambert_izzo(r_vec[:,1], r_vec[:,2], t_obs_s[2] - t_obs_s[1], mu, 0, prograde=prograde_bool)
        # Centre posiitons should have zero velocity difference
        v2_plus = velocities[0][:,0] #unpack solutions

        DV = (v2_minus - v2_plus)
        return DV

    # Getting into Priovano Notation
    p = RA.concat(DEC)                  # Concatenate RA and DEC into a single array    
    x0 = range_mag.cons()
    J_dv_x = J 

    if J_dv_x is None or J_dv_x.shape != (3,3):
        raise ValueError("Jacobian 3x3 J_dv must be provided for DAIOD_2")

    Jinv_DA = array(np.linalg.inv(J_dv_x))
    iterMax = DA.getMaxOrder()
    
    F0 = f1(x0, p)         #Evaluate the function to get the residuals
    J_dv_p = F0.linear()   # Extract the Jacobian w.r.t. RA and DEC

    M = - Jinv_DA @ J_dv_p                      # Sensitivity Matrix (new)
    tmp = array([DA(1+i) for i in range(6)])  # Create DA array for RA and DEC variations (new)
    rho_map = x0 + M @ tmp  # Range Map (new)

    i = 1
    for _ in range(iterMax):
        if i <= iterMax:
            F = f1(rho_map, p)         #Evaluate the function to get the residuals, Old (working version) is to use x0 as rho_map
            dF = F - F.cons()
            dx = - (Jinv_DA @ dF)
            if np.any(dx.cons(), axis=0):
                dx -= dx.cons()  # Ensure zero constant term
            
            rho_map = rho_map + dx  #Newton Step
            i *= 2
            # x0 = x1
        else:
            break

    return rho_map


def f_g_series(r0, dt, f_order, g_order, mu=1.32712440018e11, v0=None):
    """
    Generate the f and g series for the Kepler problem to a maximum order of 4.
    :param r0: Initial position vector (1D column vector)
    :param dt: Time of flight in seconds
    :param f_order: Order of the f series expansion
    :param g_order: Order of the g series expansion
    :param v0: Initial velocity vector (optional, if not provided, only second order series is generated)
    :param mu: Standard gravitational parameter (default is the Sun's)
    :return: f_series, g_series - numpy arrays of the f and g series coefficients
    """
    assert r0.ndim == 1 and r0.shape[0] == 3, "r0 must be a 1D column vector with 3 components"
    assert v0 is None or (v0.ndim == 1 and v0.shape[0] == 3), "v0 must be a 1D column vector with 3 components or None"
    assert isinstance(f_order, int) and f_order >= 0, "f_order must be a non-negative integer"
    assert isinstance(g_order, int) and g_order >= 0, "g_order must be a non-negative integer"


    # Calculate the norm of the initial position vector
    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0) if v0 is not None else 0.0
    assert r0_norm > 0, "Initial position vector r0 must not be zero"
    
    # Precompute the An, Bn and M - do later for general case
    if v0 is None:
        v0 = np.zeros(3).T  # If no initial velocity is provided, assume zero velocity
    
    # expand series for f and g
    f_series = np.array([1.0,                                           #0th order
                        0.0,                                           #1st order
                        -(mu/(2*r0_norm**3))*dt**2,                     #2nd order
                        (mu/2*(np.dot(r0,v0))/r0_norm**5) * dt**3,     #3rd order                                      #3rd order
                        (mu/24 * (-2*mu/r0_norm**6) + 3*v0_norm**2/r0_norm**6 - 15*(np.dot(r0,v0))/r0_norm**7) * dt**4])  #4th order                                         #4th order

    g_series = np.array([0.0,                                           #0th order
                         dt,                                            #1st order 
                         0.0,                                           #2nd order
                         -1/6 * (mu/r0_norm**3) * dt**3,                 #3rd order
                         mu/4 * (np.dot(r0,v0)/ r0_norm**5) * dt**4])   #4th order

    # Ensure the series are of the correct order
    if f_order < len(f_series):
        f_series = f_series[:f_order+1]

    f_series = f_series.sum()


    if g_order < len(g_series):
        g_series = g_series[:g_order+1]
    
    g_series = g_series.sum()

    return f_series, g_series

def position_feasibility(r1: NDArray[np.double], r2: NDArray[np.double], r3: NDArray[np.double], v2: NDArray[np.double], mu, Re: float = 6378.137) -> bool:
    """
    Check if the Guass IOD positions are physically feasible.

    Paramters:
    r: column vectors of positions [x, y, z].T in kilometers
        Position vectors for the three observations. rows correspond to the r_2 roots, columns are the positions values.
    Re : float
        Earth radius in kilometers

    Returns:
    bool
        true if position are feasible, False otherwise
    Rasies:
    ValueError
        If positions fail feasibility check.
    """
    # Pre-condiition checks to ensure inputs are valid
    assert r1.shape == (3,), "r1 must be a 1D column vector"
    assert r2.shape == (3,), "r2 must be a 1D column vector"
    assert isinstance(Re, (int, float)), "Re must be a number"


    d_earth_sun = 1.496e+8  # Earth-Sun distance in kilometers, used for feasibility checks ( 1AU)
    # Test 1a: norm r < Earth Sun distance
    if mu == 1.32712440018e11:
        if np.linalg.norm(r1) <= d_earth_sun or np.linalg.norm(r2) <= d_earth_sun or np.linalg.norm(r3) <= d_earth_sun:
            r1[:] = np.nan
            r2[:] = np.nan
            r3[:] = np.nan
            v2[:] = np.nan
            raise ValueError("Position vectors r1 and r2 must be greater than distance of Earth as night time viewing).")
    elif mu == 3.986e5:
        if np.linalg.norm(r1) <= Re or np.linalg.norm(r2) <= Re or np.linalg.norm(r3) <= Re:
            r1[:] = np.nan
            r2[:] = np.nan
            r3[:] = np.nan
            v2[:] = np.nan
            raise ValueError("Position vectors r1 and r2 must be greater than Radius of earth).")
    
    # Test 3: Specific energy is negative - asteroid elliptic orbit
    specific_energy = (np.linalg.norm(v2)**2 / 2) - mu / (np.linalg.norm(r2))
    
    if specific_energy >= 0:
        r1[:] = np.nan
        r2[:] = np.nan
        r3[:] = np.nan
        v2[:] = np.nan
        print(f"Specific energy is not negative (non-elliptic): {specific_energy}")


    return r1, r2, r3, v2 # Return the positions if all checks pass

def Guass_8th_seed(pos_obs: NDArray, obs_dir: NDArray, t: NDArray, mu=1.32712440018e11) -> NDArray[np.double]:
    """
    Taken from Orbital Mechanics for Engineering Students (4th ed.) by Curtis. p.242 Algorithm 5.5. and Armillien Aphopis
    The Seed takes real numbers not DA numbers.

    :param Pos_obs: 2D array of Observer positions in Heliocentric frame of reference from angle rotation matrix
               Rows are compoents, columns are observations instances
    :param t: 1D array of times in seconds
    :param obs_dir: 2D array of unit vectors pointing from observer to the point of interest, every column is an observation instance the rows are components x y z
    :param 2-BP assumption: Assume the observations lie on the same plane

    :returns:
    position, range, range_mag,v_2
   """
    dt_1 = t[0] - t[1]
    dt_3 = t[2] - t[1]
    dt = dt_3 - dt_1

    #Re-write the cross and dot products in numpy structure

    p1 = np.cross(obs_dir[:,1], obs_dir[:,2])  #cross product of the second and third observation direction vectors
    p2 = np.cross(obs_dir[:,0], obs_dir[:,2])  #cross product of the first and third observation direction vectors
    p3 = np.cross(obs_dir[:,0], obs_dir[:,1])  #cross product of the first and second observation direction vectors

    D0 = np.dot(obs_dir[:,0], p1)  #Dot product of the first observation direction vector and the cross product of the second and third observation direction vectors

    D11 = np.dot(pos_obs[:,0], p1)
    D12 = np.dot(pos_obs[:,0], p2)
    D13 = np.dot(pos_obs[:,0], p3)
    D21 = np.dot(pos_obs[:,1], p1)
    D22 = np.dot(pos_obs[:,1], p2)
    D23 = np.dot(pos_obs[:,1], p3)
    D31 = np.dot(pos_obs[:,2], p1)
    D32 = np.dot(pos_obs[:,2], p2)
    D33 = np.dot(pos_obs[:,2], p3)

    A = 1 / D0 * ((-D12 * dt_3/ dt) + D22 + (D32 * dt_1/dt))
    B = 1 / (6*D0) * (D12*(dt_3**2 - dt**2)*dt_3/dt + (D32*(dt**2 - dt_1**2) * dt_1/dt))

    E = np.dot(pos_obs[:,1], obs_dir[:,1])
    R_2_squared = np.dot(pos_obs[:,1], pos_obs[:,1])

    #obtain coefficient values for the 8th degree position polynomial
    a =  -1*(A**2 + 2*A*E + (R_2_squared))
    b = -2*mu*B*(A+E)
    c = -1 * (mu**2 * B**2)

    #set up coefficients for the 8th degree position polynomial for evaluation
    coeffs = [1.0,          #r^8
              0.0,          #r^7
              a,            #r^6
              0.0,          #r^5
              0.0,          #r^4
              b,            #r^3
              0.0,          #r^2
              0.0,          #r^1
              c             #r^0
              ]

    #Check if 8th degree polynomial coefficients are floats and can be evaluated using numpy
    # Assert all coefficients are floats
    for i, coeff in enumerate(coeffs):
        assert isinstance(coeff, float), f"Coefficient at position {i} is not a float: {type(coeff)}"
        assert not np.isnan(coeff), f"Coefficient at position {i} is NaN"
        assert not np.isinf(coeff), f"Coefficient at position {i} is infinite"

    roots = np.roots(coeffs)                                                    #obtain the roots of the 8th polynomial - np.roots using eignevalue evaluation of coeffs
    real_roots = roots[np.logical_and(np.imag(roots) == 0, np.real(roots) > 0)] #Need physical roots
    assert len(real_roots) > 0, "No positive real roots found"                  #If no real roots print

    #Inform user of real roots values, and conduct pruning if more than 1 real root
    r_2_mag = real_roots.real                                                #Store possible solutions of r_2
    if len(real_roots.real) > 0:
        print("There are more than 1 positive real roots.")
        range_1_mag = np.zeros((len(real_roots)))   # columns are different root value, rows are [x,y,z] components            #topocentric ranges
        range_2_mag = np.zeros((len(real_roots)))               #topocentric ranges
        range_3_mag = np.zeros((len(real_roots)))               #topocentric ranges

        r_1 = np.zeros((3,len(real_roots)))
        r_2 = np.zeros((3,len(real_roots)))                   #helocentric positions
        r_3 = np.zeros((3,len(real_roots)))
    
        print(f"There are {len(real_roots)} positive real roots:\n")

        for i in range(len(real_roots)):
            print(f"Root {i}: {r_2_mag[i]}\n")
            # build range1, range2, range3 and conduct feasibility tests.
            # Ranges are defined through truncated Lagrange coefficients so they're not accurate and refinement required
            
            # Calculate the range_1 and r1
            num = (6*(D31 * (dt_1/dt_3) + D21 * (dt/dt_3))*(r_2_mag[i]**3)) + (mu * D31 * (dt**2 - dt_1**2)* (dt_1/dt_3))
            den = (6 * r_2_mag[i]**3) + (mu * (dt**2 - dt_3**2))
            range_1_mag[i] = 1 / D0 * ( (num / den) - D11)          #slant range magnitude for the first observation

            # Calculate the range_3 and r3
            num = (6*(D13 * (dt_3/dt_1) - D23 * (dt/dt_1))*(r_2_mag[i]**3)) + (mu * D13 * (dt**2 - dt_3**2)* (dt_3/dt_1))
            den = (6 * r_2_mag[i]**3) + (mu * (dt**2 - dt_3**2))
            range_3_mag[i] = 1 / D0 * ((num / den) - D33)           #slant range magnitude for the third observation

            # Calculate the range_2 and r2
            range_2_mag[i] = A + (mu * B) / (r_2_mag[i] ** 3)
    
            if len(real_roots) >= 1:
                if r_2_mag[i] > 164359881.2857059:          #Apophis Apoapsis
                    print(f"Range 2 is too large for root {i}, skipping feasibility check.\n")
                    range_1_mag[i] = np.nan
                    range_2_mag[i] = np.nan
                    range_3_mag[i] = np.nan

                    r_2_mag[i] = np.nan
                    r_1[:,i] = np.full((3,), np.nan)
                    r_2[:,i] = np.full((3,), np.nan)
                    r_3[:,i] = np.full((3,), np.nan)
                    range_1 = np.full((3, ), np.nan)  # If no real roots, set the range to NaN
                    range_2 = np.full((3, ), np.nan)  # If no real roots, set the range to NaN
                    range_3 = np.full((3, ), np.nan)
                    v_2 = np.full((3,), np.nan)
                    continue

                elif range_2_mag[i] < 0:
                    print(f"Range 2 is negative for root {i}, skipping feasibility check.\n")
                    range_1_mag[i] = np.nan
                    range_2_mag[i] = np.nan
                    range_3_mag[i] = np.nan
                    range_1 = np.full((3, ), np.nan)  # If no real roots, set the range to NaN
                    range_2 = np.full((3, ), np.nan)  # If no real roots, set the range to NaN
                    range_3 = np.full((3, ), np.nan)
                    
                    r_2_mag[i] = np.nan
                    r_1[:,i] = np.full((3,), np.nan)
                    r_2[:,i] = np.full((3,), np.nan)
                    r_3[:,i] = np.full((3,), np.nan)
                    v_2 = np.full((3,), np.nan)
                    continue
                
            # calculate r1,r2, r3
            r_1[:,i] = pos_obs[:,0] + range_1_mag[i]*obs_dir[:,0]
            r_2[:,i] = pos_obs[:,1] + range_2_mag[i]*obs_dir[:,1]
            r_3[:,i] = pos_obs[:,2] + range_3_mag[i]*obs_dir[:,2]

            #obtain v2 velcoity vector at r2 for feasibility check
            
            f_3, g_3 = f_g_series(r_2[:,i], dt_3, 2, 3, mu=mu)  #get f and g series for the third observation
            f_1, g_1 = f_g_series(r_2[:,i], dt_1, 2, 3, mu=mu)  #get f and g series for the first observation
            
            
            
            v_2 = 1/((f_1*g_3) - (f_3*g_1)) * (-f_3*r_1[:,i] + f_1*r_3[:,i]) 


            r_1[:,i], r_2[:,i], r_3[:,i], v_2 = position_feasibility(r_1[:,i], r_2[:,i], r_3[:,i], v_2, mu)

            if not (np.isnan(r_1[:,i]).any() or np.isnan(r_2[:,i]).any() or np.isnan(r_3[:,i]).any() or np.isnan(v_2).any()):
                print(f"Feasibility passed for root {i}:")
                print(f"  r_1: {r_1[:,i]}")
                print(f"  r_2: {r_2[:,i]}")
                print(f"  r_3: {r_3[:,i]}")

                range_1 = range_1_mag[i] * obs_dir[:,0]  #convert range to vector
                range_2 = range_2_mag[i] * obs_dir[:,1]  #convert range to vector
                range_3 = range_3_mag[i] * obs_dir[:,2]  #convert range to vector
                
                idx = i
                break   #exit loop on first feasible root
            else:
                print(f"Feasibility failed for root {i}:")
                print(f"  r_1: {r_1[:,i]}")
                print(f"  r_2: {r_2[:,i]}")
                print(f"  r_3: {r_3[:,i]}")

                range_1 = np.full((3, 1), np.nan)
                range_2 = np.full((3, 1), np.nan)
                range_3 = np.full((3, 1), np.nan)
                
                range_1_mag = np.nan
                range_2_mag = np.nan
                range_3_mag = np.nan

    else:
        print(f"There is no positive real root: {r_2}\n")
        r_1 = np.full((3, 1), np.nan)  # If no real roots, set the position to NaN
        r_2 = np.full((3, 1), np.nan)  # If no real roots, set the position to NaN
        r_3 = np.full((3, 1), np.nan)  # If no real roots, set the position to NaN
        range_1 = np.full((3, 1), np.nan)  # If no real roots, set the range to NaN
        range_2 = np.full((3, 1), np.nan)  # If no real roots, set the range to NaN
        range_3 = np.full((3, 1), np.nan)  # If no real roots, set the range to NaN
        v_2 = np.full((3,), np.nan)


    #If no real roots, set the position and range to zero
    #obtain the [r1,r2,r3] [range1,range2,range3]. Rows are the real root, columns are positions
    position = np.column_stack([r_1[:,i], r_2[:,i], r_3[:,i]])
    ranges = np.array([range_1, range_2, range_3]).T
    range_mag = np.array([range_1_mag[i], range_2_mag[i], range_3_mag[i]])

    return position, ranges, range_mag, v_2


#
def lambert_battin_DA(r1: Union[array,NDArray], r2: Union[array, NDArray], dt: float, max_iter: int,  mu: float = 3.2712440018e11, order: int = None):
    """
    :parameter:
    :param r1: Float or DA array (x,y,z compoents) of initial poisition
    :param r2: Float or DA array (x,y,z compoents) of final poisition
    :param dt : time of flight
    :param mu : standard gravitional parameter - default is sun
    :param max_iter: maximum number of iterations for Newton iteration
    :param order : Maximum algebraic order of DA variables - default global order

    Returns:
    :param v1 : Float or DA array (x,y,z compoents) of velocity departure
    :param v2 : Float or DA array (x,y,z compoents) of velocity destination
    """
    # -----------------------------------------
    # 1) promote to DA if neccessary & set a sensible algebra
    # -----------------------------------------
    if not isinstance(r1[0], DA) and not isinstance(r2[0], DA):
        print(f"Real")        #nothing happens carry r1/r2 DA settings over - real numbers
    else:
        if order is None:
            #inherit global settings
            order = DA.getMaxOrder(r1[0] if isinstance(r1[0], DA) else r2[0])
        else:
            # make sure current algebra matches requested order
            nvar = DA.getMaxVariables()
            DA.init(order,nvar + 1)     # 3 variables for r1, 3 variables for r2, 1 variable for x

    # --------------------------
    # 2) Geometry calculations - DA aware
    # ---------------------------
    c_vec = r2 - r1
    c = op.vnorm(c_vec)
    r1_mag = op.vnorm(r1)
    r2_mag = op.vnorm(r2)
    s = 0.5 * (r1_mag + r2_mag + c)
    cosTA = ((r1.dot(r2)) / (r1_mag * r2_mag))
    TA = op.acos(cosTA)

    # -----------------------------
    # 3) nomial solution for x - scalar Newton
    # -----------------------------
    def battin_F(x):  # scalar (or DA) residual
        g = s / (2 * (1 - x ** 2))
        alpha = 2 * op.asin(op.sqrt(s / (2 * g)))
        beta = 2 * op.asin(op.sqrt((s - c) / (2 * g)))
        A = g ** 1.5 * (alpha - op.sin(alpha) - beta + op.sin(beta))
        return op.log10(A) - op.log10(dt)  # f(x) = 0

    def x_initial():
        dt_p = np.sqrt(2/mu)*(s.cons()**(3/2) - c.cons()**(3/2))
        T = dt/dt_p
        x0 = (1-T) / (1+T)
        return x0
    
    k = r1[0].getMaxVariables()
    
    x = 1e-3 + DA(k)                    #guess at 0.1

    def fbattin(x, p):
        #unpack p
        r1 = p[0,:]
        r2 = p[1,:]
        

        A = battin_A_DA(x, r1, r2)      #x is too large again - need to review battin lambert paper for first guess of x
        return op.log10(A) - np.log10(dt)
    
    x0 = 0.0
    p = array([r1, r2])
    
    x_nom = newton_nomial_DA(x0, p, fbattin, tol=1e-9, MaxIter=1000)
    
    # -------------------------------
    # 4) Get x_map through newton of battin scalar equation
    # ---------------------------------
    #Get battin's x in terms of DA variables from position perturbations
        #Newton iteartor - can replace with .invert() but need to see iteration number
    if isinstance(r1[0], DA) and isinstance(r2[0], DA):
        x = battin_x_DA(r1, r2, dt, order, x)           #generate DA map if DA exists
    
    #-------------------------------------------
    # 5) Prepare for lagrange coefficients/ velocity calculations
    #-------------------------------------------
    a = s / (2 *  (1 - x**2))
    g = s / (2 * (1 - x ** 2))
    alpha = 2 * op.asin(op.sqrt(s / (2 * g)))
    beta = 2 * op.asin(op.sqrt((s - c) / (2 * g)))
    
    if TA > np.pi:
        beta = -beta

    a_min = s / 2
    t_min = np.sqrt(a_min ** 3 / mu) * (np.pi - beta + op.sin(beta))

    if dt > t_min:
        alpha = 2 * np.pi - alpha

    dE = alpha - beta
    #Calculate the velcotiy of the points
    v2, v1 = battin_vel_DA(r1, r2, a, dE, dt, mu)

    return v2, v1

def battin_A_DA(x_da, r1_da, r2_da):
    """
    Battin scalar function A (see Battin 10.4 or Armellian et al 2012) for battin_x_da function


    where
        g = s/ 2(1-x
    :param x_da: DA or float
        battin parameter
    :param r1_da:  array-like
        initial poisition vector
    :param r2_da: array-like
        final position vector
    :param s:
    :param c:
    :param mu:
    :return:
    """

    #Generate geomtric properties
    c_vec = r2_da - r1_da
    c =     op.vnorm(c_vec)
    r1mag = op.vnorm(r1_da)
    r2mag = op.vnorm(r2_da)
    s = 0.5 * (r1mag + r2mag + c)

    # Battin auxillary g(x)
    g = s / (2 * (1 - x_da**2)) #Issue is g is negative as x_da**2 ^^^^
    al_frac = (s/(2*g)).sqrt()
    be_frac = ((s-c)/(2*g)).sqrt()
    print(al_frac.cons())
    print(be_frac.cons())
    EPS = 1e-12
    
    if al_frac.cons() >= 1.0:
        al_frac = al_frac - EPS
    if be_frac.cons() >= 1.0:
        be_frac = be_frac - EPS
    
    alpha = 2 * (al_frac.arcsin())
    beta = 2 * (be_frac.arcsin())

    #construct A
    A = g**(3/2) * (alpha - alpha.sin() - beta + beta.sin())
    if A.cons() < 0:
        print(f"A is negative")
    return A

def battin_x_DA(r1_da: Union[array,NDArray], r2_da: Union[array,NDArray], dt: float, order: int, x0: float) -> DA:
    """
    :param r1_da: (1,3) ndarray of DA variables
        DA series of initial position vector
    :param r2_da: (1,3) ndarray of DA variables
        DA series of final position vector
    :param tof_nom:
        Nominal time of flight between the two points (no DA)
    :param order:
        Max order in expansion series of r1 etc
    :return:
    x_da:
        DA expansion of Battin's parameter x(d) wrt to r1_da and r2_da
    """

    # 2) get DA battin variable - expand around nominal solution
    x_da = x0 + DA(7)
    # 3) obtain f(x,DA) = ln A - ln dt
    A = battin_A_DA(x_da, r1_da, r2_da)

    def f(x_da):
        return battin_A_DA(x_da, r1_da, r2_da).log() - np.log(dt)
    
    # 4) Newton iteration to obtain DA x. This is the Implicit solver
    p = array([r1_da, r2_da])
    x_da = Implicit_solver_DA(x_da, p, f)

    return x_da

def Nf(x, f, df):
#Newton iteration
    #x is a scalar value   
    if isinstance(x, DA):
        k = DA.getMaxVariables()        #Max variable is always the temp variable
        f = f.plug(k,0)
        df = df.plug(k,0)

        if np.allclose(df.cons(), 0):
            raise ZeroDivisionError("Derivative is zero in Newton iteration")  
        x1 = x - (f / df)    # f /df 
        return x1
    
    #x is an array x = [x1, x2, x3]
    #f is the callable function
    #df is the inverse Jacobian of f wrt x
    
    if isinstance(x,array) and df.dtype == np.float64:      #DAIOD case (Csnt Jacobian)
        k = DA.getMaxVariables()
        f = f.plug(k,0)
        df = df.plug(k,0)

        step = f / df     
        x1 = x - step
        return x1
    
    if isinstance(x,array) and isinstance(df[0][0], DA):         #Higer Order Map case
        k = DA.getMaxVariables()
        NumVar = len(x) 
        DAVar = k - NumVar + 1
       
        for i in range(NumVar):
            f = f.plug(DAVar + i, 0)
            for j in range(NumVar):
                df[i][j] = df[i][j].plug(DAVar + j,0)

        J = df.cons()
        inv_J = np.linalg.inv(J)  #Inverse Jacobian
        step = inv_J @ f
        x1 = x - step
        return x1

def Implicit_solver_DA(x_da: Union[float,DA], p, f: callable):
        """
            X needs to bethe last DA variable
                x_da: Dependant Variable x(p)
                p_da: Independant DA Variables (p_nom + DA(p_da))
                f:    Function for Newtons method
                order: order to conduct the Newton iteration. MUST BE LESS THAN DA.MAXORDER()
            Return
                x_da as a function of DA(p_da). x_da = x_nom + DA(p_da)
        """
     
        i = 1
        k = DA.getMaxVariables()
        xp = x_da + DA(k)

        if isinstance(xp, np.ndarray):
            xp = array(xp)
        
        def df(fx):
            return fx.deriv(k)
    
        while i <= (DA.getMaxOrder()):
          F = f(xp, p)
          dF = df(F)
          x_da = Nf(xp, F, dF)
          i *= 2
          xp = x_da

        x_da = x_da.plug(k,0)
        

        return x_da

def Implicit_solver_DAVec(x0: Union[array,NDArray], p, f: callable, NumVariables, x0DA=False, DAIOD=False, JacDAIOD=0):
    """
    Conduct the Implicit Solver (Newton Iteration) for multi-variables inputs.
    params: x_da: array of initial variables
            p: array of dependant variables
            f: function to solve - i.e. f(x_da; p) = 0 
            NumVariables: number of variables for Newton iterations
    return: x_da: root soultion of f(x_da; p) = 0 (array) 
    """
    
    def Jac(fx):        #Calculate the Jacobian of the Matrix f(x; p). f(x)/DA(1)

        Jac = array([[fx[i].deriv(k+j) for j in range(NumVariables)] for i in range(NumVariables)])
        return Jac
    
    iter = 1
    MaxIter = DA.getMaxOrder()  #Maximum number of iterations for the Newton iteration  
    k = DA.getMaxVariables() - NumVariables + 1
    x = array([x0[i] + DA(k+i) for i in range(NumVariables)])         #Initialise Variables for Automatic differentiation


    if not DAIOD:
        k = DA.getMaxVariables() - NumVariables + 1
        
        #while iter <= MaxIter:   #Higher Order Taylor Map Newton Iteration (Recalculate Inverse Jacobian at each iteration for HOTM solution)
        F = f(x, p)          #Evaluate the function at x
        Finv = F.invert()
        rhs = array.identity(3)
        z = Finv.eval(rhs)
        J = F.linear()
        x = F.invert()
        
        return x, J
        
        #J = Jac(F)

         #   x_da = Nf(x, F, J)
          #  iter *= 2
        
        #if x0DA:        #Return all DA parts in the solution: x = x0 + M(p,x)
        #    return x0
    
        #else:           #Return  x = x0 + M(p)
        #    for i in range(NumVariables):
        #       x_da[i] = x_da[i].plug((k+i),0)
        #    return x0
    
        #Calculate the Jacobian of the callable funtion f wrt to the DA variables
    
    #Newton Iteration - to obtain x = x0 + M(p)
    if DAIOD == True:                       #DAIOD Range specific Newton Iteration (constant inverse Jacobian at x(0))
        assert np.linalg.det(JacDAIOD) != 0   #Check that Jacobian is invertible
        #Conduct Numerical Inversion of the Jacobian
        L, Pivot = la.lu_factor(JacDAIOD)
        J0inv = la.lu_solve((L, Pivot), np.eye(JacDAIOD.shape[0]))

        #DA Newton loop
        while iter <= (DA.getMaxOrder()):
            
            x_da  = x0 - (f(x0,p) @ np.linalg.inv(JacDAIOD))  #f(x0,p) is a vector, J0inv is the inverse Jacobian fif numpy
            x_da = Nf(x0, f(x0), J0inv)
            iter *= 2
            x0 = x_da
        
        return x_da

def battin_vel_DA(r1: Union[array,NDArray], r2: Union[array,NDArray], a: Union[DA,float], dE: Union[DA,float], dt: float, mu: float) -> DA:
    """
    :param r1: 3-vector initial position vector (DA series or real array)
    :param r2: 3-vector final position vector (DA series or real array)
    :param a: Semi major axis DA series
    :param dE: dE = alpha - beta
    :param dt: tof (float)
    :param mu: Standard gravitional parameter (float)
    :return:
    """

    f = 1 - (a / op.vnorm(r1)) * (1 - op.cos(dE))
    g = dt - op.sqrt(a**3 / mu) * (dE - op.sin(dE))
    g_dot = 1 - (a / op.vnorm(r2)) * (1 - op.cos(dE))
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v2, v1

def lagrange_coefficients(a: Union[float, DA], dE: Union[float, DA], r1: Union[array, NDArray], v1: Union[array, NDArray], sigma_1, mu: float) -> Union[array, NDArray]:
        """
        Calculate the lagrange coefficients for position and velocity vectors.
        :param a: Semi-major axis
        :param dE: Eccentric anomaly
        :param r1: Initial position vector
        :param mu: Standard gravitational parameter
        :return: Position and velocity vectors at t0 + dt
        Source: 16.346 Astrodynamics Fall 2008 MIT OpenCourseWare
        """
        f = 1 - (a / op.vnorm(r1)) * (1 - op.cos(dE))
        g = a*sigma_1/op.sqrt(mu) * (1 - op.cos(dE)) + (op.vnorm(r1) * op.sqrt(a/mu) * (op.sin(dE)))
        r2 = f*r1 + g*v1

        ft = - op.sqrt((mu*a)) / ((op.vnorm(r1)*op.vnorm(r2))) * op.sin(dE)
        gt = 1 - (a/(op.vnorm(r2)) * (1 - op.cos(dE)))
        v2 = ft*r1 + gt*v1

        #conservation of angular momentum

        assert np.isclose(f*gt - ft*g, 1, atol=1e-9), f"Lagrange Coefficients incorrect"

        return r2, v2

def newton_nomial_DA(x0: Union[float, NDArray], p: Union[DA, array, float, NDArray], f: callable, tol: float , MaxIter: float, order) -> float:
    
    """
    Newtons Method applied to DA to obtain Nomial Solution for dependant variable
    DA must have been initialised Prior

    :param x0: Initial guess 
    :param p: Everything other variable in f
    :parma f: Callable function f(x; p) = 0 which will be evaluated at every newton iteration
    :return: solution for x around the nominal p such that f(x) = 0
    """

    Max_variable = DA.getMaxVariables()
    
    xp = x0 + DA(Max_variable)  
    flag = True
    iter = 1
    DA.pushTO(2)
    while flag:

        F = f(xp,p)
        if isinstance(F, np.ndarray):
            F = array(F)
        
        dF = F.linear()
        if dF[Max_variable-1] == 0:
            print(f"Iteration Number: {iter}\n")
            raise ValueError("Derivative became zero during iteration") 
        

        if np.all(abs(F.cons()) < tol) or iter > MaxIter:
            flag = False
            if iter < MaxIter:
                print(f"Maximum Number of Iterations reached")

        x = xp - (F.cons()/dF[Max_variable-1])
        iter += 1
        xp = x
    
    if isinstance(xp, np.ndarray):
        xp = array(xp)
    
    x_nom = xp.cons() #Plug the last variable to zero to obtain the solution
    DA.popTO()
    return x_nom


def newton_nominal_DAVec(x0: Union[float, NDArray], p: Union[DA, array, float, NDArray], f: callable, order: int, tol: float=1e-9 , MaxIter: float=1000) -> float:
    """
    Newtons Method applied to DA to obtain Nomial Solution for dependant variable
    DA must have been initialised Prior

    :param x0: Initial guess 
    :param p: dependance variable in f such that x(p).
    :parma f: Callable function f(x; p) = 0 which will be evaluated at every newton iteration
    :return: Nomial solution for x as a Taylor Polynomial series 
    """
    NumVariables = len(x0)
    
    DA.init(order, NumVariables)
    k = DA.getMaxVariables() - NumVariables + 1
    
    #Initialise Variables for Automatic differentiation
    var_x = array.identity(NumVariables)
    xp = x0 + var_x

    flag = True
    iter = 1

    while flag:
        print(f"Iteration: {iter}")
        if iter == 5:
            print(f"xp: {xp}")
        F = f(xp, p)

        Jac = F.linear()  # Get the linear part of the function F

        assert np.linalg.det(Jac) != 0, "Jacobian is singular, therefore invertable"

        x = xp - (np.linalg.inv(Jac) @ F.cons())

        iter += 1
        if np.linalg.norm(F.cons()) < tol or iter > MaxIter:  # Check convergence
            flag = False
            if iter > MaxIter:
                print(f"Maximum Number of Iterations reached")
                raise ValueError("No convergence: Relax Tolerance or increase Iterations")
    
        xp = x
    
    xp = xp.cons()
    DA.popTO()  # Pop the top of the stack for DA variables
    return xp

def kepler_F(a: Union[float, DA], sigma: Union[float, DA], r1_norm: Union[float, DA], dM: Union[float, DA]) -> Union[float, DA]:
    """
    Kepler's equation F(dE) = dE + (sigma/sqrt(a)) * (1 - cos(dE) - (1- r1/a)*sin(dE))
    :param a: Semi-major axis
    :param sigma: Specific angular momentum
    :param dM: Change in mean anomaly
    :return: Value of Kepler's equation at E
    """

    Max_Variable = dM.getMaxVariables()
    dE = dM.cons() + DA(Max_Variable)

    for _ in range(1000):
        F = (dE + (sigma / op.sqrt(a)) * (1 - op.cos(dE)) - (1 - r1_norm / a) * op.sin(dE)) - dM
        if abs(F.cons()) < 10e-12:  # Convergence criterion
            break
        dF = F.deriv(Max_Variable)
        # dF = 1.0 + (sigma / op.sqrt(a)) * (op.sin(dE) - (1 - r1_norm / a) * op.cos(dE)) -> analytically derived dF/dM
        if dF.cons() == 0:
            raise ValueError("Derivative became zero during iteration")
        dE -= F.cons() / dF.cons()
        print(f"{F.cons() / dF.cons()}")
        print(f"dE:\n{dE}\n")
        
        
    return dE.cons()

def Kepler_DA(r1: Union[array, NDArray], v1: Union[array, NDArray], dt: float, mu: float = 3.2712440018e11,order: int = None):
    """
        High-order Kepler solver
          r0, v0 : 3-component ndarray of float *or* DA (if DA components are carried through calculation)
          dt     : propagation time [same units as μ]
          mu     : GM of central body
          order  : DA truncation order
        Returns (r, v) at t0+dt
    """
    if not DA.initialized:
        DA.init(order, r1.getMaxVariables()+1) # Initialize DA with order and number of variables
        if order is None:
            order = DA.getMaxOrder(r1[0]) + 1 if isinstance(r1[0], DA) else 0
    
    #Prepare variables
    sigma = r1.dot(v1) / op.sqrt(mu)

    energy = v1.dot(v1) / 2 - (mu / op.vnorm(r1))  #Specific orbital energy
    
    a = - mu / (2 * energy)  #Semi-major axis
    if a.cons() <= 0.0:
        raise ValueError("ERROR: hyperbolic or parabolic semi major axis")

    n = op.sqrt(mu/a**3)
    #Obtain time of passage through perigee 
    print(dt)
    dM = n*dt
    # -------------------------------------
    #Calculate the Nominal change in Eccentric anomaly - via Newton iteration
    # -------------------------------------
    a_nom = a.cons() if isinstance(a, DA) else a
    dM_nom = dM.cons() if isinstance(dM, DA) else dM
    sigma_nom = sigma.cons() if isinstance(sigma, DA) else sigma
    r1_nom = r1.cons() if isinstance(r1, DA) else r1
    dt_nom = dt if isinstance(dt, DA) else dt


    dE_nom = kepler_F(a, sigma, op.vnorm(r1), dM)
    # -------------------------------------
    # raise dE to DA variable and create taylor map
    # -------------------------------------

    dE_da = dE_nom + DA(dM.getMaxVariables())  # Create DA variable for dE
    p_da = array([r1, v1])          # Shape (2,3)
    
    def F_da(dE_da):
        #Unpack p array
        #Generate Variables for Function

        r1 = p_da[0,:]
        v1 = p_da[1,:]

        sigma = r1.dot(v1) / op.sqrt(mu)
        energy = v1.dot(v1) / 2 - mu / op.vnorm(r1)  #Specific orbital energy
        a = - mu / (2 * energy)  #Semi-major axis
        dM = op.sqrt(mu / a**3) * dt    # change in mean anomaly

        #Construct F.
        return dE_da + (sigma / op.sqrt(a)) * (1 - op.cos(dE_da)) - (1 - op.vnorm(r1) / a) * op.sin(dE_da) - dM
    
    #problem is here :(
    dE_da = Implicit_solver_DA(dE_da, F_da)

    #Recompute variables with final dE_da Map 


    # -------------------------------------
    # Calcualte the final position and velocity vectors through lagrange coefficients
    # -------------------------------------

    r2, v2 = lagrange_coefficients(a, dE_da, r1, v1, sigma, mu)

    return r2, v2

