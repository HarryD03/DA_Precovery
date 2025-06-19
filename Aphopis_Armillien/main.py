## Import Packages

from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Union, overload, Tuple

import numpy as np
from Tools.demo.sortvisu import steps
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from time import perf_counter
#Aphopis Armillien Script
def position_feasibility(r1: NDArray[np.double], r2: NDArray[np.double], Re: float = 6378.137) -> bool:
    """
    Check if the Guass IOD positions are physically feasible.

    Paramters:
    r: array-like
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
    # Test 1: ranges > 0

    # Test 2: r2, r1, r3 < Re - the asteroid has to be further away from earth at observation time as taken at night

    # Test 3: coplanarity of r1, r2, r3
    return


#Methodology: DA-based OD algorithm. - 2 BP implementation (uses Guass, Kepler and lambert implimentation)

# 0) Input observation variables and time of observation - Topocentric reference frame
RA = np.array([180, 180, 180]) # Right Ascension of observation (measured east/west direction sky longitude) [0; 24] (hours, minutes, seconds)
DEC = np.array([0, 0, 0])      # Declination of observation (measured north/south direction sky latitude) [-90; 90] (degrees, arcminutes, arcseconds)
obs_1 = np.array([2003, 11, 8, 12])    #placeholder   obs_1[0] is year, obs_1[1] is month, obs_1[2] is day, obs_1[3] is UT.
obs_2 = np.array([2003, 11, 8, 14])    #placeholder
obs_3 = np.array([2003, 11, 8, 15])    #placeholder
east_longitude = 10.0              #placeholder East Longitude of observer (degrees)

def create_da_los_vectors(ra: Union[array, NDArray], dec: Union[array, NDArray]) -> Union[array, NDArray]:
    """
    Calculate the line of sight unit vectors for the given right ascension and declination.
    Run this when the Nomial solution of RA and DEC is known.
    Run again if Nomial solution changes.
    :param ra:
        1xN array of right ascension angles (radians) in the range [0; 24].
    :param dec:
        1xN array of declination angles (radians) in the range [-pi/2, pi/2].
    :return:
        3XN array of LOS vectors. column strucutre: [ρ̂x, ρ̂y, ρ̂z]ᵢ. Row index corresponds to observation instance, and it repreats for N observations
    """
    assert ra.shape[0] == 1, "RA must be a row vector"
    assert dec.shape[0] == 1, "DEC must be a row vector"
    assert ra.shape[1] == dec.shape[1], "Number of RA and DEC observations must match"

    num_obs = ra.shape[1]
    los_vectors = np.empty((3, num_obs), dtype=object)

    for i in range(num_obs):
        los_vectors[:,i] = array([
            op.cos(dec[0,i]) * op.cos(ra[0,i]),
            op.cos(dec[0,i]) * op.sin(ra[0,i]),
            op.sin(dec[0,i])
        ])

    return los_vectors


los_nomial = create_da_los_vectors(RA, DEC)

def J0(y,m,d) -> float:
    """
    Calculate the Julian days for the year,month and day of observation
    :param y: year
    :param m: month
    :param d: days
    :return: J0 - julian day
    """
    assert 1901 <= y <= 2099, "Year must be between 1901 and 2099"
    assert 1 <= m <= 12, "Month must be between 1 and 12"
    assert 1 <= d <= 31, "Day must be between 1 and 31"

    j0 = 367*y - int((7*(y +int((m+9)/12))/4)) + int((275*m)/9) + d + 1721013.5

    return j0

def zeroTo360(theta: float) -> float:
    """
    Normalizes an angle to lie within [0, 360] degrees (inclusive of 360)

    :param theta: angle in degrees
    :return: normalized angle between 0 and 360 degrees
    """
    result = np.mod(theta, 360)
    if result == 0 and theta != 0:
        result = 360
    return result

def LST(y: float, m: float, d: float, ut: float, EL: float) -> float:
    """
    This function calculates the local sideral time (LST) of observations.
    :param y: year
    :param m: month
    :param d: day
    :param ut: universal time (hours)
    :param EL: east longitude (degrees)
    :return:
    Local sideral time: lst (degrees)
    """

    j0 = J0(y, m, d)        # get Julain days

    j = (j0 - 2451545.0)/36525.0 #where j is time between Julain centeries between j0 and j2000
    theta_g0 = 100.4606184 + 36000.77004*j + 0.000387933*j**2 - 2.583e-8*j**3   #Greenwich sideral time at 0h UT (degrees)
    theta_g0 = zeroTo360(theta_g0)                                              #ensure between 0 and 360 degrees
    theta_g = theta_g0 + 360.98564736629*ut                                  #greenwich sidereal time at UT (degrees)
    lst = theta_g + EL                                                          #local sideral time (degrees)

    lst = zeroTo360(lst)                                                        #ensure between 0 and 360 degrees

    time_epoch = j0 + ut/24.0
    return lst, time_epoch

def equatorial_to_eclipitcJ2000(dr: float, lst: float, h_e: float) -> float:
    """
    Rotate Coordinates from equatorial reference frame to ecliptic reference frame at J2000 epoch.
    :param dr: Perpendicular distance of the observation site from Earth's rotation axis
    :param lst:
    :param h_e:
    :return:
        r_obs: The position vector of the observation station within the ecliptical J2000 reference frame.
    """
    di = 23.4               #angular displacement of Equator to ecliptic plane (degrees)
    r_obs_eq = np.array([dr*np.cos(lst), dr*np.sin(lst),h_e]).T
    R = np.array([
        [1, 0, 0],
        [0, np.cos(np.deg2rad(di)), np.sin(np.deg2rad(di))],
        [0, -np.sin(np.deg2rad(di)), np.cos(np.deg2rad(di))]
    ])

    r_obs_ec = np.cross(R, r_obs_eq)
    return r_obs_ec

lst, t_obs = LST(2003, 11, 8, 12, east_longitude)              #placeholders values
r_obs = equatorial_to_eclipitcJ2000(1000, lst, 0)            #placeholder values

# 1) First-Guess Orbit: Classic Guass method (Right Ascension, Declination, t1->t3). Solve for middle range 'epoch'

# Initialise/Pre-condition observation times
r_E = 6371                                          #radius of Earth in km
pos_helocentric = r_obs + r_E                       #Helocentric position of observation site at observation 1
t = np.array([obs_1[3], obs_2[3], obs_3[3]])              #UT of different observations
def Guass_8th_seed(pos_obs: Union[NDArray, array], obs_dir: Union[NDArray, array], t: NDArray) -> NDArray[np.double]:
    #Taken from Orbital Mechanics for Engineering Students (4th ed.) by Curtis. p.242 Algorithm 5.5. and Armillien Aphopis
    #Inputs:
    #   Pos_obs: 2D array of Observer positions in Heliocentric frame of reference from angle rotation matrix
    #           Rows are compoents, columns are observations instances
    #   t: 1D array of times in seconds
    #   obs_dir: 1D array of unit vectors pointing from observer to the point of interest
    # 2-BP assumption:
    #   Assume the observations lie on the same plane
    mu = 1.32712440018e11               #Suns gravitional parameter
    dt_1 = t[1] - t[0]
    dt_3 = t[2] = t[1]
    dt = dt_3 - dt_1

    p1 = obs_dir[1].cross(obs_dir[2])
    p2 = obs_dir[1].cross(obs_dir[2])
    p3 = obs_dir[0].cross(obs_dir[1])

    D0 = obs_dir[0].dot(p1)
    D11 = pos_obs[0].dot(p1)
    D12 = pos_obs[0].dot(p2)
    D13 = pos_obs[0].dot(p3)
    D21 = pos_obs[1].dot(p1)
    D22 = pos_obs[1].dot(p2)
    D23 = pos_obs[1].dot(p3)
    D31 = pos_obs[2].dot(p1)
    D32 = pos_obs[2].dot(p2)
    D33 = pos_obs[2].dot(p3)

    A = 1 / D0 * ((-D12 * dt_3/ dt) + D22 + (D32 * dt_1/dt))
    B = 1 / (6*D0) * (D12*(dt_3**2 - dt**2)*dt_3/dt + (D32*(dt**2 - dt_1**2) * dt_1/dt))
    E = pos_obs[1].dot(obs_dir[1])
    R_2_squared = pos_obs[1].dot(pos_obs[1])

    #obtain coefficient values for the 8th degree position polynomial
    a = -1 * (A**2 + 2*A*E + (op.R_2_squared.sqr()))
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
    r_2 = real_roots.T                                                #Store possible solutions of r_2
    if len(real_roots) > 1:
        print("There are more than 1 positive real roots.")
        range_1 = np.zeros(len(real_roots)).T
        range_2 = np.zeros(len(real_roots)).T
        range_3 = np.zeros(len(real_roots)).T

        r_1 = np.zeros(len(real_roots)).T
        r_3 = np.zeros(len(real_roots)).T
        for i in range(len(real_roots)):
            print(f"Root {i}: {r_2[i]}\n")
            # build range1,range2,range3 and conduct feasibility tests.
            # Ranges are defined through truncated Lagragne coefficients so they're not accuarate and refinement required
            num = (6*(D31 * (dt_1/dt_3) + D21 * (dt/dt_3))*(r_2[i]**3)) + (mu * D31 * (dt**2 - dt_1**2)* (dt_1/dt_3))
            den = (6 * r_2[i]**3) + (mu * (dt**2 - dt_3**2))
            range_1[i] = 1 / D0 * ( (num / den) - D11)

            num = (6*(D13 * (dt_3/dt_1) + D23 * (dt/dt_1))*(r_2[i]**3)) + (mu * D13 * (dt**2 - dt_3**2)* (dt_3/dt_1))
            den = (6 * r_2[i]**3) + (mu * (dt**2 - dt_3**2))
            range_3[i] = 1 / D0 * ((num / den) -D33)

            range_2[i] = A + (mu * B) / (r_2 ** 3)

            # calculate r1, r3
            r_1[i] = pos_obs[0] + range_1*obs_dir[0]
            r_3[i] = pos_obs[2] + range_2*obs_dir[2]

            #Assess the 3 positions for feasibility

            #Test 1: ranges > 0

            #Test 2: r2, r1, r3 < Re - the asteroid has to be further away from earth at observation time as taken at night


    else:
        print(f"There is 1 positive real root: {r_2}\n")
        num = (6 * (D31 * (dt_1 / dt_3) + D21 * (dt / dt_3)) * (r_2 ** 3)) + (
                    mu * D31 * (dt ** 2 - dt_1 ** 2) * (dt_1 / dt_3))
        den = (6 * r_2 ** 3) + (mu * (dt ** 2 - dt_3 ** 2))
        range_1 = 1 / D0 * ((num / den) - D11)

        num = (6 * (D13 * (dt_3 / dt_1) + D23 * (dt / dt_1)) * (r_2 ** 3)) + (
                    mu * D13 * (dt ** 2 - dt_3 ** 2) * (dt_3 / dt_1))
        den = (6 * r_2 ** 3) + (mu * (dt ** 2 - dt_3 ** 2))
        range_3 = 1 / D0 * ((num / den) - D33)

        range_2 = A + (mu * B) / (r_2 ** 3)

        # calculate r1, r3
        r_1 = pos_obs[0] + range_1 * obs_dir[0]
        r_3 = pos_obs[2] + range_2 * obs_dir[2]

    #obtain the [r1,r2,r3] [range1,range2,range3]. Rows are the real root, columns are positions
    position = [r_1,r_2,r_3]
    ranges = [range_1,range_2,range_3]

    return position, ranges

pos_obs, range_obs = Guass_8th_seed(pos_helocentric, los_nomial, t)

# 2) Define DA variables for observations and ranges
DA.init(5,9)
RA_da = array([RA[0] + DA(1), (RA[1]) + DA(2), (RA[2]) + DA(3)])
DEC_da = array([DEC[0] + DA(4), DEC[1] + DA(5), DEC[2] + DA(6)])
range_da = array([range[0] + DA(7), range[1] + DA(8), range[2] + DA(9)])        #obtain nomial states from Guass_8th_seed function



los_vectors = create_da_los_vectors(RA_da, DEC_da)  #X Y Z compoents represented by rows, observation instance represented by columns
LosMap: array = los_vectors - los_vectors.cons()    #The DA map for line of sight unit vector. X Y Z components represented by rows, instances via columns.

#Isolate the 2nd and 3rd instance of the los vectors
los_2 = los_nomial[:,1] + LosMap[:,1]
los_3 = los_vectors[:,2] + LosMap[:,2]

#DA evaluation of equation (32) in Armellin - range and los are DA vectors.
"""
Take the nomial solution of the position (Calculated by Guass method), and create a DA variable for the cart position in terms
of the other independant variables. i.e. RA DEC Range. DA RA and DA DEC are defined through the LOS map. DA Range is defined through
the rangeMap. This returns a carteisan position variable with a nomial and DA components
"""
#rangeMap dot LosMap = rMap
#Take the second element for r2
rangeMap: array = range_da - range_da.cons()

r2Map = rangeMap[1].dot(LosMap[:,1])
r3Map = rangeMap[2].dot(LosMap[:,2])

r2_DA = pos_obs[1] + r2Map
r3_DA = pos_obs[2] + r3Map


# 3) Forward Propagate point 2 to point 3 Using Battin's scalar implicit equation: -> Obtain taylor map at v2 as a function of observations

def NLambertBattin(s: Union[float, DA], x: Union[float, DA], c: Union[float, DA], dt: float):
    """
    This function performs calculations related to the Lambert Battin problem, which
    is used in orbital mechanics to determine trajectory and velocity vectors for
    spacecraft. The inputs `s`, `x`, `c`, and `dt` are parameters that influence
    the computations of the angles, intermediate results, and the resulting
    logarithmic output based on the provided formula in Armellian et al 2012.

    :param s: A float or dual number (DA) representing the semi-parameter
        related to the Lambert Battin formulation.
    :param x: A float or dual number (DA) representing battin variab;e, typically
        related to the shape or configuration in the problem space.
    :param c: A float or dual number (DA) denoting chord length of the two positions.
    :param dt: A float representing time interval or duration used in the
        computation of resulting trajectories.
    :return: A tuple containing two elements:
        - The first element is a float or dual number (DA) representing the computed
          logarithm of the quantity `A` adjusted by time `dt`.
        - The second element is a float or dual number (DA) representing the
          derivative of the function.
    """
    g = s /(2 * (1-x**2))
    dg = (s * x) / (1-x**2)
    alpha = 2*op.asin(op.sqrt(s/(2*g)))
    dalpha = (-s * dg)/ (g**2 * op.sqrt(-s * (s - (2*g)) / g**2))
    beta = 2*op.asin(op.sqrt( (s-c) / (2 * g)))
    dbeta = ( (c - s)* g) / (g**2*(op.sqrt( (s - c) * (c + (2 * g) -s ) / (g**2) )))

    A = g**(3/2) * (alpha - op.sin(alpha) - beta + op.sin(beta))
    dA = -1/2*(op.sqrt(g)) * ( 3*dg (-alpha + op.sin(alpha) + beta - op.sin(beta))
                               + 2*g*(dalpha * (dalpha * (op.cos(x) - 1) - dbeta*(op.cos(beta) - 1))))

    #Real solution handling
    if isinstance(A, float):
        f = np.log10(A) - np.log10(dt)
        df = dA/A
        return f, df, alpha, beta
    #DA handling
    if isinstance(A,DA):
        f = op.log10(A) - op.log10(dt)
        return f, df, alpha, beta

def Newton_iteration(x: Union[float, DA], 
                    update_f: Callable[[Union[float, DA]], Tuple[Union[float, DA], Union[float, DA]]], 
                    tol: float, 
                    max_iter: int) -> Union[float, DA]:
    """
    Unified Newton iteration method that handles both real numbers and DA types.
    Args:
        x: Initial guess (float or DA)
        update_f: Function that takes x and returns tuple of (f, f_prime) at that x
        tol: Tolerance for convergence (only used for real numbers)
        max_iter: Maximum number of iterations (only used for real numbers)
    Returns:
        Converged solution (float or DA)
    Raises:
        ValueError: If derivative becomes zero or max iterations reached without convergence
    """
    if isinstance(x, DA):
        order = x.getMaxOrder()
        for i in range(1, order + 1):
            f, f_prime = update_f(x)
            if f_prime == 0:
                raise ValueError("Derivative became zero during iteration")
            x = x - f/f_prime
            i *= 2
            if i == order + 1:
                return x
    else:
        k = 0
        while k < max_iter:
            xp = x
            f, f_prime = update_f(xp)
            if abs(f_prime) < 1e-15:  # Numerical threshold for zero
                raise ValueError("Derivative too close to zero")
            x = xp - f/f_prime
            if abs(xp - x) <= tol:
                return x
            k += 1
        raise ValueError(f"Failed to converge after {max_iter} iterations")


def lambert_update_wrapper(s: Union[float, DA], c: Union[float, DA], dt: float):
     
        def update_function(x: Union[float, DA]) -> Tuple[Union[float, DA], Union[float, DA]]:
            f, df, _, _ = NLambertBattin(s, x, c, dt)  # Ignore alpha, beta during iteration
            return f, df

        return update_function
 
def lambert_battin_zeroth(r1: Union[array,NDArray], r2: Union[array,NDArray], dt: float, obs: array) -> Tuple[NDArray, NDArray]:
    """
    Solve Lambert Problem using Battin's scalar implicit equation.
    Find the zeroth solution for the taylor polynomial series
    :param r1: Initial Position.                            Rows are components of the position vector.
    :param r2: Final Position.                              Rows are components of the position vector.
    :param dt: time of flight between the two points
    
    Returns:
        Tuple containing:
        - v2: Zeroth order (nomial) velocity at point 2
        - v1: Zeroth order (nomial) velocity at point 1
    
    Raises:
        ValueError: If solution results in hyperbolic or parabolic orbit

    """
    mu = 1.32712440018e11  # Suns gravitional parameter [km^3/s^2]

    # Obtain Nomial variables for taylor seed

    if isinstance(r1, array):
        r1 = r1.cons()
    if isinstance(r2, array):
        r2 = r2.cons()

    c = np.linalg.norm(r2 - r1)                             #Chord length of two points
    s = (np.linalg.norm(r1) + np.linalg.norm(r2) + c) / 2   #Semi-perimeter of two points
    TA = op.acos(r1.dot(r2) / (op.vnorm(r1) * op.vnorm(r2)))
    mu = 1.32712440018e11                                   #Suns gravitional parameter [km^3/s^2]

    #Need to obtain the first guess - zeroth order solution to ensure DA converges
    x_guess = 0.0                                               #initial guess as Battins formulation exisits in domain [-1,1]
    
    #Newton iteration to obtain x^0 term
    update_func = lambert_update_wrapper(s, c, dt)                      #This is required as alpha and beta are not used in the update function
    x_0 = Newton_iteration(x_guess, update_func, 1e-12, 1000)      #nominal x value with r.cons()

    #TODO:
    #   Need to isolate the first term of [r] if inputted as DA variable (r_zeroth = r.cons())
    #   Need to conduct Newton iteration of x for DA variables, using x as the initial guess and enabling [r] as full DA variables.
    #   Need to ensure post processing is DA compliant syntax

    #post process semi-major axis of final result
    a = s / (2*(1 - x**2))

    #re-calculate alpha and beta for final semi-major axis - provided by Vallado p.497
    if a.cons()  <= 0.0:
        raise ValueError("ERROR: hyperbolic or parabolic semi major axis")

    _, _, alpha, beta = NLambertBattin(s, x, c, dt)
    if TA > np.pi:
        beta = -beta

    a_min = s / 2
    t_min = np.sqrt(a_min ** 3 / mu) * (np.pi - beta + np.sin(beta))

    if dt > t_min:
        alpha = 2 * np.pi - alpha

    delta_E = alpha - beta
    f = 1 - (a / op.vnorm(r1)) * (1 - np.cos(delta_E))
    g = dt - (np.sqrt(a ** 3 / mu) * (delta_E - np.sin(delta_E)))
    g_dot = 1 - (a / np.linalg.norm(r2)) * (1 - np.cos(delta_E))

    v1_nom = (r2 - (f * r1)) / g
    v2_nom = (g_dot * r2 - r1) / g

    return v2_nom, v1_nom, x




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
        DA.init(0,1)        #nothing happens carry r1/r2 DA settings over - real numbers
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

    def battin_dF(x):
        g = s / (2 * (1.0 - x ** 2))
        dg = (s * x) / (1.0 - x ** 2)
        alpha = 2.0 * op.asin(op.sqrt(s / (2 * g)))
        dalpha = (-s * dg) / (g ** 2 * op.sqrt(-s * (s - (2 * g)) / g**2))
        beta = 2.0 * op.asin(op.sqrt((s - c) / (2 * g)))
        dbeta = ((c - s) * g) / (g ** 2 * (op.sqrt((s - c) * (c + (2 * g) - s) / (g ** 2))))

        A = g ** (3 / 2) * (alpha - op.sin(alpha) - beta + op.sin(beta))
        dA = -1 / 2 * (op.sqrt(g)) * (3 * dg*(-alpha + op.sin(alpha) + beta - op.sin(beta))
                                      + 2 * g * (dalpha * (dalpha * (op.cos(x) - 1) - dbeta * (op.cos(beta) - 1))))
        return dA/A

    x = 0.0                    #guess at 0.0

    for _ in range(max_iter):
        f0 = battin_F(x)
        df0 = battin_dF(x)
        x -= f0/df0
        if abs(f0) < 10e-12:
            break

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
    t_min = np.sqrt(a_min ** 3 / mu) * (np.pi - beta + np.sin(beta))

    if dt > t_min:
        alpha = 2 * np.pi - alpha

    dE = alpha - beta
    #Calculate the velcotiy of the points
    v2, v1 = battin_vel_DA(r1, r2, a, dE, dt, mu)

    return v2, v1

def battin_A_DA(x_da, r1_da, r2_da, mu=1.32712440018e11):
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
    g = s / (2 * (1 - x_da**2))

    alpha = 2 * op.asin( op.sqrt( s / (2 * g)))
    beta = 2 * op.asin( op.sqrt( (s - c) / (2*g)))

    #construct A
    A = g**(3/2) * (alpha - op.sin(alpha) - beta + op.sin(beta))
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
        DA expansion of Battin's parameter x(d)
    """

    #1) geometric scalars
    c_vec = r2_da - r1_da
    c =     op.vnorm(c_vec)
    r1mag = op.vnorm(r1_da)
    r2mag = op.vnorm(r2_da)
    s = 0.5 * (r1mag + r2mag + c)

    # 2) get DA battin variable - expand around nominal solution
    x_da = x0 + DA(7)
    # 3) obtain f(x,DA) = ln A - ln dt
    A = battin_A_DA(x_da, r1_da, r2_da)
    f = DA.log10(A) - np.log10(dt)

    # 4) Newton iteration to obtain DA x. Do this or DA.invert(f,7)
    k = 1
    while k <= (DA.getMaxOrder()+1):
        x_da = x_da - f / f.deriv(7)      # x = x - f/f'(x)
        f = DA.log10(battin_A_DA(x_da, r1_da, r2_da)) - math.log10(dt) #update f
        k *= 2

    return x_da

def battin_vel_DA(r1: DA, r2: DA, a: DA, dE: DA, dt: float, mu: float) -> DA:
    """
    :param r1: 3-vector initial position vector (DA series)
    :param r2: 3-vector final position vector (DA series)
    :param a: Semi major axis DA series
    :param dE: dE = alpha - beta
    :param dt: tof (float)
    :param mu: Standard gravitional parameter (float)
    :return:
    """

    f = 1 - (a / op.vnorm(r1)) * (1 - op.cos(dE))
    g = dt - op.sqrt(a**3 / mu) * (dE - op.sin(dE))
    g_dot = 1 - (a / op.vnorm(r2)) * (1 - op.cos(dE))
    v2 = (r2 - f * r1) / g
    v1 = (g_dot * r2 - r1) / g
    return v2, v1

#-------------------------------------------
# APPLY BATTIN SOLUTION TO LAMBERT - obtain v2 and v3 for r2 and r3
#--------------------------------------------
dt_3 = t[3] - t[2]

v3, v2 = lambert_battin_DA(pos_obs[1], pos_obs[2], dt_3, 100)

# ------------------------------------------
# 4) Kepler Back Propagation: t2 -> t1 -> Propagate full state to t1
from typing import Callable, Tuple, Union
# ------------------------------------------
def Kepler_DA(r1: Union[array, NDArray], v1: Union[array, NDArray], dt: float, mu: float = 3.2712440018e11,order: int = None):
    """
        High-order Kepler solver
          r0, v0 : 3-component ndarray of float *or* DA (if DA components are carried through calculation)
          dt     : propagation time [same units as μ]
          mu     : GM of central body
          order  : DA truncation order
        Returns (r, v) at t0+dt
    """
    if not DA.initialized():
        DA.init(order, N_GEN+1) # Initialize DA with order and number of variables
        if order is None:
            order = DA.getMaxOrder(r1[0]) + 1 if isinstance(r1[0], DA) else 0
    
    #Prepare variables
    sigma = r1.dot(v1) / op.sqrt(mu)

    energy = r1.dot(r1) / 2 - mu / op.sqrt(op.vnorm(r1))  #Specific orbital energy
    a = -mu / (2 * energy)  #Semi-major axis
    if a <= 0.0:
        raise ValueError("ERROR: hyperbolic or parabolic semi major axis")

    dM = op.sqrt(mu / a**3) * dt    # change in mean anomaly
    
    # -------------------------------------
    #Calculate the Nominal change in Eccentric anomaly - via Newton iteration
    # -------------------------------------
    a_nom = a.cons() if isinstance(a, DA) else a
    dM_nom = dM.cons() if isinstance(dM, DA) else dM
    sigma_nom = sigma.cons() if isinstance(sigma, DA) else sigma
    r1_nom = r1.cons() if isinstance(r1, DA) else r1
    dt_nom = dt if isinstance(dt, DA) else dt

    def kepler_F(a: Union[float, DA], sigma: Union[float, DA], r1_norm: Union[float, DA], dM: Union[float, DA]) -> Union[float, DA]:
        """
        Kepler's equation F(dE) = dE + (sigma/sqrt(a)) * (1 - cos(dE) - (1- r1/a)*sin(dE))
        :param a: Semi-major axis
        :param sigma: Specific angular momentum
        :param dM: Change in mean anomaly
        :return: Value of Kepler's equation at E
        """
        dE = dM
        for _ in range(max_iter):
            F = dE + (sigma / op.sqrt(a)) * (1 - op.cos(dE)) - (1 - r1_norm / a) * op.sin(dE)
            dF = 1.0 + (sigma / op.sqrt(a)) * (op.sin(dE) - (1 - r1_norm / a) * op.cos(dE))
            if dF == 0:
                raise ValueError("Derivative became zero during iteration")
            dE += F / dF
            if abs(F) < 10e-12:  # Convergence criterion
                break
        return dE

    dE_nom = kepler_F(a_nom, sigma_nom, r1_nom, dM_nom)
    # -------------------------------------
    # raise dE to DA variable and create taylor map
    # -------------------------------------
    dE_da = dE_nom + DA(N_GEN + 1)  # Create DA variable for dE
    F_da = dE_da + (sigma / op.sqrt(a)) * (1 - op.cos(dE_da)) - (1 - r1 / a) * op.sin(dE_da) - dM
    
    dE_da = F_da.invert()  # Invert F_da to get dE_da as a function of DA variables.
                           # F_da.invert() returns the taylor polynomial series of dE_da in terms of the other DA variables.

    # -------------------------------------
    # Calcualte the final position and velocity vectors through lagrange coefficients
    # -------------------------------------
    def lagrange_coefficients(a: Union[float, DA], dE: Union[float, DA], r1: Union[array, NDArray], mu: float) -> Tuple[Union[array, NDArray], Union[array, NDArray]]:
        """
        Calculate the lagrange coefficients for position and velocity vectors.
        :param a: Semi-major axis
        :param dE: Eccentric anomaly
        :param r1: Initial position vector
        :param mu: Standard gravitational parameter
        :return: Position and velocity vectors at t0 + dt
        """
        f = 1 - (a / op.vnorm(r1)) * (1 - op.cos(dE))
        g = op.sqrt(a**3 / mu) * (dE - op.sin(dE))
        g_dot = 1 - (a / op.vnorm(r2)) * (1 - op.cos(dE))

        r = r1 * f + g * v1
        v = (g_dot * r2 - r1) / g

        return r, v

    r2, v2 = lagrange_coefficients(a, dE_da, r1, mu)

    return r2, v2

dt = t[0] - t[1]  # time of flight between point 1 and point 2
r1_DA, v1_DA = Kepler_DA(r2_DA, v2, dt, order=DA.getMaxOrder(), mu=1.32712440018e11) 


#--------------------------------------------
# Define observables to asteroids poisition vector at t1 by evaluating equation (30) and inverting (31) in DA
#           compute ra, dec and los through r1 v1. -> evaluate equation (30) in Armellian et al 2012
#           los = pos - obs_pos        
#--------------------------------------------

los_computed_1 = r1_DA - pos_obs[0]  # LOS between computed asteroid position at t1 and the observation position at t1 equation (30).

los_computed_dir = array([
    op.cos() * op.cos(RA_da[0]),  # X component of the LOS vector
    op.cos(DEC_da[0]) * op.sin(RA_da[0]),  # Y component of the LOS vector
    op.sin(DEC_da[0]),                      # Z component of the LOS vector
    ])

DirMap: array = los_computed_dir - los_computed_dir.cons()  # The DA map for line of sight unit vector. X Y Z components represented by rows, instances via columns.
InvMap = DirMap.invert()  # Invert the DA map to obtain the delta angles for RA and DEC.

RA_computed = InvMap.eval(los_computed_1)
DEC_computed = InvMap.eval(los_computed_1)

# ------------------------------------------
# The computed DA angles are defined as a function of the observation errors.
# Now want to find exact values of rho 2 and rho 3 and taylor expansion about t2.

# Need to define the DA residuals between the computed and observed angles.
# ------------------------------------------- 

RA_residual = RA_computed - RA_da            # Residual for Right Ascension (nomial + infitismal part)
DEC_residual = DEC_computed - DEC_da          # Residual for Declination

# For exact values of rho 2 and rho 3, the nomial/constant part of the residuals need to equal 0.
# ----------------------------------------
# a) find variation of rho 2 and 3 to cancel the RA and DEC residual constants (residual -> 0)
# b) Express r2,v2 as a taylor polynomial of observation errors/variations only.

RA_residual_variations = RA_residual - RA_residual.cons()  # DA map for Right Ascension residuals
DEC_residual_variations = DEC_residual - DEC_residual.cons()  # DA map for Declination residuals

DirMap = array([
    RA_residual_variations, #Taylor polynomial maps
    DEC_residual_variations,
    RA_da[0] - RA_da[0].cons(),   #identity variables to make it 8 genreators??
    RA_da[1] - RA_da[1].cons(),
    RA_da[2] - RA_da[2].cons(),
    DEC_da[0] - DEC_da[0].cons(),
    DEC_da[1] - DEC_da[1].cons(),
    DEC_da[2] - DEC_da[2].cons()
])

InvMap = DirMap.invert()  # Invert the DA map to obtain the delta angles for RA and DEC.

rho_2_map = InvMap[-1]  # Extract the rho_2 map from the inverted map - DA part for rho_2
rho_3_map = InvMap[-2]  # Extract the rho_3 map from the inverted map

rho_2_poly_plus = rho_2_map.Eval([-RA_residual.cons(), -DEC_residual.cons()])   #subsitute variation of angle residual
rho_3_poly_plus = rho_3_map.Eval([-RA_residual.cons(), -DEC_residual.cons()])   #subsitute variation of angle residual

rho2_const = rho_2_poly_plus.cons()  # Extract the constant part of the rho_2 polynomial ρ2^{1+}   ← Newton increment
rho3_const = rho_3_poly_plus.cons()  # Extract the constant part of the rho_3 polynomial ρ3^{1+}   ← Newton increment

rho2_variation = rho_2_poly_plus - rho2_const # Extract the variation part of the rho_2 polynomial
rho3_variation = rho_3_poly_plus - rho3_const # Extract the variation part of the rho_3 polynomial

# ------------------------------------------ 
# Iteration two 
# -------------------------------------------

rho_2_minus = rho_2_#range_2_minus = range_2 before the update + range_2 after update + variation of rho_2.

"""
1) Iteration procedure ends when the angle residual constant are smaller than a prescribed tolerance.
rho_2 = rho_2_minus + rho_2_plus = rho_2.const() + rho_2_map(RA[0],RA[1], RA[2], DEC[0],DEC[1],DEC[2])
rho_3 = rho_3_minus + rho_3_plus = rho_3.const() + rho_3_map(RA[0],RA[1], RA[2], DEC[0],DEC[1],DEC[

2) find refined line of sight vectors for point 2 and point 3:


3) Obtain the position vector at point 2 and point 3 using the refined rho_2 and rho_3 values.
    r = q + rho_2 * los_2

4) conduct Lambert Arc to get velcoities at point 2 and point 3
"""

# 5) Enforce the re-computed point 1 angles ,catch the measuremts and refine ranges:
#       a) From residual maps between computed and observed and progress to 0
#       b) invert maps to obtain updated delta-ranges.
#       c) repeat till observation error -> 0.

# Outcome: Taylor series [r2,v2] as a function of 6 observation errors
#       a) reproduce Guass at 0 error through DA
#       b) evalate

#Methodology: DA-based OD algorithm evaluation (propagate to point 4) and observe error between DA and pointwise evaluation -> taken from aphosis close encounter paper.

# 1) Obtain [r2,v2] as a function of 6 observation errors through DA

# Guass + Lambert + Kepler + DA correction at point 1 + Kepler to point 2. 

# 2) Define dynamics model -> 2BP
# 3) Inegrate once with a DA numerical integration scheme 
#       a) DA-RK78 integrator of taylor expansion -> DONE in DACE
#       b) Extract state at desired epoch (t2) -> Done in DACE
# 4) Verify if Domain splitting required (optional) -> DO later 
#       a) evaluate truncation error and conduct ADS???
# 5) Invert taylor expansion to obtain a polynomial map