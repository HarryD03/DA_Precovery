import numpy as np
from typing import Callable, List, Union, overload, Tuple
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray
import poliastro as pl
from astropy.time import Time


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
    # Special case: if theta is exactly a multiple of 360 (including 360 itself), return 360.0
    if np.isclose(result, 0.0) and theta > 0:
        return 360.0
    return result

def LST(y: float, m: float, d: float, ut: float, EL: float) -> float:
    """
    This function calculates the local sideral time (LST) of observations.
    :param y: year
    :param m: month
    :param d: day
    :param ut: universal time [hours, minutes, seconds]
    :param EL: east longitude (degrees)
    :return:
    Local sideral time: lst (degrees)
    """

    j0 = J0(y, m, d)        # get Julain days

    j = (j0 - 2451545.0)/36525.0 #where j is time between Julain centeries between j0 and j2000
    theta_g0 = 100.4606184 + 36000.77004*j + 0.000387933*j**2 - 2.583e-8*j**3   #Greenwich sideral time at 0h UT (degrees)
    theta_g0 = zeroTo360(theta_g0)                                              #ensure between 0 and 360 degrees
    
    UT = ut[0] + ut[1]/60 + ut[2]/3600
    theta_g = theta_g0 + 360.98564736629*UT/24                                  #greenwich sidereal time at UT (degrees)
    lst = theta_g + EL                                                          #local sideral time (degrees)

    lst = zeroTo360(lst)                                                        #ensure between 0 and 360 degrees

    time_epoch = j0 + UT/24.0
    return lst, time_epoch

def lst(obs_time):
    """
    Calculate Local Sidereal Time (LST) for a given date and location.

    Parameters:
    y (int or array): Year(s)
    m (int or array): Month(s)
    d (int or array): Day(s)
    ut (float or array): Universal Time(s) in hours
    EL (float or array): East Longitude(s) in degrees

    Returns:
    array: Local Sidereal Time(s) in hours
    """
    if isinstance(y, np.ndarray):
        # Ensure all input arrays have the same length
        assert len(y) == len(m) == len(d) == len(EL)

        # Convert input arrays to numpy arrays
        y = np.array(y)
        m = np.array(m)
        d = np.array(d)
        EL = np.array(EL)

        # Calculate LST for each input value
        lsts = np.zeros_like(y)
        t_observation = np.zeros_like(y)
        for i in range(len(y)):
            # Calculate Julian Date
            JD = J0(y[i], m[i], d[i])

            # Calculate Greenwich Mean Sidereal Time (GMST)
            GMST = gmst(JD,ut[i])

            # Calculate LST
            LST = GMST + EL[i] 

            # Store LST value
            lsts[i] = LST
            t_observation[i] = JD + ut[i]/24

        return lsts, t_observation
    else:
        # Single input value case
        JD = J0(y, m, d)
        GMST = gmst(JD, ut)
        LST = GMST + EL
        t_observation = JD + ut/24
        return LST, t_observation


def gmst(JD, UT):
    """
    Calculate Greenwich Mean Sidereal Time (GMST) from a Julian Date (JD).

    Parameters:
    JD (float): Julian Date (JD)

    Returns:
    float: Greenwich Mean Sidereal Time (GMST) in hours
    """
    T = (JD - 2451545.0) / 36525.0
    GMST0 = 100.4606184 + 36000.77004*T + 0.000387933*T**2 - 2.583e-8*T**3   #Greenwich sideral time at 0h UT (degrees)
    GMST0 = zeroTo360(GMST0)

    GMST = GMST0 + 360.98564736629*(UT)

    return GMST

def equatorial_to_eclipitcJ2000(dr: float, lst: NDArray, h_e: float) -> float:
    """
    Rotate Coordinates from equatorial reference frame to ecliptic reference frame at J2000 epoch.
    :param dr: Perpendicular distance of the observation site from Earth's rotation axis
    :param lst: the local side time of the observing site
    :param h_e: The height of the observer above the equatorial plane
    :return:
        r_obs: The position vector of the observation station within the ecliptical J2000 reference frame.
    """
    di_rad = np.deg2rad(23.4)               #angular displacement of Equator to ecliptic plane (degrees)
    r_obs_eq = np.array([dr*np.cos(lst), dr*np.sin(lst), np.full_like(lst, h_e)])
    
    R = np.array([
        [1, 0, 0],
        [0, np.cos(di_rad), np.sin(di_rad)],
        [0, -np.sin(di_rad), np.cos(di_rad)]
    ])
    r_obs_ec = np.zeros((3,3))
    for i in range(0, 3):
        r_obs_ec[:,i] = R @ r_obs_eq[:,i]

    return r_obs_ec

def create_da_los_vectors(ra: Union[array, NDArray], dec: Union[array, NDArray]) -> Union[array, NDArray]:
    """
    Calculate the line of sight unit vectors for the given right ascension and declination.
    Run this when the Nomial solution of RA and DEC is known.
    Run again if Nomial solution changes.

    N represents the number of observations, and each observation is represented by a pair of right ascension (RA) and declination (DEC).
    :param ra:
        1xN array of right ascension angles (radians) in the range [0; 2pi].
    :param dec:
        1xN array of declination angles (radians) in the range [-pi/2, pi/2].
    :return:
        3XN array of LOS vectors. column strucutre: [ρ̂x, ρ̂y, ρ̂z]ᵢ. Row index corresponds to observation instance, and it repreats for N observations
    """
    assert ra.shape == dec.shape, "Number of RA and DEC observations must match"

    num_obs = len(ra)
    los_vectors = np.empty((3, num_obs), dtype=object)

    #DA section
    if isinstance(ra, array):
        for i in range(num_obs):
            los_vectors[:,i] = array([
                op.cos(dec[i]) * op.cos(ra[i]),
                op.cos(dec[i]) * op.sin(ra[i]),
                op.sin(dec[i])
                ])

    #ndarray Section
    if isinstance(ra[0],float):
        for i in range(num_obs):
            los_vectors[:,i] = np.array([
                op.cos(dec[i]) * op.cos(ra[i]),
                op.cos(dec[i]) * op.sin(ra[i]),
                op.sin(dec[i])
                ])
    
    else:
        print("Input Error")

    return los_vectors

def get_earth_ephemeris(years, months, days, uts):
    # Create a Time object for each observation
    times = [Time(f"{year}-{month}-{day} {ut[0]:02d}:{ut[1]:02d}:{ut[2]:02d}", scale='utc') for year, month, day, ut in zip(years, months, days, uts)]

    # Get the ephemeris of Earth for each time
    earth_ephemeris = [pl.Earth().get_ephemeris(time) for time in times]

    return earth_ephemeris

def CC2MEE(r: NDArray, v: NDArray) -> NDArray:
    """
    Function for converting Cartesian coordinates to Modified Equinoctial Elements (MEE).
    DA compatible version
    """
    assert r.shape == v.shape, "Position and velocity vectors must have the same shape"
 

    return

def CC2COE(r: Union[array, NDArray], v: Union[array, NDArray], mu) -> Union[NDArray,array]:
    """
    Function for converting Cartesian coordinates to Classical Orbital Elements (COE).
    DA compatible version
    params:
    r: Position vector [i,j,k] around the center of mass of the body defined by mu
    v: Velocity vector [i,j,k] around the center of mass of the body defined by mu
    mu: Gravitational parameter of the central body
    :return: 
    Classical Orbital Elements (COE) in the form of a 6x1 array [semi-major axis, eccentricity, inclination, right ascension of the ascending node, argument of perigee, true anomaly]
    """
    assert r.shape == v.shape, "Position and velocity vectors must have the same shape"
    assert r.shape[0] == 3, "Position vector must be a 3D vector"
    assert v.shape[0] == 3, "Velocity vector must be a 3D vector"
    assert isinstance(r, (array, NDArray)), "Position vector must be a DA array or numpy array"
    
    if isinstance(r, array):
        r_norm = r.vnorm()
        v_norm = v.vnorm()
        iK = array([0, 0, 1])

        v_radial = r.dot(v)/r_norm 
        if v_radial.cons() > 0:
            print("Satellite Flying away from Perigee")
        elif v_radial.cons() < 0:
            print("Satellite is moving towards perigee")
        else:
            print("Radial Velcoity is Zero")
            raise ValueError
        
        h = r.cross(v)
        h_norm = h.vnorm() 
        h_z = h[2]

        ecc_vec = 1/mu * ((v_norm**2 - mu/r_norm) * r - (r*v_radial) * v)
        ecc = ecc_vec.vnorm()

        specfic_energy = v_norm**2/2 - mu/r
        if ecc.cons() != 1:
            a = -mu/(2*specfic_energy)
            p = a*(1-ecc**2)
        else:
            p = h_norm**2/mu
            a = np.inf
        

        N = iK.cross(h)    #Node line
        N_norm = N.vnorm()  #Node line
        N_i = N[0]
        N_j = N[1]
        if N_j.cons() >= 0:
            RAAN = op.acos(N_i/N_norm)
        else: 
            RAAN = 2*np.pi - op.acos(N_i/N_norm)


        inc = op.acos(h_z/h_norm)
        
        om = op.acos(N.dot(ecc_vec) / (N_norm * ecc))
        if ecc_vec[2].cons() < 0:
            om = 2*pi - om

        TA = op.acos( (ecc_vec.dot(r))/ (ecc * r_norm) )
        if v_radial.cons() < 0:
            TA = 2*pi - TA

        COE = [a, ecc, inc, RAAN, om, TA]
        return COE
    
    elif isinstance(r, NDArray):
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        iK = np.array([0, 0, 1])

        v_radial = np.dot(r, v) / r_norm 
        if v_radial > 0:
            print("Satellite Flying away from Perigee")
        elif v_radial < 0:
            print("Satellite is moving towards perigee")
        else:
            print("Radial Velcoity is Zero")
            raise ValueError
        
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h) 
        h_z = h[2]

        ecc_vec = (1/mu) * ((v_norm**2 - mu/r_norm) * r - (r*v_radial) * v)
        ecc = np.linalg.norm(ecc_vec)

        specfic_energy = v_norm**2/2 - mu/r
        if ecc != 1:
            a = -mu/(2*specfic_energy)
            p = a*(1-ecc**2)
        else:
            p = h_norm**2/mu
            a = np.inf
        
        N = np.cross(iK, h)
        N_norm = np.linalg.norm(N)
        N_i = N[0]
        N_j = N[1]
        if N_j >= 0:
            RAAN = np.arccos(N_i/N_norm)
        else:
            RAAN = 2*np.pi - np.arccos(N_i/N_norm)
        inc = np.arccos(h_z/h_norm)
        om = np.arccos(np.dot(N, ecc_vec) / (N_norm * ecc))
        if ecc_vec[2] < 0:
            om = 2*np.pi - om
        TA = np.arccos( (np.dot(ecc_vec, r))/ (ecc * r_norm) )
        if v_radial < 0:
            TA = 2*np.pi - TA

        COE = [a, ecc, inc, RAAN, om, TA]
        return COE


def COE2CC(COE, mu):
    """
    Function for converting Classical Orbital Elements (COE) to Cartesian coordinates.
    DA compatible version
    params:
    COE: Classical Orbital Elements (COE) in the form of a 6x1 array [semi-major axis, eccentricity, inclination, right ascension of the ascending node, argument of perigee, true anomaly]
    mu: Gravitational parameter of the central body
    :return: 
    Position vector [i,j,k] and velocity vector [i,j,k] around the center of mass of the body defined by mu
    """

    a, ecc, inc, RAAN, om, TA = COE

    if ecc < 1.0:

    # Compute the semi-latus rectum
        p = a * (1 - ecc**2)
    elif ecc > 1.0:
        p = a * (ecc**2 -1)
    else:
        raise ValueError("Parabolic case (e=1) unavailable")

    if p <= 0:
        raise ValueError("Computed p <= 0; check a and e consistency.")
    

    if isinstance(COE,array):
        r_PQW = p/(1+ecc*op.cos(TA)) * array([op.cos(TA), op.sin(TA), 0])
        v_PQW = array([-op.sqrt(mu/p) * op.sin(TA), (op.sqrt(mu/p)*(ecc + op.cos(TA))), 0])

        ROT = ROT3(om) @ ROT1(inc) @ ROT3(RAAN)
        
        r_ijk = ROT @ r_PQW
        v_ijk = ROT @ v_PQW

    return r_ijk, v_ijk

def ROT1(x):
    if isinstance(x, DA):
        ROT1 = array.zeros(3,3)
    else:
        ROT1 = np.zeros((3,3))

    ROT1[0,0] = 1
    ROT1[1,1] = op.cos(x)
    ROT1[2,1] = -op.sin(x)
    ROT1[1,2] = op.sin(x)
    ROT1[2,2] = op.cos(x)
    
    return ROT1

def ROT3(x):
    if isinstance(x, DA):
        ROT3 = array.zeros(3,3)
    else:
        ROT3 = np.zeros((3,3))
    
    ROT3[0,0] = op.cos(x)
    ROT3[1,0] = -op.sin(x)
    ROT3[0,1] = op.sin(x)
    ROT3[1,1] = op.cos(x)
    ROT3[2,2] = 1
    
    return ROT3