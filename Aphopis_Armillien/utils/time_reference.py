import numpy as np
from typing import Callable, List, Union, overload, Tuple
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray

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

    r_obs_ec = np.dot(R, r_obs_eq)
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