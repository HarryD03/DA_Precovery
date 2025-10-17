## Import Packages

from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Union, overload

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
RA = np.array([180, 180, 180]) # placeholder
DEC = np.array([0, 0, 0])      # placeholder


los_nomial = create_da_los_vectors(RA, DEC)
obs_time = np.arra

# 1) First-Guess Orbit: Classic Guass method (Right Ascension, Declination, t1->t3). Solve for middle range 'epoch'

# Initialise/Pre-condition observation times


def Guass_8th_seed(pos_obs: Union[NDArray, array], obs_dir: Union[NDArray, array], t: NDArray, mu: float) -> NDArray[np.double]:
    #Taken from Orbital Mechanics for Engineering Students (4th ed.) by Curtis. p.242 Algorithm 5.5. and Armillien Aphopis
    #Inputs:
    #   Pos_obs: 1D array of Observer positions in Heliocentric frame of reference from angle rotation matrix
    #   t: 1D array of times in seconds
    #   obs_dir: 1D array of unit vectors pointing from observer to the point of interest
    # 2-BP assumption:
    #   Assume the observations lie on the same plane

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



# 2) Define DA variables for observations and ranges
DA.init(5,9)
RA_da = array([RA[0] + DA(1), (RA[1]) + DA(2), (RA[2]) + DA(3)])
DEC_da = array([DEC[0] + DA(4), DEC[1] + DA(5), DEC[2] + DA(6)])
range_da = array([range[0] + DA(7), range[1] + DA(8), range[2] + DA(9)])        #obtain nomial states from Guass_8th_seed function

def create_da_los_vectors(ra: Union[array, NDArray], dec: Union[array, NDArray]) -> Union[array, NDArray]:
    """
    Calculate the line of sight unit vectors for the given right ascension and declination.
    Run this when the Nomial solution of RA and DEC is known.
    Run again if Nomial solution changes.
    :param ra:
        1xN array of right ascension angles (radians) in the range [0, 2pi].
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

r2_DA = position[1] + r2Map
r3_DA = position[2] + r3Map


# 3) Forward Propagate point 2 to point 3 Using Battin's scalar implicit equation: -> Obtain taylor map at v2 as a function of observations
def lambert_battin()
#       a) Once numerically -> DA root/x^0 term
#       b) Expand a nomial solution wrt 6 error variables -> build a DA map



# 4) Kepler Back Propagation: t2 -> t1 -> obtain predicted angles of right ascension and declination for point 1
# 5) Enforce the re-computed point 1 angles ,catch the measuremts and refine ranges:
#       a) From residual maps between computed and observed and progress to 0
#       b) invert maps to obtain updated delta-ranges.
#       c) repeat till observation error -> 0.

# Outcome: Taylor series [r2,v2] as a function of 6 observation errors
#       a) reproduce Guass at 0 error through DA
#       b) evalate

#Methodology: DA-based OD algorithm evaluation (propagate to point 4) and observe error between DA and pointwise evaluation -> taken from aphosis close encounter paper.

# 1) Obtain [r2,v2] as a function of 6 observation errors through DA
# 2) Define dynamics model
# 3) Inegrate once with a DA numerical integration scheme
#       a) DA-RK78 integrator of taylor expansion
#       b) Extract state at desired epoch
# 4) Verify if Domain splitting required (optional)
#       a) evaluate truncation error and conduct ADS???
# 5) Invert taylor expansion to obtain a polynomial map

