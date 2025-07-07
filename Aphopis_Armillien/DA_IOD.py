from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Union, overload, Tuple

import numpy as np
from Tools.demo.sortvisu import steps
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from utils.lambert_izzo import lambert_izzo
from utils import time_reference
from utils import iod

import poliastro as pl
import astropy.time as at
import astropy.constants as ac
import astropy.coordinates as acoords
import astropy.units as u
from datetime import datetime
from time import perf_counter

#TODO:
#    1. Change Guass seed so returns magnitude of range vector and complete position computation outside of function done
#    2. Ensure the all helocentric positions are calculated from seperation observation sites done
#    3. Ensure Newton iterations can process 2D matrices correctly (rows = compoents, columns = instances) 


"""
    This file contains the implementation of the DA_IOD algorithm without Automatic Domain splitting. 
    See "INITIAL ORBIT DETERMINATION BASED ON PROPAGATION OF ORBIT SETS WITH DIFFERENTIAL ALGEBRA".
"""
#------------------------------------------------------------------------------------------------------
# Helper Functions 
#------------------------------------------------------------------------------------------------------

## Observer Values: Assume Equatorial topocentric Reference frame

#Observer Inputs 
RA_deg = np.array([180, 180, 180]) * u.deg         # Right Ascension of observation (measured east/west direction sky longitude) [0; 24] (hours, minutes, seconds)
DEC_deg = np.array([0, 0, 0]) * u.deg              # Declination of observation (measured north/south direction sky latitude) [-90; 90] (degrees, arcminutes, arcseconds)

obs_times = at.Time([
    at.Time('2003-11-08 12:00:00', scale='ut1'),
    at.Time('2003-11-08 14:00:00', scale='ut1'),
    at.Time('2003-11-08 15:00:00', scale='ut1')
])
#Observeratory Location
east_longitude_deg = 139.4170 * u.deg                 # placeholder East Longitude of observer//observity
h_e = 0.58805 #0.0                   # placeholder Height of observer//observity
dr = 0.807392                     # placeholder Perpendicular distance of the observation site from Earth's rotation axis

body = 'sun'
order = 4


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

mu = ac.G * ac.M_sun
mu = mu.to('km**3 / s**2').value

earth = acoords.get_body_barycentric('earth', obs_times)
sun = acoords.get_body_barycentric('sun', obs_times)

earth_pos_helio = earth - sun
earth_pos_helio = earth_pos_helio.xyz.to(u.km).value
print(earth_pos_helio)

# ----------------------------------------------------------------------------------------
# Convert Observer Values to Ecliptic Coordinates
# ----------------------------------------------------------------------------------------
LST = obs_times.sidereal_time('mean', longitude=east_longitude_deg)
LST_deg = LST.to(u.deg)

# Time of Observation (julian days)
hms = obs_times.ymdhms
hours = hms['hour']
minutes = hms['minute']
seconds = hms['second']
ut_hours = (hours + minutes/60 + seconds/3600) * u.hour
t_obs_J0 = (obs_times.jd + ut_hours.value / 24) * u.day      #Epoch of observation in julian days
t_obs_s = t_obs_J0.to(u.s).value

r_obs_earth = time_reference.equatorial_to_eclipitcJ2000(dr, LST_deg.to(u.rad).value, h_e)                                #placeholder values

r_obs_heliocentric = earth_pos_helio + r_obs_earth                                                 #observation position
#----------------------------------------------------------------------------------------
## Start DA_IOD
#----------------------------------------------------------------------------------------

#Generate Guass seed
i_rho = time_reference.create_da_los_vectors(RA_deg.to(u.rad).value, DEC_deg.to(u.rad).value)          #returns 3x3 matrix                         #Line of sight unit vector
_, _, range_mag = iod.Guass_8th_seed(r_obs_heliocentric, i_rho, t_obs_s, mu)   #Obtain nomial states from Guass_8th_seed function (3x1)

DA.init(order, 3)

range_mag = array([range_mag[0] + DA(1), range_mag[1] + DA(2), range_mag[2] + DA(3)])   # range is 'x' in the implicit solver, move the DA initialisation inside the solver

range_vec = np.zeros_like(i_rho)
for i in range(0, len(range_mag)):
    range_vec[:,i] = op.dot(range_mag[i], i_rho[:,i])

r_vec = np.zeros_like(range_vec)
for i in range(0, len(range_vec)):
    r_vec[:,i] = range_vec[:,i] + r_obs_heliocentric[:,i]

#Lambert Arc part 1
vel = []
for i in range(0, len(r_vec[0,:]) - 1):
    velocities = lambert_izzo(r_vec[i,:], r_vec[i+1,:], t_obs_s[i+1] - t_obs_s[i], mu, 0, cw=False)

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

def f(range_vec):                                       #deltV = residual + M(dranges)
    
    """
    f(range_vec) returns the velocity difference between the second and first velocity estimates
    for the central observation. The velocity difference is calculated by first calculating the
    positions of the observations using the range vector and the observer position. The positions
    are then used to calculate the velocities between each observation via the Izzo solution to
    Lambert's problem. The velocity difference is then calculated as the difference between the
    second and first velocity estimates. This function is used in the Newton method to find the
    root of the velocity difference.
    """
    r_vec = np.zeros_like(range_vec)
    for i in range(0, len(range_vec)):                  #define posiiton vector for lamber_izzo
        r_vec[:,i] = range_vec[:,i] + r_obs_heliocentric[:,i]

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

    return DV #DV = [dv_i, dv_j, dv_k] + M(dranges)

range_vec_L1 = iod.Implicit_solver_DAVec(range_vec, 0, f, 3)

#Lambert Arc part 2 - Now correct range has been obtained, find state in terms of observations
DA.init(order, 6)
RA_rad_DA = array([RA_deg[i].to(u.rad).value + DA(i + 1) for i in range(3)])
DEC_rad_DA = array([DEC_deg[i].to(u.rad).value + DA(i + 4) for i in range(3)])

i_rho = time_reference.create_da_los_vectors(RA_rad_DA, DEC_rad_DA)          #returns 3x3 matrix                         #Line of sight unit vector
range_vec = range_vec_L1



