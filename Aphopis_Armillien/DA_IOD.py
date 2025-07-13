from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Union, overload, Tuple

import time
import numpy as np
from Tools.demo.sortvisu import steps
from daceypy import DA, array, ADS
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

# TODO 
#   0. Finish the DAIOD Algorithm so that the second lambert arc section is calculated correctly done
#   1. Implement a Method to convert between RA and DEC errors to Helocentric coordinate errors. RA_DEC_2_Helocentric()
#   1b. Create a Carteisan Helocentric -> Earth RA_DEC_Range function (to assess the range,range rate phase space)
#   2. Ensure 3 sigma scaling is implemented correctly. done
#   3. ADS for the inital Domain
#   4. ADS for the propagated Domain (done)

"""
    This file contains the implementation of the DA_IOD algorithm without Automatic Domain splitting. 
    See "INITIAL ORBIT DETERMINATION BASED ON PROPAGATION OF ORBIT SETS WITH DIFFERENTIAL ALGEBRA".
"""


#------------------------------------------------------------------------------------------------------
# Helper Functions 
#------------------------------------------------------------------------------------------------------


def RK78(Y0: array, X0: float, X1: float, f: Callable[[array, float], array]) -> array:
    """
    Propagate using RK78.
    """
    Y0 = Y0.copy()

    N = len(Y0)

    H0 = 0.001
    HS = 0.1
    H1 = 100.0
    EPS = 1.e-12
    BS = 20 * EPS

    Z = array.zeros((N, 16))
    Y1 = array.zeros(N)

    VIHMAX = 0.0

    HSQR = 1.0 / 9.0
    A = np.zeros(13)
    B = np.zeros((13, 12))
    C = np.zeros(13)
    D = np.zeros(13)

    A = np.array([
        0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0,
        93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0,
        1201146811.0/1299019798.0, 1.0, 1.0,
    ])

    B[1, 0] = 1.0/18.0
    B[2, 0] = 1.0/48.0
    B[2, 1] = 1.0/16.0
    B[3, 0] = 1.0/32.0
    B[3, 2] = 3.0/32.0
    B[4, 0] = 5.0/16.0
    B[4, 2] = -75.0/64.0
    B[4, 3] = 75.0/64.0
    B[5, 0] = 3.0/80.0
    B[5, 3] = 3.0/16.0
    B[5, 4] = 3.0/20.0
    B[6, 0] = 29443841.0/614563906.0
    B[6, 3] = 77736538.0/692538347.0
    B[6, 4] = -28693883.0/1125000000.0
    B[6, 5] = 23124283.0/1800000000.0
    B[7, 0] = 16016141.0/946692911.0
    B[7, 3] = 61564180.0/158732637.0
    B[7, 4] = 22789713.0/633445777.0
    B[7, 5] = 545815736.0/2771057229.0
    B[7, 6] = -180193667.0/1043307555.0
    B[8, 0] = 39632708.0/573591083.0
    B[8, 3] = -433636366.0/683701615.0
    B[8, 4] = -421739975.0/2616292301.0
    B[8, 5] = 100302831.0/723423059.0
    B[8, 6] = 790204164.0/839813087.0
    B[8, 7] = 800635310.0/3783071287.0
    B[9, 0] = 246121993.0/1340847787.0
    B[9, 3] = -37695042795.0/15268766246.0
    B[9, 4] = -309121744.0/1061227803.0
    B[9, 5] = -12992083.0/490766935.0
    B[9, 6] = 6005943493.0/2108947869.0
    B[9, 7] = 393006217.0/1396673457.0
    B[9, 8] = 123872331.0/1001029789.0
    B[10, 0] = -1028468189.0/846180014.0
    B[10, 3] = 8478235783.0/508512852.0
    B[10, 4] = 1311729495.0/1432422823.0
    B[10, 5] = -10304129995.0/1701304382.0
    B[10, 6] = -48777925059.0/3047939560.0
    B[10, 7] = 15336726248.0/1032824649.0
    B[10, 8] = -45442868181.0/3398467696.0
    B[10, 9] = 3065993473.0/597172653.0
    B[11, 0] = 185892177.0/718116043.0
    B[11, 3] = -3185094517.0/667107341.0
    B[11, 4] = -477755414.0/1098053517.0
    B[11, 5] = -703635378.0/230739211.0
    B[11, 6] = 5731566787.0/1027545527.0
    B[11, 7] = 5232866602.0/850066563.0
    B[11, 8] = -4093664535.0/808688257.0
    B[11, 9] = 3962137247.0/1805957418.0
    B[11, 10] = 65686358.0/487910083.0
    B[12, 0] = 403863854.0/491063109.0
    B[12, 3] = - 5068492393.0/434740067.0
    B[12, 4] = -411421997.0/543043805.0
    B[12, 5] = 652783627.0/914296604.0
    B[12, 6] = 11173962825.0/925320556.0
    B[12, 7] = -13158990841.0/6184727034.0
    B[12, 8] = 3936647629.0/1978049680.0
    B[12, 9] = -160528059.0/685178525.0
    B[12, 10] = 248638103.0/1413531060.0

    C = np.array([
        14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0,
        181606767.0/758867731.0, 561292985.0/797845732.0,
        -1041891430.0/1371343529.0, 760417239.0/1151165299.0,
        118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0,
    ])

    D = np.array([
        13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0, -808719846.0/976000145.0,
        1757004468.0/5645159321.0, 656045339.0/265891186.0,
        -3867574721.0/1518517206.0, 465885868.0/322736535.0,
        53011238.0/667516719.0, 2.0/45.0, 0.0,
    ])

    Z[:, 0] = Y0

    H = abs(HS)
    HH0 = abs(H0)
    HH1 = abs(H1)
    X = X0
    RFNORM = 0.0
    ERREST = 0.0

    while X != X1:

        # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):

            for i in range(N):

                Y0[i] = 0.0
                # EVALUATE RHS AT 13 POINTS
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]

                Y0[i] = H * Y0[i] + Z[i, 0]

            Y1 = f(Y0, X + H * A[j])

            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):

            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            # EXECUTE 7TH,8TH ORDER STEPS
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()

        # ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM

    Y1 = Z[:, 0]

    return Y1

def TBP(x: array, t: float) -> array:
    """
    2D Two Body Problem dynamics.
    """
    pos: array = x[:2]
    vel: array = x[2:]
    r = pos.vnorm()
    acc: array = -mu * pos / (r ** 3)
    dx = vel.concat(acc)
    return dx


# evaluation functions for ADS
def base_propagation(domain_0: ADS, t0: float, tf: float) -> ADS:
    """
    Base ADS propagation function.
    Each subdomain is propagated from its original DA box.
    reset approach - Alwyas do domain split from the beginning domain.
    """
    x0 = domain_0.box
    xf = RK78(x0, t0, tf, TBP)                      #Propagate from 0 to i+1
    return ADS(domain_0.box, domain_0.nsplit, xf)


def advanced_propagation(domain_0: ADS, t0: float, tf: float) -> ADS:
    """
    Advanced ADS propagation function.
    The Previous maps, and their sub domains are carried forward and evaluated
    """
    x0 = domain_0.manifold
    xf = RK78(x0, t0, tf, TBP)                      #Propagate from i to i+1
    return ADS(domain_0.box, domain_0.nsplit, xf)

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

    if not isinstance(RA, array) and not isinstance(DEC, array): #Case 1: DAIOD with independant Range Variables
        assert(isinstance(range_mag, array)), "Range Vector is not Initialised in DA"
        range_mag0 = range_mag
        flag = True
        iter = 1

        while flag:
            tol = 1e-8
            iter += 1
            range_mag = iod.newton_nomial_DAVec(range_mag0, np.zeros_like(range_mag0), f, 1e-16, 100) 

            #Evalute the Polynomial at DV = 0 
            difference = (range_mag - range_mag0).vnorm()
            if difference.cons() < tol:
                print(f"Iteration Number: {iter}\n")
                range_mag_L1_Jac = range_mag_L1.linear()      #extract jacobian part
                range_mag_L1_csnt = range_mag_L1.cons()       #extract constant
                flag = False
        
            range_mag0 = range_mag_L1
        
    #Post Process range vectors
    i_rho = time_reference.create_da_los_vectors(RA, DEC)
    range_vec = op.dot(range_mag, i_rho)


    if isinstance(RA, array) and isinstance(DEC, array):        #Case 2: DAIOD with Range DA Variables dependant on angle DA.
        assert(isinstance(range_mag, array)), "Range Vector is not Initialised in DA"
        
        i_rho = time_reference.create_da_los_vectors(RA, DEC)          #returns 3x3 matrix   
                      #Line of sight unit vector
        range_vec0 = op.dot(range_mag.cons(), i_rho)                         #Taylor Polynomial Map
        range_vec = iod.Implicit_solver_DAVec(range_vec0, 0, f1, 6, x0DA=False, DAIOD=True, JacDAIOD=op.dot(range_mag.linear(), i_rho.cons()))

    r_vec = np.zeros_like(range_vec)
    for i in range(0, len(range_vec)):
        r_vec[:,i] = range_vec[:,i] + r_obs_heliocentric[:,i]

    velocities = []
    velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, cw=False)
    solution = velocities[0]

    v1 = solution[:,0]
    v2 = solution[:,1]

    print(f"v1: {v1.cons()}\n")                 #debugging purposes
    print(f"v2: {v2.cons()}\n")                 #debugging purposes

    rv = array([r_vec[:,1], v2])               #Define Epoch state of arc - DA parts correspond to observation uncertanity 

    return rv, range_mag

def DAIOD_ADS(domain: ADS, RA, DEC, range_mag, r_obs_heliocentric, t_obs_s, mu):

    rv, _ = DAIOD(RA, DEC, range_mag, r_obs_heliocentric, t_obs_s, mu)
    #obtain range rate as a function of DA(range)

    #domain is [range,rangerate]
    return ADS(domain.box, domain.nsplit, rv)

# ------------------------------------------------------------------------------------------------------
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

#Telescope characteristics - Input as arcsecs
three_sigmasRA = np.array([1, 1, 1]) * u.arcsec
three_sigmasDA = np.array([1, 1, 1]) * u.arcsec

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

r_obs_earth = time_reference.equatorial_to_eclipitcJ2000(dr, LST_deg.to(u.rad).value, h_e)         #placeholder values

r_obs_heliocentric = earth_pos_helio + r_obs_earth                                                 #observation position from sun in heliocentric frame
#----------------------------------------------------------------------------------------
## Start DA_IOD
#----------------------------------------------------------------------------------------

#Generate Guass seed DA_IOD part 1

i_rho = time_reference.create_da_los_vectors(RA_deg.to(u.rad).value, DEC_deg.to(u.rad).value)          #returns 3x3 matrix                         #Line of sight unit vector
_, _, range_mag = iod.Guass_8th_seed(r_obs_heliocentric, i_rho, t_obs_s, mu)   #Obtain nomial states from Guass_8th_seed function (3x1)

DA.init(order, 3)
range_mag = array([range_mag[0] + DA(1), range_mag[1] + DA(2), range_mag[2] + DA(3)])   # range is 'x' in the implicit solver, move the DA initialisation inside the solver

#Lambert Arc part 1 - DAIOD part 2
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

flag = True
range_mag0 = range_mag
iter = 1

while flag:
    tol = 1e-8
    iter += 1
    range_mag_L1 = iod.newton_nomial_DAVec(range_mag0, np.zeros_like(range_mag), f, 1e-16, 100) 

    #Evalute the Polynomial at DV = 0 

    difference = (range_mag_L1 - range_mag0).vnorm()
    if difference.cons() < tol:
        print(f"Iteration Number: {iter}\n")
        range_mag_L1_Jac = range_mag_L1.linear()      #extract jacobian part
        range_mag_L1_csnt = range_mag_L1.cons()       #extract constant
        flag = False


#Lambert Arc part 2 - Now correct range has been obtained, find state in terms of observations
DA.init(order, 6)

RA_rad_DA = array([RA_deg[i].to(u.rad).value + three_sigmasRA[i].to(u.rad).value*DA(i + 1) for i in range(3)])
DEC_rad_DA = array([DEC_deg[i].to(u.rad).value + three_sigmasDA[i].to(u.rad).value*DA(i + 4) for i in range(3)])
# Scale RA and DEC DA part so that when evaluated its within the [-1,1] range


i_rho = time_reference.create_da_los_vectors(RA_rad_DA, DEC_rad_DA)          #returns 3x3 matrix   
                      #Line of sight unit vector
range_vec = op.dot(range_mag_L1_csnt, i_rho)                         #Taylor Polynomial Map

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

range_vec_anglesDA = iod.Implicit_solver_DAVec(range_vec, 0, f1, 6, x0DA=False, DAIOD=True, JacDAIOD=range_mag_L1_Jac)

r_vec = np.zeros_like(range_vec)
for i in range(0, len(range_vec)):
    r_vec[:,i] = range_vec_anglesDA[:,i] + r_obs_heliocentric[:,i]

velocities = []

velocities = lambert_izzo(r_vec[:,0], r_vec[:,1], t_obs_s[1] - t_obs_s[0], mu, 0, cw=False)
solution = velocities[0]

v1 = solution[:,0]
v2 = solution[:,1]

print(f"v1: {v1.cons()}\n")                 #debugging purposes
print(f"v2: {v2.cons()}\n")                 #debugging purposes

X_0 = array([r_vec[:,1], v2])               #Define Epoch state of arc - DA parts correspond to observation uncertanity 

#----------------------------------------------------------------------------------------
## End DA_IOD
#----------------------------------------------------------------------------------------
## ADS check of Initial Domain
#----------------------------------------------------------------------------------------

#Domain: [range_min, range_max] x [rangerate_min, rangerate_max]
#Initial domain is obtained through converting rv (DA=angles) -> range and range rate 

toll = 1e-5
Nmax = 10
init_domain = ADS(X_0, [])
init_list = [init_domain]
final_lists = []
final_list = init_list.copy()
final_lists.append(final_list) # add also initial domains

final_list = ADS.eval(
        final_list, toll, Nmax,
        DAIOD)












# ------------------------------------------------------------------------------
# ADS Propagation
# ------------------------------------------------------------------------------
 
# Step 1: Define uncertanity perimeter of the domain

TF = 50.                 #Final time
T0 = 0.                  #Initial time
Ns = 33                  #Number of gridpoints        
Ts = 51                  #Number of timesteps

# step 1a: convert the RA and DEC uncertanity to a Helocentric coordinate for the uncertanity 'domian'.

xb = RA_DEC_2_Helocentric(three_sigmasRA, three_sigmasDA)
yb = RA_DEC_2_Helocentric(three_sigmasRA, three_sigmasDA)

# step 1b: generate the grid perimeter of the initial domain 

tgrid = np.linspace(T0, TF, Ts)     #Generate time grid
xgrid = np.linspace(-1, 1, Ns)      #Generate x grid [-1, 1] is a must
ygrid = np.linspace(-1, 1, Ns)      #Generate y grid [-1, 1] is a must

lb = np.ones((Ns,2))                #Left boundary of uncertanity set
lb[:, 0] = lb[:,0]*(-xb)
lb[:, 1] = yb*ygrid

rb = np.ones((Ns,2))                #Right boundary of uncertanity set
rb[:, 0] = rb[:,0]*(xb)
rb[:, 1] = yb*ygrid

bb = np.ones((Ns,2))                #Bottom boundary of uncertanity set
bb[:, 0] = xgrid
bb[:, 1] = bb[:,1]*(-yb)

tb = np.ones((Ns,2))                #Top boundary of uncertanity set
tb[:, 0] = xgrid
tb[:, 1] = tb[:,1]*(yb)

# Define the perimeter of the entire initial domain
perimeter = np.concatenate((tb, rb, bb, lb))
perimeter_norm = np.concatenate((tb, rb, bb, lb))

perimeter_norm[:, 0] = perimeter[:,0]/xb
perimeter_norm[:, 1] = perimeter[:,1]/yb


init_domain = ADS(X_0, [])
toll = 1e-4
Nmax = 100

init_list = [init_domain]
final_lists = []
final_list = init_list.copy()
final_lists.append(final_list) # add also initial domains

start_advanced = time.time()
for i in range(Ts - 1):
    final_list = ADS.eval(
        final_list, toll, Nmax,
        lambda domain: advanced_propagation(domain, t0=tgrid[i], tf=tgrid[i+1]))
    final_lists.append(final_list)

    print('time ', tgrid[i+1], 'reached!')

print('execution time advanced ADS: ', time.time() - start_advanced)






