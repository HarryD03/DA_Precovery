"""
This module contains utilities for the propagation of DAIOD (Differential Algebra Initial Orbit Determination) algorithms via Automatic Domain Splitting (ADS).
It includes the implementation of the DAIOD algorithm without automatic domain splitting.
"""

from typing import Callable
from daceypy import DA, array, ADS
from utils import dynamics
import numpy as np
from astropy import units as u
import scipy
from scipy.integrate import solve_ivp
import poliastro as pl
import functools
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

    direction = 1.0 if X1 > X0 else -1.0
    H = abs(HS) * direction
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
            H = HH1 * direction
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0 * direction
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * direction > 0:
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

def base_propagationADS(domain0: ADS, t0: float, tf: float, dynamics: Callable) -> ADS:
    """
    Base ADS propagation function.
    Each subdomain is propagated from its original DA box.
    reset approach - Alwyas do domain split from the beginning domain.

    :params
    domain0: Initial Domain 
    t0: Initial time
    tf: Propagtion time
    :returns
    ADS Object: (initial Domain, Number of splits, Propagated manifold)
    """
    x0 = domain0.box                                     #Initial State
    xf = RK78(x0, t0, tf, dynamics)                      #Propagate from 0 to i+1
    return ADS(domain0.box, domain0.nsplit, xf)

def advanced_propagationADS(domain0: ADS, t0: float, tf: float, dynamics: Callable, mu=3.986e5) -> ADS:
    """
    Advanced ADS propagation function.
    The Previous maps, and their sub domains are carried forward and evaluated

    :params
    domain0: Initial Domain 
    t0: Initial time
    tf: Propagtion time
    dynamics: Dynamics function for Propagation
    :returns
    ADS Object: (Root Domain, Number of splits, Propagated manifold)
    """
    x0 = domain0.manifold
                           #Get the manifold of the previous state
    xf = RK78(x0, t0, tf, dynamics)                     #Propagate from i to i+1

    return ADS(domain0.box, domain0.nsplit, xf)

def advanced_propagationDA(XI: array, tgrid, dynamics: Callable, mu=3.986e5):
    """
        Sequential Propagation of single domain defined by DAIOD

    params:
    XI: Initial State
    tgrid: 1D array of timesteps to propagate through
    dynamics: Propagation Dynamics
    returns:
    XFN: Propagated State at every Timestep Structure: (States x Time)
    """

    Ts = len(tgrid)
    XFN = array.zeros((6,Ts))
    XFN[:,0] = XI
    x0 = XI 
    for i in range(Ts-1):
        t0 = tgrid[i]
        tf = tgrid[i+1]
        xf = RK78(x0,t0, tf, dynamics)
        XFN[:,i+1] = xf.copy()

        x0 = xf 

    return XFN

def base_propagationPW(x0, t0, tf, dynamics, mu=3.986e5):
    """
        Propagates the Initial State through standard point-wise propagation
        :param x0: initial state
        :param t0: initial time
        :param tf: final time
        :param dynamics: dynamics function for propagation
        :param perimeter: 3D perimeter for propagation
    """
    xf = RK78_scipy(x0, t0, tf, dynamics)

    return xf


def _propagate_bounding_box_edges_facesPW(X0, box, t_span, dynamics):
    """
        Propagate the edges (2D) or faces (3D) of a 3D bounding Box
    
    :param X0: Nominal state vector [6,] array [x,y,z,vx,vy,vz]
    :param box: 3D bounding box for propagation [3,], array [x,y,z]
    :param t_span: Time span for propagation [2,], array [t0, tf]
    :param dynamics: Dynamics function for Propagation
    :returns: Propagated state vector
    """
    #convert relative coordinates to actual initial states
    n_perimeter_points = len(box)
    X0_boundary = np.zeros((n_perimeter_points, 6))

    #relative perturbtaion to Absolute conversion
    for i in range(n_perimeter_points):
        X0_boundary[i,:] = X0
        X0_boundary[i, 0:3] += box[i]  # Add the box coordinates to the nominal state

    XF_boundary = np.zeros_like(X0_boundary)
    successful_propagation = 0

    for i in range(n_perimeter_points):
        try:
            XF_boundary[i, :] = RK78_scipy(X0_boundary[i, :], t_span[0], t_span[1], dynamics)
            successful_propagation += 1

        except Exception as e:
            print(f"Error propagating boundary point {i}: {e}")
            XF_boundary[i, :] = np.nan  # Mark as NaN if propagation fails
    
    #Remove failed Propagations
    valid_mask = ~np.isnan(XF_boundary).any(axis=1)
    x0_valid = X0_boundary[valid_mask,:]
    xf_valid = XF_boundary[valid_mask,:]
    
    if successful_propagation == 0:
        raise RuntimeError("All boundary points failed to propagate.")
    
    #Compute final position bounding box
    XF_pos_box = xf_valid[:,0:3]
    XF_pos_nominal = RK78_scipy(X0, t_span[0], t_span[1], dynamics)[:3]  # Nominal position at final time

    return xf_valid, XF_pos_nominal

def RK78_scipy(X0: array, T0: float, TF: float, f: Callable[[array, float], array], 
               rtol: float = 1e-12, atol: float = 1e-14) -> array:
    """
    Propagate using SciPy's DOP853 integrator (8th order Runge-Kutta).
    
    :param X0: Initial state vector
    :param X0: Initial time
    :param X1: Final time
    :param f: Dynamics function (RHS of ODE system)
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :returns: Final state vector
    """
    
    # Define the RHS function for SciPy (note different argument order)
    def rhs_scipy(t, X0):
        return dynamics.TBP_CC_FP(X0, t=t)

    # Integrate using DOP853 (8th order Runge-Kutta)
    sol = solve_ivp(rhs_scipy, [T0, TF], X0, method='DOP853',
                    rtol=rtol, atol=atol, dense_output=False)
    
    if not sol.success:
        raise RuntimeError(f"SciPy integration failed: {sol.message}")
    
    return sol.y[:, -1]

