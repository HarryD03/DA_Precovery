import numpy as np
import pytest
from utils.iod import Guass_8th_seed, f_g_series, DAIOD_1, DAIOD_2
import numpy as np
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray
from utils.lambert_izzo import lambert_izzo, householder_iter_DA_nom, householder_iter_DA_Map, x2tof
from utils.poli_izzo import izzo
from utils.iod import newton_nominal_DAVec, Implicit_solver_DAVec, newton_nomial_DA, Implicit_solver_DA
import pickle 
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


def test_householder_map(): #see if Householder function is correct
    """
        Newton is known to be correct
    """
    root = 2.094551481542327

    def f(x, p):
        return x**3 - 2*x - p
    
    p0 = 5
    x0 = 5
    order = 5
    MaxIter = 30 
    tol = 1e-9
    #Conduct Newton For x_nom
    DA.init(4,1)
    
    x_nom_newton = newton_nomial_DA(x0, p0, f, tol, MaxIter, order)

    #Conduct householder for x_nom 
    x_nom_householder = householder_iter_DA_nom(x0, p0, f, tol, MaxIter)
    
    DA.init(4,2)
    p = p0 + DA(1)
    x_DA_newton = Implicit_solver_DA(x_nom_newton, p, f)
    x_DA_householder = householder_iter_DA_Map(x_nom_householder, p, DA.getMaxVariables(), f, tol, MaxIter)

    error_nom = np.abs(x_nom_householder - x_nom_newton)
    error_linear = np.linalg.norm(x_DA_householder.linear() - x_DA_newton.linear())

    print("Checking if Newton is correct")
    assert np.isclose(x_nom_newton, root, rtol= 1e-12), "Newton must be identical to true root"

    assert error_nom < 1e-12, "Zeroth Order solution must be idenitical" 
    assert error_linear < 1e-9, "Linear order solution must be < 1e-9, Appprox Identical "
    
def test_householder_DAIOD(): #See if householder function is correct within the DAIOD environmnet.

    
    DA.init(4, 3)
    #Get Input variables 
    with open('householder_params.pkl', 'rb') as P:
        p_da = pickle.load(P)

    T_da = p_da[0]
    L_da = p_da[1] 
    M = p_da[2]

    def f(x,p):
        """
        Function to obtain f(x) = T(x) - T* for Householder Iteration scheme
        """
        T, L, M = p
        return x2tof(x, M, L) - T

    # Extract float parameters for nominal solutions
    p0 = [T_da.cons(), L_da.cons(), M]
    x0 = 0.7852417534231431 # initial guess
    order = 5
    MaxIter = 30 
    tol = 1e-9
    #Conduct Newton For x_nom
    # Conduct Householder for x_nom 
    x_nom_householder = householder_iter_DA_nom(x0, p0, f, tol, MaxIter)
    x_nom_newton = newton_nomial_DA(x0, p0, f, tol, MaxIter, order)
    #Create Taylor Maps - Use the DA parameters consistently
    x_DA_housholder = householder_iter_DA_Map(x_nom_householder, p_da, DA.getMaxVariables(), f, tol, MaxIter)
    x_DA_newton = Implicit_solver_DA(x_nom_newton, p_da, f)

    #Error observation 
    error_nom = np.abs(x_nom_householder - x_nom_newton)
    error_linear = np.linalg.norm(x_DA_housholder.linear() - x_DA_newton.linear())

    assert error_nom < 1e-12, "Zeroth Order solution must be idenitical" 
    assert error_linear < 1e-9, "Linear order solution must be < 1e-9, Appprox Identical "
     

    
     