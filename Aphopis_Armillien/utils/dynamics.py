import numpy as np
from typing import Callable, List, Union, overload, Tuple
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray
import poliastro as pl
from astropy.time import Time


def TBP_CC(rv: array, mu:float, t: float) -> array:
    """
    Two Body Problem in Cartesian Coordiantes
    No Perturbations
    DA only
    """
    
    pos: array = rv[:3]
    vel: array = rv[3:]
    r = pos.vnorm()         #obtain the euclidien norm (distance of the vector)
    acc = -mu * pos / (r**3)
    drv = vel.concat(acc)    #combines the velocity vector and acceleration vector into one vector

    return drv


def TBP_MEE_FP(MEE, mu, P, t):
    """
    Two Body Problem Dynamics in MEE reference frame
    :params
    MEE: Modified Equinotial Elements 
    P: Perturbations in RTN
    :returns
    dMEE: Derivative of MEE
    """
    p, f, g, h, k, L = MEE

    A = np.zeros((6,3))
    w = 1 + f*np.cos(L) + g*np.sin(L)
    s_sqr = 1 + h **2 + k**2
    if np.linalg.norm(P) > 0:   #Tes    t if Perturbations exist if they do find A 
        #Calculate the Perturbation conversion Matrix 'A'
        A[1,0] = np.sqrt(p/mu) * np.sin(L)
        A[2,0] = -np.sqrt(p/mu) * np.cos(L)

        A[0,1] = 2*p/w * np.sqrt(p/mu)
        A[1,1] = np.sqrt(p/mu) * 1/w * ((w+1) * np.cos(L) + f)
        A[2,1] = np.sqrt(p/mu) * ((w+1) * np.sin(L) + g)
        
        A[1,3] = - np.sqrt(p/mu) * g/w *(h * np.sin(L) - k* np.cos(L))
        A[2,3] = np.sqrt(p/mu) * f/w * (h * np.sin(L) - k * np.cos(L))
        A[3,3] = np.sqrt(p/mu) * (s_sqr * np.cos(L)/ (2*w) )
        A[4,3] = np.sqrt(p/mu) * (s_sqr * np.sin(L) / (2*w) )
        A[5,3] = np.sqrt(p/mu) * (h * np.sin(L) - (k * np.cos(L)) ) 
    
    b = np.zeros_like(MEE)
    b[6] = np.sqrt(mu*p)*(w/p)**2

    dMEE = A @ P + b.T
    
    return dMEE


def TBP_MEE_DA(MEE, mu, P, t):
    """
    Two Body Problem Dynamics in MEE reference frame (DA compatible)
    :params
    MEE: Modified Equinoctial Elements (daceypy.array)
    P: Perturbations in RTN (daceypy.array)
    :returns
    dMEE: Derivative of MEE (daceypy.array)
    """
    from daceypy import array
    import daceypy.op as op

    p, f, g, h, k, L = MEE

    A = array.zeros((6, 3))
    w = 1 + f * op.cos(L) + g * op.sin(L)
    s_sqr = 1 + h ** 2 + k ** 2

    if P.vnorm() > 0:  # DA-compatible norm check
        # Calculate the Perturbation conversion Matrix 'A'
        A[1, 0] = op.sqrt(p / mu) * op.sin(L)
        A[2, 0] = -op.sqrt(p / mu) * op.cos(L)

        A[0, 1] = 2 * p / w * op.sqrt(p / mu)
        A[1, 1] = op.sqrt(p / mu) * (1 / w) * ((w + 1) * op.cos(L) + f)
        A[2, 1] = op.sqrt(p / mu) * ((w + 1) * op.sin(L) + g)

        A[1, 2] = -op.sqrt(p / mu) * g / w * (h * op.sin(L) - k * op.cos(L))
        A[2, 2] = op.sqrt(p / mu) * f / w * (h * op.sin(L) - k * op.cos(L))
        A[3, 2] = op.sqrt(p / mu) * (s_sqr * op.cos(L) / (2 * w))
        A[4, 2] = op.sqrt(p / mu) * (s_sqr * op.sin(L) / (2 * w))
        A[5, 2] = op.sqrt(p / mu) * (h * op.sin(L) - (k * op.cos(L)))

    b = array.zeros(6)
    b[5] = op.sqrt(mu * p) * (w / p) ** 2

    dMEE = A @ P + b

    return dMEE