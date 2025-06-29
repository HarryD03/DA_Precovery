import numpy as np
import pytest
from Aphopis_Armillien.utils.lambert_izzo import lambert_izzo, findxy, x2tof, x2tof2, hypergeometricF, householder_iter_DA_nom, householder_iter_DA_Map
from poliastro.iod import izzo
from daceypy import DA, array
import daceypy.op as op

# Example geometry from Vallado Example 5.2
R1 = np.array([15945.34, 0.0, 0.0])
R2 = np.array([-12214.83899, 10249.46731, 0.0])
DT = 76 * 60
MU = 1.32712440018e11

V1_REF = np.array([2.058925, 2.915965, 0.0])
V2_REF = np.array([-3.451565, 0.910315, 0.0])

def _compute_LT(r1, r2, dt, mu):
    c_vec = r2 - r1
    c = op.vnorm(c_vec)
    r1_mag = op.vnorm(r1)
    r2_mag = op.vnorm(r2)
    s = 0.5 * (r1_mag + r2_mag + c)
    L = op.sqrt(1 - c / s)
    T = op.sqrt((2 * mu) / (s ** 3)) * dt
    return L, T, s

@pytest.mark.basic
def test_lambert_izzo():
    """Compare against reference Vallado velocities for single revolution."""
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    v1, v2 = lambert_izzo(R1_DA, R2_DA, DT, MU, multi_revs=0)

    v1_num = np.array([float(v1[i][0]) if not hasattr(v1[i][0], "cons") else v1[i][0].cons() for i in range(3)])
    v2_num = np.array([float(v2[i][0]) if not hasattr(v2[i][0], "cons") else v2[i][0].cons() for i in range(3)])
   
    assert np.allclose(v1_num, V1_REF, rtol=1e-3)
    assert np.allclose(v2_num, V2_REF, rtol=1e-3)


@pytest.mark.basic
def test_findxy():
    """x,y from findxy should satisfy x2tof(x)=T for single rev."""
    L, T = _compute_LT(R1, R2, DT, MU)
    x_list, y_list = findxy(L, T, multi_revs=0)
    assert len(x_list) == 1 and len(y_list) == 1

    x = x_list[0]
    T_calc = x2tof(x, 0, L)
    assert np.isclose(float(T_calc), float(T), rtol=1e-6)


@pytest.mark.basic
def test_x2tof():
    """For small x, x2tof should agree with x2tof2"""
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    L, T, _ = _compute_LT(R1_DA, R2_DA, DT, MU)

    x = 0.9 + DA(NVar+1)

    T1 = x2tof(x, 0, L)
    T2 = x2tof2(x, 0, L)
    #Test the imbedded x2tof2 is called properly 
    assert np.isclose(T1.cons(), T2.cons(), rtol=1e-6)

@pytest.mark.basic
def test_x2tof_battin():
    """Test example for Battin solution section through Palistro"""

    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    L_DA, T_DA, s = _compute_LT(R1_DA, R2_DA, DT, MU)

    x0 = 0.999
    x = x0 + DA(NVar+1)

    y0 = op.sqrt(1+((L_DA.cons())**2)*(x0**2 - 1))
    T0 = T_DA.cons()
    T1 = x2tof(x,0,L_DA)            #my function
    Tref = izzo._tof_equation(x0,y0,T0,L_DA.cons(), 0)  #Palistro function
    Tref = Tref + T0                                    #So T are the same [see palistro function defintinon]
    
    #Test they're roughly equal
    assert np.isclose(T1.cons(), Tref, rtol=1e-6)

@pytest.mark.basic
def test_x2tof_general():
    """Test example for Battin solution section through Palistro"""

    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    L_DA, T_DA, s = _compute_LT(R1_DA, R2_DA, DT, MU)

    x0 = 0.5
    x = x0 + DA(NVar+1)

    y0 = op.sqrt(1+((L_DA.cons())**2)*(x0**2 - 1))
    T0 = T_DA.cons()
    T1 = x2tof(x,0,L_DA)            #my function
    Tref = izzo._tof_equation(x0,y0,T0,L_DA.cons(), 0)  #Palistro function
    Tref = Tref + T0                                    #So T are the same [see palistro function defintinon]
    
    #Test they're roughly equal
    assert np.isclose(T1.cons(), Tref, rtol=1e-6)






@pytest.mark.basic
def test_x2tof2():
    """Dummy test for x2tof2 (expand as needed)."""
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    L_DA, T_DA, s = _compute_LT(R1_DA, R2_DA, DT, MU)

    x0 = 0.9
    x = x0 + DA(NVar+1)
    y0 = op.sqrt(1+((L_DA.cons())**2)*(x0**2 - 1))
    T0 = T_DA.cons()

    T1 = x2tof2(x, 0, L_DA)
    T2 = izzo._tof_equation(x0,y0,T0,L_DA.cons(), 0)  #Palistro function
    T2 = T2 + T0
    #Test the imbedded x2tof2 is called properly 
    assert np.isclose(T1.cons(), T2, rtol=1e-6)


def _hypergeometricF_ref(S1, tol=1e-11):
    """
    Copy Paste from Reference PyKep
    """
    term = 1.0
    S = 1.0
    j = 0
    while abs(term) > tol:
        term *= (3.0 + j) * (1.0 + j) / ((2.5 + j) * (j + 1)) * S1
        S += term
        j += 1
    return S

@pytest.mark.basic
def test_hypergeometricF():
    """Numerical check against simple series implementation."""
    val = 0.05
    ref = izzo.hyp2f1b(val)
    calc = hypergeometricF(val)
    assert np.isclose(float(calc), ref, rtol=1e-8)

@pytest.mark.basic
def test_householder_iter_DA_nom():
    """Solve x^2=2 using Householder iterator."""
    DA.init(4, 1)
    def f(x):
        return x*x - 2
    x = householder_iter_DA_nom(1.0, f, 1)
    assert np.isclose(x.cons(), np.sqrt(2), atol=1e-6)

@pytest.mark.basic
def test_householder_iter_DA_Map():
    """Check mapping sensitivity for x^2=p around p=2."""
    DA.init(4, 2)
    p = 2 + DA(1)
    def f(x):
        return x*x - p
    
    x_nom = householder_iter_DA_nom(1.0, lambda xx: xx*xx - 2, 2)
    x_da = householder_iter_DA_Map(x_nom, 2, f)
    assert np.isclose(x_da.cons(), np.sqrt(2), atol=1e-6)
    # derivative dx/dp should be 1/(2*sqrt(2))
    deriv = x_da.deriv(1).cons()
    assert np.isclose(deriv, 1/(2*np.sqrt(2)), atol=1e-6)