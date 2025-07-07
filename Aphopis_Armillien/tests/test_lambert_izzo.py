import numpy as np
import pytest
from Aphopis_Armillien.utils.lambert_izzo import lambert_izzo, findxy, x2tof, x2tof2, hypergeometricF, householder_iter_DA_nom, householder_iter_DA_Map
from Aphopis_Armillien.utils.iod import newton_nomial_DA, Implicit_solver_DA
from poliastro.iod import izzo
from astropy import units as u
from poliastro.bodies import Earth
from astropy.tests.helper import assert_quantity_allclose

from daceypy import DA, array
import daceypy.op as op

# Example geometry from Vallado Example 5.2
R1 = np.array([15945.34, 0.0, 0.0])
R2 = np.array([12214.83399, 10249.46731, 0.0])
DT = 76 * 60
MU = 3.986e5

k = Earth.k
r0 = [15945.34, 0.0, 0.0] * u.km
r = [12214.83399, 10249.46731, 0.0] * u.km
tof = 76.0 * u.min

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

@pytest.mark.parametrize("lambert", [izzo.lambert])
def test_lambert_izzo(lambert):
    """Compare against reference Vallado velocities for single revolution.
        NOTE: We do -(Variable) in DA. This means the taylor coefficents in DA change and effectly mirror the nomimal solution and the distrubution
        """
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)
    

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])
    M = 0
    
    velocities = lambert_izzo(R1_DA, R2_DA, DT, MU, M)
    
    v1test, v2test = next(lambert(k, r0, r, tof))


    solution1 = velocities[0]

    v1 = solution1[:,0]
    v2 = solution1[:,1]

    print(f"My Lambert v2: {v2.cons()}\n")
    print(f"Palistro Lambert v2: {v2test}\n")
    print(f"Expected v2: {V2_REF}")

    assert np.allclose(v1.cons(), V1_REF, rtol=1e-5)
    assert_quantity_allclose(v1.cons(), v1test, rtol=1e-5)
    assert_quantity_allclose(v1test, V1_REF, rtol=1e-5)
    
    assert np.allclose(v2.cons(), V2_REF, rtol=1e-3)


@pytest.mark.basic
def test_findxy():
    """x,y from findxy should satisfy x2tof(x)=T for single rev."""
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    
    
    L, T, _ = _compute_LT(R1_DA, R2_DA, DT, MU)
    M = 0 #Single rev case
    xy = findxy(L, T, M)    #[x,y]
    assert len(xy) == 1

    T_calc = x2tof(xy[0][0], 0, L)

    assert np.isclose(T_calc.cons(), T.cons(), rtol=1e-6)

    xy_palistro = izzo._find_xy(L.cons(), T.cons(), 0, 1000, 1e-6)
    print(xy_palistro)
    for x, y in xy_palistro:
        x = x
        y = y
    
    assert np.isclose(x, xy[0][0].cons(), rtol=1e-6)
    assert np.isclose(y, xy[0][1].cons(), rtol=1e-6)

@pytest.mark.basic
def test_findxy_multi():
    """x,y from findxy should satisfy x2tof(x)=T for multi rev."""
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    
    
    L, T, _ = _compute_LT(R1_DA, R2_DA, DT, MU)
    M = 2 #Single rev case

    xy = findxy(L, T, M)    #[x,y]
    assert len(xy) == 2     # [[x0l, y0l], [x0r, y0r]]
    print(f"length:\n{len(xy)}\n")
    print(f"{xy}")
    
    T_calc_x0r = x2tof(xy[1][0], M, L) 
    T_calc_x0l = x2tof(xy[0][0], M, L)

    #Clockwise seems to be default motion -> x0r
    assert np.isclose(T_calc_x0r.cons(), T.cons(), rtol=1e-6)

    xy_palistro = izzo._find_xy(L.cons(), T.cons(), 0, 1000, 1e-6)
    print(xy_palistro)
    
    for x, y in xy_palistro:

        i = 0
        x = x
        y = y

        assert np.isclose(x, xy[i][0].cons(), rtol=1e-6)
        assert np.isclose(y, xy[i][1].cons(), rtol=1e-6)

        i += 1

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
def test_householder_iter_DA_nom_Basic():
    """Solve x^2=2 using Householder iterator. Compare solution with op.root()"""
    DA.init(4, 1)
    def f(x):
        return x*x - 2
    
    
    x = householder_iter_DA_nom(1.0, f, 1)
    assert np.isclose(x, np.sqrt(2), rtol=1e-6)

@pytest.mark.regression
def test_householder_iter_DA_nom():
    """
        Test nomial iteration scheme with poliastro householder
    """
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    L_DA, T_DA, s = _compute_LT(R1_DA, R2_DA, DT, MU)
    M=0

    def f(x):
        """
        Function to obtain f(x) = T(x) - T* for Householder Iteration scheme
        """
        return x2tof(x, M, L_DA) - T_DA
    
    T00 = op.acos(L_DA) + (L_DA*op.sqrt(1-L_DA**2))      
    T0 = T00 + M  * np.pi
    T1 = 2/3 * (1 - L_DA**3)         #Standard Time parameter if parabolic orbit (Max Energy transfer)

    #Generate initial coniditions
    if T_DA.cons() >= T0.cons():
        x0 = (T0/T_DA)**(2/3) - 1
    elif T_DA.cons() < T1.cons():
        x0 = (5/2) * (T1*(T1-T_DA))/(T_DA*(1-L_DA**5)) + 1
    elif T1.cons() < T_DA.cons() and T_DA.cons() < T0.cons():
        x0 = (T0/T_DA)**(op.log2(T1/T0)) - 1
    else:
        raise ValueError("Parameterised Time of Flight Parameter does not fall into any of the possible solutiions")
    

    x0 = x0.cons()      #x0 must be defined as an independant DA variable within the iteraiton loop 

    x_nom = householder_iter_DA_nom(x0, f, NVar + 1, tol=1e-9, MaxIter=1000)
    x = izzo._householder(x0, T_DA.cons(), L_DA.cons(), M, 1e-9, 1000)
    assert np.isclose(x_nom, x, rtol=1e-9)      #relaxed error bound due to numerical deriv vs analytical 

    #Compare to standard Newton_nomial to see numerical error small between solvers
    x_newton = newton_nomial_DA(x0,T_DA ,f,tol=1e-9, MaxIter=1000)
    assert np.isclose(x_nom, x_newton, rtol=1e-9)

@pytest.mark.basic
def test_householder_iter_DA_Map_basic():
    """Check mapping sensitivity for x^2=p around p=2."""
    DA.init(4, 2)
    
    p = 2 + DA(1)
    def f(x):
        return x*x - p
    
    x_nom = householder_iter_DA_nom(1.0, f, 2)
    print(x_nom)
    x_da = householder_iter_DA_Map(1.0, 2, f)
    assert np.isclose(x_da.cons(), np.sqrt(2), atol=1e-6)
    # derivative dx/dp should be 1/(2*sqrt(2))
    deriv = x_da.deriv(1).cons()
    assert np.isclose(deriv, 1/(2*np.sqrt(2)), atol=1e-6)

@pytest.mark.regression
def test_householder_iter_DA_Map():
    """
        DA polynomial Map verfied through 'implicit_function()' which is correct 
    """

    DA.init(4, 2)
    p = 2 + DA(1)
    T_DA = 0
    def f(x):
        return x*x - p
    
    x_nom = householder_iter_DA_nom(1.0, f, 2)
    x_da = householder_iter_DA_Map(x_nom, DA.getMaxVariables(), f)

    x_newton_nom =  newton_nomial_DA(1.0 ,T_DA ,f,tol=1e-9, MaxIter=1000)
    x_newton_da = Implicit_solver_DA(x_newton_nom, T_DA, f)


    order1 = x_newton_da.getMaxOrder()
    order2 = x_da.getMaxOrder()
    assert order1 == order2, "DA orders do not match"

    print(f"The Householder Function:\n{x_da}\n")
    print(f"The Newton Iteration Reference: \n{x_newton_da}\n")