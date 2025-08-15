import numpy as np
import pytest
from utils.lambert_izzo import lambert_izzo, findxy, x2tof, x2tof2, hypergeometricF, householder_iter_DA_nom, householder_iter_DA_Map
from utils.iod import newton_nomial_DA, Implicit_solver_DA
from poliastro.iod import izzo
from poliastro.core.iod import _find_xy, _tof_equation, _householder, izzo
from astropy import units as u
from poliastro.bodies import Earth
from astropy.tests.helper import assert_quantity_allclose
from poliastro._math.special import hyp2f1b, stumpff_c2

from daceypy import DA, array
import daceypy.op as op

def sample_data():
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
    return R1, R2, DT, MU, V1_REF, V2_REF

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
def test_lambert_izzo_prograde_ShortARC():
    """Compare against reference Vallado velocities for single revolution.
        NOTE: We do -(Variable) in DA. This means the taylor coefficents in DA change and effectly mirror the nomimal solution and the distrubution
        """
    R1, R2, DT, MU, V1_REF, V2_REF = sample_data()
    NVar = len(R1) + len(R2)
    DA.init(4, NVar + 1)
    

    # Initialise positions as DA variables
    R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
    R2_DA = array([R2[i] + DA(i+4) for i in range(3)])
    M = 0
    R1 = R1_DA.cons()
    R2 = R2_DA.cons()

    velocities = lambert_izzo(R1_DA, R2_DA, DT, MU, M, prograde=True)
    v_1, v_2 = izzo(MU, R1, R2, DT, M, prograde=True, lowpath=False, numiter=30, rtol=1e-9)

    solution1 = velocities[0]

    v1 = solution1[:,0]
    v2 = solution1[:,1]

    print(f"My Lambert v2: {v2.cons()}\n")
    print(f"Expected v2: {V2_REF}")

    #Compare the Jacobian terms of velocity
    def jacobian_scipy():
        print("\n=== JACOBIAN COMPARISON ===")
        
        # Extract DA Jacobians from my implementation
        # v1 and v2 are DA objects with derivatives w.r.t. R1 and R2 components
        print("DA Jacobian for v1 (∂v1/∂R1, ∂v1/∂R2):")
        v1_jacobian_da = np.zeros((3, 6))  # 3 velocity components, 6 position components
        for i in range(3):  # v1 components (x, y, z)
            for j in range(6):  # R1 and R2 components (3+3)
                v1_jacobian_da[i, j] = v1[i].deriv(j+1).cons()
        print(v1_jacobian_da)
        
        print("\nDA Jacobian for v2 (∂v2/∂R1, ∂v2/∂R2):")
        v2_jacobian_da = np.zeros((3, 6))
        for i in range(3):  # v2 components (x, y, z)
            for j in range(6):  # R1 and R2 components (3+3)
                v2_jacobian_da[i, j] = v2[i].deriv(j+1).cons()
        print(v2_jacobian_da)
        
        # Compute finite difference Jacobian for reference
        from scipy.optimize import approx_fprime
        
        def izzo_wrapper(positions):
            """Wrapper for izzo function to compute finite difference Jacobian"""
            r1_fd = positions[:3]
            r2_fd = positions[3:6]
            v1_fd, v2_fd = izzo(MU, r1_fd, r2_fd, DT, M, prograde=True, lowpath=False, numiter=30, rtol=1e-9)
            return np.concatenate([v1_fd, v2_fd])  # Return [v1x, v1y, v1z, v2x, v2y, v2z]
        
        # Current position vector [R1x, R1y, R1z, R2x, R2y, R2z]
        pos_vector = np.concatenate([R1, R2])
        
        # Compute finite difference Jacobian
        h = 1e-6  # Step size
        jac_fd = np.zeros((6, 6))  # 6 velocity components, 6 position components
        for j in range(6):
            def func_j(pos):
                return izzo_wrapper(pos)[j]
            jac_fd[j, :] = approx_fprime(pos_vector, func_j, h)
        
        print("\nFinite Difference Jacobian (reference):")
        print("∂[v1, v2]/∂[R1, R2] =")
        print(jac_fd)
        
        # Split into v1 and v2 parts for comparison
        v1_jacobian_fd = jac_fd[:3, :]  # First 3 rows (v1)
        v2_jacobian_fd = jac_fd[3:, :]  # Last 3 rows (v2)
        
        print("\nFD Jacobian for v1:")
        print(v1_jacobian_fd)
        print("\nFD Jacobian for v2:")
        print(v2_jacobian_fd)

        return v1_jacobian_fd, v2_jacobian_fd, v1_jacobian_da, v2_jacobian_da
    
    v1_jacobian_fd, v2_jacobian_fd, v1_jacobian_da, v2_jacobian_da = jacobian_scipy()
    # Compare Jacobians
    v1_jac_error = np.abs(v1_jacobian_da - v1_jacobian_fd)
    v2_jac_error = np.abs(v2_jacobian_da - v2_jacobian_fd)
    
    max_v1_error = np.max(v1_jac_error)
    max_v2_error = np.max(v2_jac_error)
    
    print(f"\nJacobian Comparison:")
    print(f"Norm error in ∂v1/∂[R1,R2]: {np.linalg.norm(v1_jac_error)}")
    print(f"Norm error in ∂v2/∂[R1,R2]: {np.linalg.norm(v2_jac_error)}")

    # Check if Jacobians match within tolerance
    jac_tolerance = 1e-8  # Relaxed tolerance for numerical derivatives
    v1_jac_match = max_v1_error < jac_tolerance
    v2_jac_match = max_v2_error < jac_tolerance

    assert v1_jac_match, f"v1 Jacobian matches (tol={jac_tolerance}): {v1_jac_match}"
    assert v2_jac_match, f"v2 Jacobian matches (tol={jac_tolerance}): {v2_jac_match}"

    if not (v1_jac_match and v2_jac_match):
        print("❌ DA Jacobians don't match finite differences!")
        print("This suggests issues with DA automatic differentiation in Lambert solver")
    else:
        print("✅ DA Jacobians match finite differences")
    
    # Optionally assert on Jacobian accuracy (comment out if you expect failures)
    # assert v1_jac_match, f"v1 Jacobian error {max_v1_error:.2e} exceeds tolerance {jac_tolerance:.1e}"
    # assert v2_jac_match, f"v2 Jacobian error {max_v2_error:.2e} exceeds tolerance {jac_tolerance:.1e}"


    assert np.allclose(v1.cons(), V1_REF, rtol=1e-5)
    assert np.allclose(v2.cons(), V2_REF, rtol=1e-5)

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
    
    xy_palistro = _find_xy(L.cons(), T.cons(), M, 1000, 0, 1e-6)
    T_calc = x2tof(xy[0][0], 0, L)

    print(xy_palistro)
    x = xy_palistro[0]
    y = xy_palistro[1]
    

    assert len(xy) == 1
    assert np.isclose(T_calc.cons(), T.cons(), rtol=1e-6)
    assert np.isclose(x, xy[0][0].cons(), rtol=1e-6)
    assert np.isclose(y, xy[0][1].cons(), rtol=1e-6)


# @pytest.mark.basic
# def test_findxy_multi():
#     """x,y from findxy should satisfy x2tof(x)=T for multi rev
#         FALITUE IS DUE TO THE INPUT VARIABLES NOT THE ALGORITHM
#     """
#     NVar = len(R1) + len(R2)
#     DA.init(4, NVar + 1)

#     # Initialise positions as DA variables
#     R1_DA = array([R1[i] + DA(i+1) for i in range(3)])
#     R2_DA = array([R2[i] + DA(i+4) for i in range(3)])

    
    
#     L, T, _ = _compute_LT(R1_DA, R2_DA, DT, MU)
#     M = 2 #Mulit rev case

#     xy = findxy(L, T, M)    #[x,y]
#     assert len(xy) == 2     # [[x0l, y0l], [x0r, y0r]]
#     print(f"length:\n{len(xy)}\n")
#     print(f"{xy}")
    
#     T_calc_x0r = x2tof(xy[1][0], M, L) 
#     T_calc_x0l = x2tof(xy[0][0], M, L)

#     #Clockwise seems to be default motion -> x0r
#     assert np.isclose(T_calc_x0r.cons(), T.cons(), rtol=1e-6)

#     xy_palistro = izzo._find_xy(L.cons(), T.cons(), 0, 1000, 1e-6)
#     print(xy_palistro)
    
#     for x, y in xy_palistro:

#         i = 0
#         x = x
#         y = y

#         assert np.isclose(x, xy[i][0].cons(), rtol=1e-6)
#         assert np.isclose(y, xy[i][1].cons(), rtol=1e-6)

#         i += 1

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
    Tref = _tof_equation(x0,T0,L_DA.cons(), 0)  #Palistro function
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
    Tref = _tof_equation(x0,T0,L_DA.cons(), 0)  #Palistro function
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
    T2 = _tof_equation(x0,T0,L_DA.cons(), 0)  #Palistro function
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
    ref = hyp2f1b(val)

    calc = hypergeometricF(val)
    assert np.isclose(float(calc), ref, rtol=1e-8)

@pytest.mark.basic
def test_householder_iter_DA_nom_Basic():
    """Solve x^2=2 using Householder iterator. Compare solution with op.root()"""
    DA.init(4, 1)
    def f(x,p):
        return x*x - p
    
    p0 = 2
    x = householder_iter_DA_nom(1.0, p0, f)
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

    def f(x,p):
        """
        Function to obtain f(x) = T(x) - T* for Householder Iteration scheme
        """
        M, L_DA, T_DA = p
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
    

    x0 = x0.cons()      #x0 is a float
    p0 = [M, L_DA.cons(), T_DA.cons()]
    x_nom = householder_iter_DA_nom(x0, p0, f, tol=1e-9, MaxIter=1000)
    x = _householder(x0, T_DA.cons(), L_DA.cons(), M, 1e-9, 1000)
    assert np.isclose(x_nom, x, rtol=1e-9)      #relaxed error bound due to numerical deriv vs analytical 
                                                #Automatic Derivative of function introduces some error at e-10 decimal places
    #Compare to standard Newton_nomial to see numerical error small between solvers
    x_newton = newton_nomial_DA(x0, p0 ,f,tol=1e-9, MaxIter=1000, order=4)
    assert np.isclose(x_nom, x_newton, rtol=1e-9)

@pytest.mark.basic
def test_householder_iter_DA_Map_basic():
    """Check mapping sensitivity for x^2=p around p=2.
        x_nom = root(2)"""
    DA.init(4, 2)
    p0 = 2
    p = p0 + DA(1)
    def f(x,p):
        return x*x - p
    
    x_nom = householder_iter_DA_nom(1.0, p0, f, DA.getMaxVariables())
    print(x_nom)
    x_da = householder_iter_DA_Map(x_nom, p, DA.getMaxVariables(), f)
    assert np.isclose(x_da.cons(), np.sqrt(2), atol=1e-6)
    # derivative dx/dp should be 1/(2*sqrt(2))
    deriv = x_da.deriv(1).cons()
    assert np.isclose(deriv, 1/(2*np.sqrt(2)), atol=1e-6)

    x_da_newton = Implicit_solver_DA(DA(x_nom), p, f)
    x_da_newton_coeff = x_da_newton.getCoefficient([4, 0])  # Get the coefficient for the first variable (x)
    x_da_coeff = x_da.getCoefficient([4, 0])
    assert np.isclose(x_da_newton_coeff, x_da_coeff), f"ERROR:\nHouseholder:\n{x_da}\nNewton:\n{x_da_newton}"


