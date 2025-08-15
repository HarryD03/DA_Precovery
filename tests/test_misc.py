import numpy as np
from utils.iod import lambert_battin_DA, battin_A_DA, Implicit_solver_DA, Nf, newton_nomial_DA, kepler_F, Implicit_solver_DAVec, newton_nominal_DAVec
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import householder_iter_DA_nom, householder_iter_DA_Map, x2tof
from scipy.optimize import approx_fprime
# =========================  TESTS  ================================

def test_battin_A_DA():
    """
    Accuracy/shape/type tests for battin_A_DA(), NumPy + DA branches.
    Worked example built from simple geometry (two coplanar points).
    """

    # --- Geometry & inputs (arbitrary but deterministic) ------------
    r1 = np.array([7000.0,     0.0, 0.0])      # km
    r2 = np.array([8000.0,   500.0, 0.0])      # km
    x  = 0.2                                    # Battin parameter
    mu = 1.32712440018e11                       # default

    # ---- Independent reference calculation (NumPy) ----------------
    c_vec   = r2 - r1
    c       = np.linalg.norm(c_vec)
    r1mag   = np.linalg.norm(r1)
    r2mag   = np.linalg.norm(r2)
    s       = 0.5 * (r1mag + r2mag + c)
    g       = s / (2 * (1 - x**2))
    alpha   = 2 * np.arcsin(np.sqrt(s / (2 * g)))
    beta    = 2 * np.arcsin(np.sqrt((s - c) / (2 * g)))
    A_ref   = g**1.5 * (alpha - np.sin(alpha) - beta + np.sin(beta))

    """
    # --------------------------- NumPy branch -----------------------
    A_num = battin_A_DA(x, r1, r2, mu)
    assert np.isclose(A_num, A_ref, rtol=1e-6)
    assert isinstance(A_num, float)
    """

    # ----------------------------- DA branch ------------------------
    DA.init(4,7)                              # 6 vars, 4-th order

    r1_da = array([r1[i] + DA(i + 1) for i in range(3)])   # DA(1..3)
    r2_da = array([r2[i] + DA(i + 4) for i in range(3)])   # DA(4..6)
    x_da  = x + DA(7)

    A_da  = battin_A_DA(x_da, r1_da, r2_da)

    assert isinstance(A_da, DA)
    assert np.isclose(A_da.cons(), A_ref, rtol=1e-6), f"Obtain {A_da.cons()}, Expected {A_ref}"

# ------------------------------------------------------------------


def test_Implicit_solver_DA():
    """
    End-to-end test of Implicit_solver_DA on  f(x)=x²−2, expecting
    convergence to √2 ≈ 1.41421356.
    Test DA scalar - In theory if p is a column vector, it will still work - define func differently.
    Aim of test - see if the Automatic derivative function works

    This can now be integrated into a Full Scalar Solver
    """
    

    p0 = 2
    x0 = 1.0
    order = 6

    def func(x_da, p):
        return x_da**2 - p

    x_nom = newton_nomial_DA(x0, p0, func, tol=1e-14, MaxIter=1000, order=order)
    print(f"Nominal x:\n{x_nom}")

    DA.init(6, 2)  # Initialize DA with order and number of variables
    p = p0 + DA(1)
    x_init = x_nom
    root_da = Implicit_solver_DA(x_init, p, func)
    print(f"Solution is:\n{root_da}")

    p = p0 + DA(1)
    x_init = DA(x_nom)
    def ex4_2_1(p0: float, x0: float):
        def Nf(x, p):
            return x - (x * x - p) / (2 * x)
        tol = 1e-14
        x0 = x0   # x0 is just some initial guess
        i = 0

        # double precision computation => obtain nominal solution/ expansion point
        flag = True
        while flag:
            xp = x0
            x0 = Nf(xp, p0)
            i += 1
            flag = abs(xp-x0) > tol and i < 1000

        print(f"Test: Nominal Newton finder:\n{x0}\n")
        # DA computation => slow
        p = p0 + DA(1)
        x = x0
        i = 1
        while i <= DA.getMaxOrder():
            x = Nf(x, p)
            i *= 2


        print(f"Test: Full DA Newton\n{x}\n")
        return (x)

    
    test_da = ex4_2_1(p0,x0)

    assert test_da.getCoefficient([5,    0]) == root_da.getCoefficient([5, 0]), f"Expected {test_da} == {root_da}"

def test_NewtonDAVec_jacobian():
    """Test Newton's method for solving a nonlinear system of equations
    """
    def compute_jacobian_finite_difference(f, x, p, h=1e-8):
        """
        Compute Jacobian of function f with respect to x using finite differences
        
        Args:
            f: Function that takes (x, p) and returns array
            x: Point to evaluate Jacobian (array)
            x: Point to evaluate Jacobian (array)
            p: Parameters (kept constant)
            h: Step size for finite differences
        
        Returns:
            jacobian: Matrix where jacobian[i,j] = ∂f_i/∂x_j
        """
        def f_wrapper(x_var):
            return f(x_var, p)
        
        # Get function dimension
        f_val = f_wrapper(x)
        n_eq = len(f_val)  # Number of equations
        n_var = len(x)     # Number of variables
        
        jacobian = np.zeros((n_eq, n_var))
        
        # Compute each row of Jacobian
        for i in range(n_eq):
            def f_i(x_var):
                return f_wrapper(x_var)[i]
            
            jacobian[i, :] = approx_fprime(x, f_i, h)
        
        return jacobian
    
    def f(x_da, p):
        f11 = x_da[0] + 2*x_da[1] - p[0]
        f22 = x_da[0]**2 + 4*x_da[1]**2 - p[1]
        if isinstance(x_da, array):
            F = array([f11, f22])
        else:
            F = np.array([f11,f22])
        return F

    x0 = np.array([1, 2])
    p = np.array([2, 4])
    DA.init(4, 2)
    x_nom = newton_nominal_DAVec(x0, p, f, 4)

    DA.init(4,4)
    p_da = array([p[i] + DA(i + 1) for i in range(2)])  # Create DA parameters
    x_da = Implicit_solver_DAVec(x_nom, p_da, f, 2)

    # Obtain the Jacobian function 
    Jac_da = np.array( [ [1, 2], [0, 8] ] )
    Jac_scipy = compute_jacobian_finite_difference(f, x_nom, p)

    diff = Jac_da - Jac_scipy
    assert np.linalg.norm(diff) < 1e-6, f"Jacobian mismatch: DA {Jac_da} vs SciPy {Jac_scipy}"

    x_ref = np.array([0, 1])
    assert np.allclose(x_nom, x_ref, rtol=1e-12)


def test_Implicit_solver_DA_maybeVec():
    
    def f(x_da, p):
        return x_da**2 - p
    
    x0 = np.array([1.0, 2.0, 1.0])
    p0 = np.array([2.0, 4.0, 2.0])

    x_nom0 = newton_nomial_DA(x0, p0, f, tol=1e-14, MaxIter=1000, order=4)
    print(f"Nominal x:\n{x_nom0}")

    DA.init(4, 4)  # Initialize DA with order and number of variables
    p = array([p0[i] + DA(i + 1) for i in range(3)])

    x_da = Implicit_solver_DA(x_nom0, p, f)
    print(f"Solution is:\n{x_da}")

    x_nom1 = newton_nomial_DA(x0[1], p0[1], f, tol=1e-14, MaxIter=1000, order=4)
    print(f"Nominal x :\n{x_nom0}\nNominal x (vectorized):\n{x_nom1}")

    DA.init(4, 2)  # Initialize DA with order and number of variables
    p = p0[1] + DA(1)
    root_da = Implicit_solver_DA(x_nom1, p, f)
    print(f"Solution:\n{x_da}\nSolution (vectorized):\n{root_da}\n")

def test_Implicit_solver_DA_vectorized():
    def f(x_da, p):
        return x_da**2 - p
        
    x0 = np.array([1.0, 2.0, 1.0])
    p0 = np.array([2.0, 4.0, 2.0])

    x_nom0 = newton_nomial_DA(x0, p0, f, tol=1e-14, MaxIter=1000, order=4)
    print(f"Nominal x:\n{x_nom0}")

    DA.init(4, 4)  # Initialize DA with order and number of variables
    p = array([p0[i] + DA(i + 1) for i in range(3)])

    x_da = Implicit_solver_DA(x_nom0, p, f)
    print(f"Solution is:\n{x_da}")

    x_nom1 = newton_nomial_DA(x0[1], p0[1], f, tol=1e-14, MaxIter=1000, order=4)
    print(f"Nominal x :\n{x_nom0}\nNominal x (vectorized):\n{x_nom1}")

    DA.init(4, 2)  # Initialize DA with order and number of variables
    p = p0[1] + DA(1)
    root_da = Implicit_solver_DA(x_nom1, p, f)
    print(f"Solution:\n{x_da}\nSolution (vectorized):\n{root_da}\n")
    
def test_newton_nominal_DAVec():
    """
    Newton's method for solving a Nominal root with DA automatic differentiation.
    f(x)=x²−2, expecting
    convergence to √2 ≈ 1.41421356.
    """
    def f(x_da, p):
        return x_da**2 - p

    # Set initial guess and parameters
    x0 = np.array([1.0, 1.0, 1.0])
    p = np.array([2.0, 2.0, 2.0])

    x = newton_nominal_DAVec(x0, p, f, 4)

    print(f"Nominal root:\n{x}")
    assert np.allclose(x, np.array([1.41421356, 1.41421356, 1.41421356]), atol=1e-6), f"Expected [1.41421356, 1.41421356, 1.41421356]\nGot {x}"

def test_newton_nominal_DAVec():
    """
        Conduct Newton for a nonlinear system of equations
    """
    def f(x_da, p):
        """Root function for testing"""
        f11 = x_da[0] + 2*x_da[1] - 2
        f22 = x_da[0]**2 + 4*x_da[1]**2 - 4
        F = array([f11, f22])
        return F
    
    x0 = np.array([1, 2])
    p = np.array([0, 0])
    x = newton_nominal_DAVec(x0, p, f, 4)
    x_ref = np.array([0, 1])
    assert np.allclose(x, x_ref, rtol=1e-12)


def test_Implicit_solver_DAVec():
    """Root function for testing
    Basic Test for Independent vector inputs
    DA independent inputs

    """
    def f(x_da, p):
        return x_da**2 - p

    x0 = np.array([1.0, 1.0, 1.0])
    p = np.array([2.0, 2.0, 2.0])

    x = newton_nominal_DAVec(x0, p, f, 4)
    # Set initial guess and parameters

    DA.init(5, 6)

    p = array([2.0 + DA(1), 2.0 + DA(2), 2.0 + DA(3)])

    # Call implicit_solver_DAVec - This is incorrect.
    root_da = Implicit_solver_DAVec(x, p, f, 3, x0DA=False, DAIOD=False)

    # Print the result
    print(f"Root: {root_da}")
    # Check result
    assert isinstance(root_da, array)
    assert len(root_da) == 3
    coeff = root_da[0].getCoefficient([1,0,0,0,0,0])
    print(coeff)
    assert np.isclose(coeff, 3.5355339059327373e-01, atol=1e-6)

def test_Implicit_solver_DAVecComplex():
    """Root function for testing
    Basic Test for Independant vector inputs 
    DA independent inputs
    
    """
    DA.init(4, 6)  # When greater than 4th order theres an error - Its todo with the .inv() function for inverse jacobian I think
    def f(x_da):
        return x_da**2 - p
    # Initialize DA variables

    # Set initial guess and parameters
    x_init = array([1.0, 2.0, 3.0])         #Must have a resonable initial guess
    p = array([1.0 + DA(1), 2.0 + DA(2), 3.0 + DA(3)])

    # Call implicit_solver_DAVec
    root_da = Implicit_solver_DAVec(x_init, p, f, 3, x0DA=False, DAIOD=False)
    # Print the result
    print(f"Root: {root_da}")
    # Check result
    assert isinstance(root_da, array)
    assert len(root_da) == 3
    assert np.allclose(root_da.cons(), [1.0, 1.0, 1.0], atol=1e-6)
    coeff = root_da[0].getCoefficient([4,0,0,0,0,0])
    assert np.allclose(coeff, -3.90625e-02, atol=1e-6)


def test_lambert_battin_DA():
    """
    Validates lambert_battin_DA for both float and DA inputs
    using Curtis Example 5.2 as ground truth.
    """
    # -------------------------------------------------
    # Example 5.2 data  (Vallado, pp. 498)
    # -------------------------------------------------
    r1_vals = np.array([ 15945.34, 0.0,  0.0])     # km
    r2_vals = np.array([-12214.83899,  10249.46731,  0.0])    # km
    dt       = 76 * 60                                   # s
    mu_val   = 1.32712440018e11                              # km³/s²
    max_iter = 100
    order    = 4                                        # DA order

    v1_ref = np.array([2.058925,  2.915965,  0.0])   # km/s
    v2_ref = np.array([-3.451565, 0.910315, 0.0])  # km/s

    # -------------------------------------------------
    # 2️⃣ DACEyPy-DA branch
    # -------------------------------------------------
    DA.init(order, 6)                      # 7 generators, chosen order

    # r1_da → val + DA(1..3);  r2_da → val + DA(4..6)
    r1_da = array([(r1_vals[i] + DA(i+1)) for i in range(3)])
    print(f"Position 1 DA:\n{r1_da}\n")
    r2_da = array([r2_vals[i] + DA(i + 4) for i in range(3)])
    print(f"Position 2 DA:\n{r2_da}\n")
    v2_da, v1_da = lambert_battin_DA(r1_da, r2_da,
                                     dt, max_iter,
                                     mu=mu_val, order=order)
    # --- type & shape ---
    assert all(isinstance(c, DA) for c in v1_da)
    assert all(isinstance(c, DA) for c in v2_da)

    # --- constant parts vs. reference ---
    v1_da_const = np.array([v1_da[i].cons() for i in range(len(v1_da))])
    v2_da_const = np.array([v2_da[i].cons() for i in range(len(v2_da))])

    assert np.allclose(v1_da_const, v1_ref, rtol=1e-3)
    assert np.allclose(v2_da_const, v2_ref, rtol=1e-2)

    # -------------------------------------------------
    # 3️⃣  Physical sanity: energy (elliptic ⇒ ε < 0)
    # -------------------------------------------------
    def specific_energy(r, v):
        return 0.5 * np.dot(v, v) - mu_val / np.linalg.norm(r)

    assert specific_energy(r1_vals, 0) < 0
    assert specific_energy(r2_vals, 0) < 0

def test_newton_nomial_DA():
    #Obtain dE_nom solution via Kepler_F
    def _leo_state_km():
        """
        Circular prograde LEO state (km-units). Taken form Example 2-4 page 94

        The position is on +X; velocity on +Y so that r·v = 0.
        """
        r0 = np.array([1131.340, -2282.343, 6672.423])       # km
        v0 = np.array([-5.64305, 4.30333, 2.42879])           # km s⁻¹

        #Textbook solution
        r1 = np.array([-4219.7527, 4363.0292, -3958.7666])
        v1 = np.array([3.689866, -1.916735, -6.112511])
        return r0, v0, r1, v1

    r1, v1, _, _ = _leo_state_km()

    DA.init(4,7)
    r1 = array([r1[i] + DA(i + 1) for i in range(3)])
    v1 = array([v1[i] + DA(i + 4) for i in range(3)])
    mu = 3.986e5
    dt = 40 * 60
    sigma = r1.dot(v1) / op.sqrt(mu)
    energy = v1.dot(v1) / 2 - (mu / op.vnorm(r1))  #Specific orbital energy
    a = - mu / (2 * energy)  #Semi-major axis
    n = op.sqrt(mu/a**3)
    dM = n*dt

    #Obtain dE_nom solution via Kepler_F
    dE_nom_kepler = kepler_F(a, sigma, op.vnorm(r1), dM)
    #Obtain dE_nom solution via newton_nomial_DA and seperate kepler equation function
    def fkep(dE, p):
        
        #Unpack p
        a = p[0]
        sigma = p[1]
        r1_norm = p[2]
        dM = p[3]

        return dE + (sigma / op.sqrt(a)) * (1 - op.cos(dE)) - (1 - r1_norm/ a) * op.sin(dE) - dM
    

    p = array([a, sigma, op.vnorm(r1), dM]) #pack p 
    tol = 1e-12
    dE_nom_newton = newton_nomial_DA(dM, p, fkep, tol, MaxIter=1000)

    assert np.isclose(dE_nom_kepler, dE_nom_newton, atol = 1e-9), f"Newton Function:{dE_nom_newton}\nIntegrated Function: {dE_nom_kepler}\n"

    #Assert same 


def test_newton_nomial_DAVec():
    # Define a test function
    def f(x):
        return x**2 - p

    DA.init(4, 4)
    # Define the initial guess and parameters
    x0 = np.array([1.0, 1.0])
    p = array([2.0 + DA(1), 2.0 + DA(2)])

    # Define the tolerance and maximum number of iterations
    tol = 1e-16
    MaxIter = 10000

    # Call the newton_nomial_DAVec function
    result = newton_nomial_DAVec(x0, p, f, tol, MaxIter)

    # Print the result
    print("Result:", result)
    # Check the result
    assert np.allclose(result, np.array([1.42, 1.42]), atol=1e-2)