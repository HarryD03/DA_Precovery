import numpy as np
import pytest
from Aphopis_Armillien.utils.iod import lambert_battin_DA, battin_A_DA, battin_x_DA, battin_vel_DA, Implicit_solver_DA, Nf
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray

def test_battin_vel_DA():
    """
    Typical-case (worked-example) accuracy, type/shape, and robustness
    checks for battin_vel_DA().

    Reference: Curtis Example 5.2 (Lambert's problem).
    """

    # -------------------------------------------------
    # 1.  Inputs from Example 5.2 ambert Problem, Example 5.2 (pp. 208–211):
    # -------------------------------------------------
    r1_vals = np.array([ 5000.0, 10000.0,  2100.0])     # km
    r2_vals = np.array([-14600.0,  2500.0,  7000.0])    # km
    dt       = 3600.0                                   # s
    a_val    = 20_000.0                                 # km
    mu_val   = 398_600.0                                # km^3/s^2
    dE_val   = 1.24116018738929                         # rad (matches Curtis f-g)                 # rad

    # -------------------------------------------------
    # 2.  Independent “ground-truth” velocities
    # -------------------------------------------------

    v1_ref = np.array([-5.99249,  1.92536,  3.24564])   # km/s :contentReference[oaicite:5]{index=5}
    v2_ref = np.array([-3.31246, -4.19662, -0.385288])  # km/s :contentReference[oaicite:6]{index=6}

    #    # -------------------------------------------------
    # 1️⃣  NUMPY branch  (plain floats)
    # -------------------------------------------------
    v2_num, v1_num = battin_vel_DA(r1_vals, r2_vals,
                                   a_val, dE_val, dt, mu_val)

    assert np.allclose(v1_num, v1_ref, rtol=1e-3)
    assert np.allclose(v2_num, v2_ref, rtol=1e-2)
    for vec in (v1_num, v2_num):
        assert isinstance(vec, np.ndarray) and vec.shape == (3,)

    # -------------------------------------------------
    # 2️⃣  DACEyPy-DA branch
    #      (constant DA scalars + DA array container)
    # -------------------------------------------------
    # Helper: wrap a scalar into a constant-value DA
    DA.init(4,6)
    def da_const(x: float) -> DA:
        i = 0
        return x + DA(i)         # DA(...) returns a constant-order object

    # Build r1_da = [val0+DA(1), val1+DA(2), val2+DA(3)]
    r1_da = array([r1_vals[i] + DA(i + 1) for i in range(3)])
    # Build r2_da = [val0+DA(4), val1+DA(5), val2+DA(6)]
    r2_da = array([r2_vals[i] + DA(i + 4) for i in range(3)])

    a_da  = a_val
    dE_da = dE_val

    v2_da, v1_da = battin_vel_DA(r1_da, r2_da,
                                 a_da, dE_da, dt, mu_val)

    # Ensure outputs are DA arrays
    assert all(isinstance(c, DA) for c in v1_da)
    assert all(isinstance(c, DA) for c in v2_da)

    # Compare **constant parts** of each DA element to reference values
    v1_da_const = np.array([v1_da[i].cons() for i in range(len(v1_da))])
    v2_da_const = np.array([v2_da[i].cons() for i in range(len(v2_da))])

    assert np.allclose(v1_da_const, v1_ref, rtol=1e-3)
    assert np.allclose(v2_da_const, v2_ref, rtol=1e-2)

    # -------------------------------------------------
    #   Physical sanity: specific energy must be negative (elliptic)
    # -------------------------------------------------
    def specific_energy(r, v):
        return 0.5 * np.dot(v, v) - mu_val / np.linalg.norm(r)

    assert specific_energy(r1_vals, v1_num) < 0
    assert specific_energy(r2_vals, v2_num) < 0

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

    # --------------------------- NumPy branch -----------------------
    A_num = battin_A_DA(x, r1, r2, mu)
    assert np.isclose(A_num, A_ref, rtol=1e-6)
    assert isinstance(A_num, float)

    # ----------------------------- DA branch ------------------------
    DA.init(4,7)                              # 6 vars, 4-th order

    r1_da = array([r1[i] + DA(i + 1) for i in range(3)])   # DA(1..3)
    r2_da = array([r2[i] + DA(i + 4) for i in range(3)])   # DA(4..6)
    x_da  = x + DA(7)

    A_da  = battin_A_DA(x_da, r1_da, r2_da, mu)

    assert isinstance(A_da, DA)
    assert np.isclose(A_da.cons(), A_ref, rtol=1e-6)
    return

# ------------------------------------------------------------------

def test_Implicit_solver_DA():
    """
    End-to-end test of Implicit_solver_DA on  f(x)=x²−2, expecting
    convergence to √2 ≈ 1.41421356.
    Test DA scalar - In theory if p is a column vector, it will still work - define func differently.
    """

    DA.init(4,2)

    def func(x_da, p):
        return x_da*x_da - p

    p = 1.0 + DA(1)
    x_init = DA(1.0) + DA(2)      # deliberately off the root
    root_da = Implicit_solver_DA(x_init, p, func)
    print(f"Solution is:\n{root_da}")

    def ex4_2_1(p0: float):
        def Nf(x, p):
            return x - (x * x - p) / (2 * x)
        tol = 1e-14
        x0 = 1.0   # x0 is just some initial guess
        i = 0

        # double precision computation => fast
        flag = True
        while flag:
            xp = x0
            x0 = Nf(xp, p0)
            i += 1
            flag = abs(xp-x0) > tol and i < 1000

        # DA computation => slow
        p = p0 + DA(1)
        x = x0
        i = 1
        while i <= 4:
            x = Nf(x, p)
            i *= 2


        print(f"Test: Full DA Newton\n{x}\n{x}\n")
        return (x)

    test_da = ex4_2_1(1.0)
    assert test_da == root_da
    return


def test_lambert_battin_DA():
    """
    Validates lambert_battin_DA for both float and DA inputs
    using Curtis Example 5.2 as ground truth.
    """
    # -------------------------------------------------
    # Example 5.2 data  (Vallado, pp. 498)
    # -------------------------------------------------
    r1_vals = np.array([ 15945.34, 0.0,  0.0])     # km
    r2_vals = np.array([-12214.838,  10249.46731,  0.0])    # km
    dt       = 3600.0                                   # s
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
