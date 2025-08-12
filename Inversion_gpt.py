"""
Implicit solution expansion with DACEyPy (Section 2.1 algorithm)

Given a parametric implicit system f(x, p) = 0 with x in R^nx, p in R^np,
this script returns the k-th order Taylor expansion x(p)
around a nominal pair (x0, p0) that satisfies f(x0, p0)=0.

Reference: Pirovano, Thesis §2.1, Eqs. (2.5)–(2.11).
"""

import numpy as np

# ---- DACEyPy imports ----
# The names below reflect the DACEyPy tutorials; adjust if your version differs.
from daceypy import DA, array            # DA core (variables/operations)
from daceypy import damap as damap      # DA maps (vector-valued Taylor maps)
from daceypy import settings as daset   # global settings (order, vars)

# ----------------------------
# Problem definition (edit me)
# ----------------------------
# Example: scalar implicit equation:  f(x, p) := x - cos(p) = 0
# Exact solution is x(p) = cos(p), which makes it easy to verify the Taylor series.
#
# To use a vector system, make f return a list/np.array with nx components.

def f_numeric(x, p):
    """Plain numeric f(x,p) used by Newton to get x0."""
    # x, p are numpy arrays of shape (nx,), (np,)
    # Here nx = 1, np = 1
    return np.array([x[0] - np.cos(p[0])])

def f_DA(x_DA, p_DA):
    """
    DA version of f(x,p). Inputs are DA scalars/vectors, return a DA vector.
    Build using DACEyPy arithmetic only (no NumPy trig on DA variables).
    """
    # For DA trig, use da.sin, da.cos, etc.
    return x_DA - DA.cos(p_DA)


# ----------------------------------------------------------
# Utility: damped Newton to find a consistent (x0, p0) pair
# ----------------------------------------------------------
def newton_solve_x0(f_numeric, x0_guess, p0, tol=1e-12, maxit=50):
    x = x0_guess.copy()
    for _ in range(maxit):
        r = f_numeric(x, p0)                      # residual (nx,)
        if np.linalg.norm(r) < tol:
            break
        # Finite-difference Jacobian wrt x (small system; robust enough here)
        nx = x.shape[0]
        J = np.zeros((nx, nx))
        eps = 1e-8
        for i in range(nx):
            dx = np.zeros_like(x); dx[i] = eps
            rp = f_numeric(x + dx, p0)
            J[:, i] = (rp - r) / eps
        step = np.linalg.solve(J, -r)
        x += step
        if np.linalg.norm(step) < tol:
            break
    return x


# ----------------------------------------------------------
# Core algorithm: DA map build, invert, evaluate at f=0
# ----------------------------------------------------------
def da_implicit_expansion(f_DA, x0, p0, k_order):
    """
    Return the DA Taylor expansion x(p) about (x0,p0) up to order k_order.

    Steps (Pirovano §2.1):
      1) Initialise [x] = x0 + δx, [p] = p0 + δp.          (Eq. 2.5)
      2) Compute DA map f = T_k(x,p).                      (Eq. 2.6)
      3) Augment with identity in p to get [f; p].         (Eq. 2.7)
      4) Invert map -> [x; p] = [T; I]^{-1} [f; p].        (Eq. 2.8)
      5) Evaluate at f=0   -> [x; p] = [T; I]^{-1}[0; p].  (Eq. 2.9)
      6) First row gives δx = T(p); so [x] = x0 + T(p).    (Eqs. 2.10–2.11)
    """
    # Set global DA environment
    # number of variables = np (parameters); the unknown x is solved out by inversion
    npv = len(p0)
    daset.set_variables(npv)         # declare how many DA indeterminates δp
    daset.set_degree(k_order)        # Taylor order k

    # Build DA variables for parameters: [p] = p0 + δp
    # The standard DACE pattern is: δp_i is the i-th DA basis variable.
    p_DA = [da.variable(i) + p0[i] for i in range(npv)]

    # Build a placeholder DA vector for x: start at x0; it will be solved by map inversion.
    x_DA = [da.constant(x0i) for x0i in x0]

    # Evaluate f in DA
    f_DA_vec = f_DA(x_DA, p_DA)   # list of DA scalars (length nx)
    nx = len(f_DA_vec)

    # Build augmented map M : [f; p] = M([x; p])
    # Domain variables of the map: first nx slots = x-components, last npv slots = p-components
    # Range components:
    #   y0..y{nx-1} = f_DA_vec(x,p)
    #   y{nx}..     = identity in p
    M = damap.DAMap(nx + npv, nx + npv)  # range dim, domain dim

    # Fill f rows (range indices 0..nx-1) as DA polynomials in domain variables
    # We make "x" domain variables symbolic by defining a temporary basis and composing.
    # Helper: damap uses variable indices [0..nx+npv-1]
    # Create symbolic domain vector Z = [X_symbols, P_symbols]
    Z = [da.zero() for _ in range(nx + npv)]
    for i in range(nx + npv):
        Z[i] = da.basis(i)  # δX_0..δX_{nx-1}, δP_0..δP_{npv-1}

    # Compose f(x,p) where x = x0 + δX, p = p0 + δP
    # (we already evaluated f at x_DA=x0, p_DA=p0+δP, but to be explicit and general:)
    # Remap p: replace DA variables with Z[nx + j] + p0[j]
    p_comp = [Z[nx + j] + p0[j] for j in range(npv)]
    # Remap x: replace with x0[i] + Z[i]
    x_comp = [Z[i] + x0[i] for i in range(nx)]
    f_composed = f_DA(x_comp, p_comp)  # DA list in terms of Z

    for r in range(nx):
        M.set_row(r, f_composed[r])   # f rows

    # Identity in p (rows nx..nx+npv-1): y = p = p0 + δp = p0 + Z[nx:].
    for j in range(npv):
        M.set_row(nx + j, Z[nx + j] + p0[j])

    # Invert the map: [x; p] = Minv([f; p])
    Minv = M.inverse()

    # Evaluate at f = 0: i.e., give the Minv input vector W = [0,...,0,  p]
    # We want x as a function of δp (i.e., p = p0 + δp). Let W’s last npv entries be p.
    # Build W(Z) = [0]*nx  +  [p0 + Z[nx:]]
    W = [da.constant(0.0) for _ in range(nx)] + [Z[nx + j] + p0[j] for j in range(npv)]

    # Y = Minv( W ) gives [x; p]. Extract the first nx components for x.
    Y = Minv.evaluate(W)
    x_series = Y[:nx]   # DA polynomials δx + x0

    # Ensure form [x] = x0 + T(p) by subtracting x0 constant if your build returns δx
    # (Most builds return absolute; if you see double-adding x0, uncomment below.)
    # x_series = [xi for xi in x_series]  # already absolute here

    return x_series  # list of DA polynomials representing x(p)


# ---------------------------------
# Demo / quick self-verification
# ---------------------------------
if __name__ == "__main__":
    # Problem sizes
    nx, npv, k = 1, 1, 7

    # Nominal parameter p0 and initial guess for x at p0
    p0 = np.array([0.2])
    x0_guess = np.array([1.0])

    # Get a consistent x0 with Newton: f(x0,p0)=0
    x0 = newton_solve_x0(f_numeric, x0_guess, p0)

    # Build expansion
    x_series = da_implicit_expansion(f_DA, x0, p0, k_order=k)

    # Print the series and do a couple of checks
    print("Nominal solution x0:", x0)
    print("Taylor series x(p) about p0 (DA):", x_series[0])

    # Evaluate the series at a few δp values and compare to exact cos(p)
    test_ps = [p0[0] + dp for dp in (-0.1, 0.0, +0.1)]
    for pt in test_ps:
        # Evaluate DA poly by setting δp = pt - p0
        daset.set_variables(npv)
        daset.set_degree(k)
        dp_DA = [da.constant(pt - p0[0])]
        val = x_series[0].evaluate(dp_DA)
        print(f"p={pt:.4f}  series≈{val:.12f}   exact={np.cos(pt):.12f}")
