#import packages
import utils.time_reference as time_ref
import utils.propagation as prop
import utils.iod as iod
import utils.dynamics as dynamics
from utils.lambert_izzo import lambert_izzo
from numpy.random import default_rng

import numpy as np 
from numpy.typing import NDArray


from daceypy import DA, array, ADS
import daceypy.op as op

# Monte Carlo Simulation of zeroth order DAIOD
# Monte Carlo Simulation of Classical Gauss IOD

def los_jacobian(RA, DEC):
    ca, sa = np.cos(RA), np.sin(RA)
    cd, sd = np.cos(DEC), np.sin(DEC)
    du_dalpha = np.array([-cd*sa, cd*ca, 0.0])
    du_ddelta = np.array([-sd*ca, -sd*sa, cd])
    return np.column_stack([du_dalpha, du_ddelta])  # 3x2

def los_from_radec(RA, DEC):
    cD = np.cos(DEC); sD = np.sin(DEC)
    cA = np.cos(RA);  sA = np.sin(RA)
    return np.array([cD*cA, cD*sA, sD])

def sample_los(RA, DEC, Sigma_ra_dec, rng):
    """
    One MC draw of the line-of-sight unit vector from RA,DEC and 2x2 Covariance
    """
    #draw RA and DEC
    d_ra, d_dec = rng.multivariate_normal(mean=[0.0, 0.0], cov=Sigma_ra_dec)
    
    # Clamp perturbations to 3-sigma limits
    sigma_ra = np.sqrt(Sigma_ra_dec[0, 0])
    sigma_dec = np.sqrt(Sigma_ra_dec[1, 1])
    d_ra = np.clip(d_ra, -3*sigma_ra, 3*sigma_ra)
    d_dec = np.clip(d_dec, -3*sigma_dec, 3*sigma_dec)
    
    #u0 = np.array([np.cos(DEC)*np.cos(RA), np.cos(DEC)*np.sin(RA), np.sin(DEC)])
    #J = los_jacobian(RA, DEC)
    RA_s = RA + d_ra
    DEC_s = DEC + d_dec
    RA_s = (RA_s + 2*np.pi) % (2*np.pi)  # wrap to [0, 2π)
    DEC_s = np.clip(DEC_s, -np.pi/2, np.pi/2)  # clip to [-π/2, π/2]
    u = los_from_radec(RA_s, DEC_s)
    return u


def monte_carlo_gauss_PWiod(num_simulations: int, observer_position: NDArray, RA_nom, DEC_nom, time_sec: NDArray, RA_sigma, DEC_sigma, mu: float, prograde_bool_) -> NDArray:
    """
    Monte Carlo Simulation of Classical Gauss IOD
    :param num_simulations: Number of Simulations
    :param observer_position: Position of the observer (Same frame as target state) [3x3] 
    :param RA_nom: Nominal Right Ascension observations (radians)   [1x3]
    :param DEC_nom: Nominal Declination observations (radians)      [1x3]
    :param time_sec: Time of observations (seconds)                 [1x3]
    :param RA_sigma: RA uncertainty (radians)                 [1x3]
    :param DEC_sigma: DEC uncertainty (radians)                [1x3]
    :param mu: Gravitational parameter (km^3/s^2)

    Returns:
        NDArray: Shape (num_simulations, state_dim) where state_dim is typically 6 (position + velocity)
    """

    #TODO: No process noise / 2BP dynamics considered

    results_DAIOD = np.zeros((num_simulations, 6))
    results_gauss = np.zeros((num_simulations, 6))
    rng = default_rng(42)

    if isinstance(RA_sigma, (int, float)):
        RA_sigma = np.ones(3) * RA_sigma
    if isinstance(DEC_sigma, (int, float)):
        DEC_sigma = np.ones(3) * DEC_sigma

    #Calculate the covariance matrix of the observations
    cov = []
    for i in range(len(RA_nom)):
        cov_tmp = np.array([[(RA_sigma[i])**2, 0.0],
                           [0.0, (DEC_sigma[i])**2]])
        cov.append(cov_tmp)
        

    for i in range(num_simulations):
        # Perturb the input parameters slightly
        u = np.zeros((3,3))
        for j in range(len(RA_nom)): 
            u[:,j] = sample_los(RA_nom[j], DEC_nom[j], cov[j], rng)

        try: #DAIOD Zeroth Order (Guass + Lambert correction)
            r, range_vec, range_mag, v_2 = iod.Guass_8th_seed(observer_position, u, time_sec, mu)
            X0_Gauss = np.concatenate((r[:,1], v_2))
            
            #Ellipse condition

            results_gauss[i, :] = X0_Gauss

            #DAIOD step
            range_mag_L1_nom = iod.DAIOD_1Scipy_invert(range_mag, u, time_sec, None, observer_position, mu, tol=1e-9, prograde_bool=prograde_bool_)
            
            # Converted refined magnitude to State Vector
            range_vec = range_mag_L1_nom * u
            r_vec = range_vec + observer_position
            dt1 = time_sec[1] - time_sec[0]
            dt2 = time_sec[2] - time_sec[1]


            velocities1 = lambert_izzo(r_vec[:,0], r_vec[:,1], dt1, mu, 0, prograde=prograde_bool_)
            v2 = velocities1[0][:,1]

            X_epoch = np.concatenate((r_vec[:,1], v2))
            #Ellipse condition
            
            results_DAIOD[i, :] = X_epoch
 
        except Exception as e:
            print(f"Error occurred: {e}")
            continue
        

    return results_DAIOD, results_gauss 

def monte_carlo_propagation(results: NDArray, t0: float, tf: float, Ts: int = 100) -> NDArray:
    """
    Access the IOD VA cloud and propagate to times of interest.
    :param results: Initial state vectors from IOD (num_simulations, state_dim)
    """
    propagated_results = np.zeros((results.shape[0], 6, Ts))       # Structure: (Number Simulation x State Vector x Propagation times)
    tgrid = np.linspace(t0, tf, Ts)

    propagated_results[:,:,0] = results

    for j in range(results.shape[0]):  #For a Perturbed State
        
        X0 = results[j, :] #initial state

        for i in range(Ts-1):  # Iterate over the results:
            # Propagate each result using the desired method (e.g., Keplerian propagation)

            propagated_result = prop.base_propagationPW(X0, tgrid[i], tgrid[i+1], dynamics.TBP_CC_FP)

            propagated_results[j, :, i+1] = propagated_result
            X0 = propagated_result
        

    return propagated_results, tgrid 


# -------------------------
# Angle utilities
# -------------------------
def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def _circular_mean(a):  # radians
    s, c = np.sin(a).mean(), np.cos(a).mean()
    return (np.arctan2(s, c)) % (2*np.pi)

def vec_to_radec(rel):
    """rel: (…,3) relative position vectors. Returns (RA, DEC) arrays of shape (…)"""
    x = rel[..., 0]; y = rel[..., 1]; z = rel[..., 2]
    ra  = (np.arctan2(y, x)) % (2*np.pi)
    dec = np.arctan2(z, np.hypot(x, y))
    return ra, dec

# -------------------------
# Ensemble state stats
# -------------------------
def ensemble_state_stats(propagated_results):
    """
    propagated_results: (N, 6, Ts)
    Returns:
      mu_state: (6, Ts)   sample mean
      P_state:  (Ts, 6,6) sample covariance at each time
    """
    N, nx, Ts = propagated_results.shape
    mu_state = propagated_results.mean(axis=0)               # (6, Ts)
    Xc = propagated_results - mu_state[None, :, :]           # (N, 6, Ts)
    P_state = np.zeros((Ts, nx, nx))
    for k in range(Ts):
        Xk = Xc[:, :, k]                                     # (N, 6)
        Pk = (Xk.T @ Xk) / (N - 1)                           # unbiased
        P_state[k] = 0.5*(Pk + Pk.T)                         # symmetrize
    return mu_state, P_state

# -------------------------
# RSW/RTN triad + transforms
# -------------------------
def rsw_frame(r, v):
    R = r / np.linalg.norm(r)
    h = np.cross(r, v)
    W = h / np.linalg.norm(h)
    S = np.cross(W, R)
    T = np.vstack((R, S, W))                                 # rows are unit vectors
    return T                                                 # x_RSW = T @ x_I

def rsw_over_time(mu_state, P_state):
    """
    Build RSW transform from mean (r,v) at each time and project mean/cov.
    Returns:
      T_RSW:  (Ts, 3,3)
      mu_RSW:(6, Ts)
      P_RSW: (Ts, 6,6)
    """
    Ts = mu_state.shape[1]
    T_RSW = np.zeros((Ts, 3, 3))
    mu_RSW = np.zeros_like(mu_state)
    P_RSW  = np.zeros_like(P_state)
    Z = np.zeros((3,3))

    for k in range(Ts):
        r = mu_state[0:3, k]; v = mu_state[3:6, k]
        T = rsw_frame(r, v)
        T_RSW[k] = T
        Q = np.block([[T, Z],
                      [Z, T]])                                # 6x6
        mu_RSW[:, k] = Q @ mu_state[:, k]
        Pk = Q @ P_state[k] @ Q.T
        P_RSW[k] = 0.5*(Pk + Pk.T)
    return T_RSW, mu_RSW, P_RSW

# -------------------------
# RA/DEC stats over time
# -------------------------

def radec_stats_from_samples(RA_samps, DEC_samps):
    """
    RA_samps, DEC_samps: (N, Ts)
    Returns:
      RA_mean, DEC_mean: (Ts,)
      P_angle:           (Ts, 2,2) covariance of [dRA, dDEC] (radians^2)
      std_angle:         (Ts, 2)   1-sigma for RA and DEC
    Notes:
      RA covariance uses wrapped differences around the circular mean.
    """
    N, Ts = RA_samps.shape
    RA_mean  = np.zeros(Ts)
    DEC_mean = np.zeros(Ts)
    P_angle  = np.zeros((Ts, 2, 2))
    std_angle= np.zeros((Ts, 2))

    for k in range(Ts):
        ra0  = _circular_mean(RA_samps[:, k])
        dra  = _wrap_pi(RA_samps[:, k] - ra0)
        dec0 = DEC_samps[:, k].mean()
        ddec = DEC_samps[:, k] - dec0

        X = np.column_stack((dra, ddec))                        # (N,2)
        C = (X.T @ X) / (N - 1)
        P_angle[k]   = 0.5*(C + C.T)
        std_angle[k] = np.sqrt(np.diag(P_angle[k]))
        RA_mean[k], DEC_mean[k] = ra0, dec0

    return RA_mean, DEC_mean, P_angle, std_angle

# -------------------------
# One-call wrapper
# -------------------------
def compute_uncertainties_all(propagated_results, tgrid, observer_positions):
    """
    Inputs:
      propagated_results: (N, 6, Ts) Monte Carlo ensemble (inertial frame)
      tgrid:              (Ts,) time grid [s]
      observer_positions: (3, Ts) or callable f(t)->(3,) in the SAME frame
    Returns dict with:
      mu_state: (6,Ts)         P_state: (Ts,6,6)
      T_RSW: (Ts,3,3)          mu_RSW:  (6,Ts)     P_RSW: (Ts,6,6)
      RA_mean:(Ts,)            DEC_mean:(Ts,)      P_angle:(Ts,2,2)
      std_angle:(Ts,2)  # [σ_RA, σ_DEC]
    """
    mu_observables, P_observables = ensemble_state_stats(propagated_results)  # (6, Ts)
    
    RA = mu_observables[0, :]
    DEC = mu_observables[1, :]

    # Extract RAxDEC covariance Matrix
    P_angle = np.zeros((P_observables.shape[0], 2, 2))  # (Ts, 2,2)
    RA_DEC_cov = P_observables[:, 0:2, 0:2]
    
    RA_sigma = np.sqrt(np.clip(RA_DEC_cov[:, 0, 0], 0.0, None))  # radians
    DEC_sigma = np.sqrt(np.clip(RA_DEC_cov[:, 1, 1], 0.0, None))  # radians
    print(f"Mean RA uncertainty: {np.degrees(RA_sigma).mean()} deg, DEC uncertainty: {np.degrees(DEC_sigma).mean()} deg")

    return {
        "mu_observables": mu_observables,
        "Cov_observables":  P_observables,
        "RA_mean": RA,
        "DEC_mean": DEC,
        "Cov_RADEC" : RA_DEC_cov,
        "RA_sigma": RA_sigma,
        "DEC_sigma": DEC_sigma
    }

# -------------------------
# Convenience: quick 1-σ summaries
# -------------------------

def state_std(P_state):
    """Return 1-σ per component for position and velocity over time.
       Outputs: std_pos:(Ts,3), std_vel:(Ts,3)"""
    Ts = P_state.shape[0]
    std_pos = np.sqrt(np.stack([np.diag(P_state[k, 0:3, 0:3]) for k in range(Ts)], axis=0))
    std_vel = np.sqrt(np.stack([np.diag(P_state[k, 3:6, 3:6]) for k in range(Ts)], axis=0))
    return std_pos, std_vel

def rsw_std(P_RSW):
    """1-σ per component in RSW: position (R,S,W) and velocity (R,S,W)."""
    Ts = P_RSW.shape[0]
    std_rsw_pos = np.sqrt(np.stack([np.diag(P_RSW[k, 0:3, 0:3]) for k in range(Ts)], axis=0))
    std_rsw_vel = np.sqrt(np.stack([np.diag(P_RSW[k, 3:6, 3:6]) for k in range(Ts)], axis=0))
    return std_rsw_pos, std_rsw_vel


import numpy as np

# Common chi-square quantiles for 3 DoF
_CHI2_3DOF = {
    0.683: 3.531,   # ~68.3%
    0.950: 7.815,   # 95%
    0.990: 11.345,  # 99%
}

def _chi2_3d(conf):
    """Return chi-square quantile for df=3. Falls back to 95% if conf not in table."""
    return _CHI2_3DOF.get(round(conf, 3), _CHI2_3DOF[0.950])

def rsw_cov_ellipsoid_3d(mu_RSW_k, P_RSW_k, conf=0.95, n_theta=30, n_phi=60, block="pos"):
    """
    3D covariance ellipsoid in RSW at a single time index.

    Inputs:
      mu_RSW_k : (6,) mean state in RSW at time k  [R,S,W, vR, vS, vW]
      P_RSW_k  : (6,6) covariance in RSW at time k
      conf     : confidence level (e.g., 0.683, 0.95, 0.99)
      n_theta  : samples around azimuth (0..2π)
      n_phi    : samples in polar (0..π)
      block    : "pos" (0:3) for position ellipsoid, or "vel" (3:6) for velocity

    Returns:
      center   : (3,) ellipsoid center in chosen RSW block (km or km/s)
      XYZ      : tuple (X, Y, Z) each of shape (n_theta, n_phi) — mesh to plot
      axes     : (a1, a2, a3) semi-axes lengths (largest→smallest)
      V        : (3,3) columns are principal directions in RSW coords
    """
    idx = slice(0, 3) if block == "pos" else slice(3, 6)
    center = mu_RSW_k[idx]

    # 3x3 covariance (symmetrize for safety)
    C = P_RSW_k[idx, idx]
    C = 0.5 * (C + C.T)

    # Eigen-decomposition (ascending), clip small negatives
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)

    # Confidence scaling: ellipsoid defined by (x-μ)^T C^{-1} (x-μ) = χ²
    chi2 = _chi2_3d(conf)
    axes = np.sqrt(w * chi2)  # semi-axes along eigenvectors

    # Sort by descending axis length for consistency
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    V = V[:, order]

    # Unit sphere grid (theta: 0..2π, phi: 0..π)
    theta = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=True)
    phi   = np.linspace(0.0, np.pi, n_phi, endpoint=True)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi),   np.sin(phi)

    # Parametric sphere -> ellipsoid: μ + V diag(axes) [x_sphere]
    Xs = np.outer(ct, sp)  # (n_theta, n_phi)
    Ys = np.outer(st, sp)
    Zs = np.outer(np.ones_like(theta), cp)

    # Stack and scale/rotate
    S = np.stack([Xs, Ys, Zs], axis=0).reshape(3, -1)          # (3, n_theta*n_phi)
    E = (V @ (axes[:, None] * S)) + center[:, None]            # (3, n_pts)
    X = E[0].reshape(n_theta, n_phi)
    Y = E[1].reshape(n_theta, n_phi)
    Z = E[2].reshape(n_theta, n_phi)

    return center, (X, Y, Z), tuple(axes.tolist()), V
