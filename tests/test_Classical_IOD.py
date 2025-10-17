import utils.propagation as prop
import utils.iod as iod
import utils.dynamics as dynamics
from utils.lambert_izzo import lambert_izzo
import utils.admissable_region as ar
import utils.Classical_IOD as PW_IOD
import utils.time_reference as time_ref
from typing import Callable, List, Union, overload, Tuple
import time
import numpy as np 
from numpy.typing import NDArray
from matplotlib import pyplot as plt


def test_monte_carlo_gauss_PWiod():
    # Test parameters from Vallado p. 447
    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          # Observer position [x components at all instances]
        [3430.2, 3460.1, 3490.1],          # Observer position [y components at all instances]
        [4078.5, 4078.5, 4078.5]           # Observer position [z components at all instances]
    ])

    t = np.array([0.0, 118.10, 237.58])

    RA = np.deg2rad(np.array([43.537, 54.420, 64.318]))
    DEC = np.deg2rad(np.array([-8.7833, -12.074, -15.105]))

    i_rho_nom = time_ref.create_da_los_vectors(RA, DEC)

    r, _, _, v = iod.Guass_8th_seed(pos_obs, i_rho_nom, t, mu=3.968e5)
    results = PW_IOD.monte_carlo_gauss_PWiod(100, pos_obs, RA, DEC, t, mu=3.968e5)


    X_nom = np.concatenate((r[:,1], v))
    X = results
    COE = np.zeros((X.shape[0], 6))
    for i in range(X.shape[0]):
        COE[i]  = time_ref.CC2COE(X[i,:3], X[i,3:], 3.968e5)

    COE_nom = time_ref.CC2COE(X_nom[:3], X_nom[3:], 3.968e5)
    perigee_nom = COE_nom[0] * (1 - COE_nom[1])
    perigee = COE[:,0] * (1 - COE[:,1])

    plt.figure()
    plt.plot(perigee, COE[:,1], 'o', color='blue')
    plt.plot(perigee_nom, COE_nom[1], 'x', color='red', markersize=10)
    plt.show()
    assert np.mean(perigee) < perigee_nom * 1.01 and np.mean(perigee) > perigee_nom * 0.99


import numpy as np

def state_to_orbital_frame(state_vector, mu, eps=1e-12):
    """
    Rotate an inertial 6D state [x,y,z,vx,vy,vz] into the orbital (perifocal-like) frame: (primarly for plotting)
      - x̂ points to periapsis (eccentricity direction, if defined; else radial r̂)
      - ŷ completes the right-handed in-plane basis
      - ẑ is the orbit normal ĥ = (r×v)/|r×v|

    Returns:
      rotated_state: 6-vector in orbital frame
      R: 3×3 rotation matrix (inertial -> orbital)
    """
    r = np.asarray(state_vector[:3], dtype=float)
    v = np.asarray(state_vector[3:6], dtype=float)

    r_norm = np.linalg.norm(r)
    if r_norm < eps:
        raise ValueError("Position norm is ~0; cannot define orbital plane.")

    # Specific angular momentum (orbit normal)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < eps:
        raise ValueError("h ≈ 0 (radial or undefined trajectory); cannot define orbital plane.")

    k_hat = h / h_norm  # ẑ along orbit normal

    # Eccentricity vector (periapsis direction)
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e_norm = np.linalg.norm(e_vec)

    if e_norm > 1e-10:
        i_hat = e_vec / e_norm          # x̂ to periapsis
    else:
        i_hat = r / r_norm              # circular: use instantaneous radial direction

    j_hat = np.cross(k_hat, i_hat)      # ŷ completes right-handed triad
    j_hat /= np.linalg.norm(j_hat)

    # Re-orthogonalize i_hat just in case (numerical hygiene)
    i_hat = np.cross(j_hat, k_hat)

    # Rotation from inertial to orbital: rows are new-basis unit vectors in the old basis
    R = np.vstack((i_hat, j_hat, k_hat))

    r_orb = R @ r
    v_orb = R @ v
    return np.concatenate([r_orb, v_orb]), R



def test_monte_carlo_propagation_PWiod():
    """
    Test the Monte Carlo propagation for the PW IOD method.
    """

    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          # Observer position [x components at all instances]
        [3430.2, 3460.1, 3490.1],          # Observer position [y components at all instances]
        [4078.5, 4078.5, 4078.5]           # Observer position [z components at all instances]
    ])

    t = np.array([0.0, 118.10, 237.58])

    RA = np.deg2rad(np.array([43.537, 54.420, 64.318]))
    DEC = np.deg2rad(np.array([-8.7833, -12.074, -15.105]))

    i_rho_nom = time_ref.create_da_los_vectors(RA, DEC)

    r, _, _, v = iod.Guass_8th_seed(pos_obs, i_rho_nom, t, mu=3.968e5)
    results = PW_IOD.monte_carlo_gauss_PWiod(100, pos_obs, RA, DEC, t, mu=3.968e5)

    X_nom = np.concatenate((r[:,1], v))

    COE_nom = time_ref.CC2COE(X_nom[:3], X_nom[3:], 3.968e5)
    mu = 3.985e5
    ## Propagate the results to a future time
    t0 = 0.0 
    tf = 2*np.pi * np.sqrt(COE_nom[0]**3 / mu )

    propagated_results, tgrid = PW_IOD.monte_carlo_propagation(results, t0, tf, Ts=50, mu=3.986e5)

    sma_values = []
    orbital_trajectories = []

    # Analyze the propagated results
    for i in range(propagated_results.shape[0]):        # Loop through each starting condiitions
        initial_state = propagated_results[i, :, 0]
        coe = time_ref.CC2COE(initial_state[:3], initial_state[3:], mu)
        sma = coe[0]
        sma_values.append(sma)


        positions = propagated_results[i, :3, :].T
        velocities = propagated_results[i, 3:, :].T

        x_orbital = []
        y_orbital = []

        for j in range(positions.shape[0]-1):
            state = np.concatenate([positions[j], velocities[j]])
            orbtial_state, _ = state_to_orbital_frame(state, mu)
            x_orbital.append(orbtial_state[0])
            y_orbital.append(orbtial_state[1])

        orbital_trajectories.append((x_orbital, y_orbital, sma))


    sma_values = np.array(sma_values)
    outermost_idx = np.argmax(sma_values)
    innermost_idx = np.argmin(sma_values)

    plt.figure(figsize=(10, 8))
    x_outer, y_outer, sma_outer = orbital_trajectories[outermost_idx]

    plt.plot(x_outer, y_outer, 'r--', linewidth=1, alpha=0.8, 
             label=f'Outermost (SMA={sma_outer:.1f} km)')
    
    # Plot innermost orbit
    x_inner, y_inner, sma_inner = orbital_trajectories[innermost_idx]
    plt.plot(x_inner, y_inner, 'b--', linewidth=1, alpha=0.8, 
             label=f'Innermost (SMA={sma_inner:.1f} km)')
    

    plt.xlabel('X (km) - Periapsis Direction')
    plt.ylabel('Y (km) - In Orbital Plane')
    plt.title('Monte Carlo Propagation - Orbital Plane Evolution')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.show()
    print("Monte Carlo Propagation - Orbital Plane Evolution")