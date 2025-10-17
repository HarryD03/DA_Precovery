import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u

# --- Astronomical and Physical Constants from the Paper ---
# These constants are crucial for the energy calculations.
K_GAUSS = 0.01720209895      # Gauss's gravitational constant (AU, M_sun, day)
MU_EARTH = 2.664476e-8       # Mass of Earth / Mass of Sun
R_SI = 0.010044              # Earth's Sphere of Influence in AU

def compute_attributable(obs_times, ras, decs):
    """
    Step 1: Computes the attributable from raw observation data.
    This function condenses multiple observations into a single state vector
    representing the object's apparent motion at a reference time.

    Args:
        obs_times (list of str): Observation times in ISO format (e.g., '2025-08-18T20:00:00').
        ras (list of float): Right Ascensions in degrees.
        decs (list of float): Declinations in degrees.

    Returns:
        dict: The attributable vector containing reference time, mean position, and rates.
    """
    t_astropy = Time(obs_times, format='isot')
    t_jd = t_astropy.jd
    
    # Perform linear regression to find the rates of change in RA and Dec.
    t_rel_days = t_jd - t_jd[0]
    ra_rad = np.deg2rad(ras)
    dec_rad = np.deg2rad(decs)

    # A more robust fit would account for spherical coordinates, but linear is sufficient for a VSA.
    ra_fit = linregress(t_rel_days, ra_rad)
    dec_fit = linregress(t_rel_days, dec_rad)

    # The reference time is the mean of the observation times.
    t_ref_jd = np.mean(t_jd)
    t_ref_astropy = Time(t_ref_jd, format='jd')
    t_ref_rel = t_ref_jd - t_jd[0]

    attributable = {
        't_ref_astropy': t_ref_astropy,
        'epsilon': ra_fit.intercept + ra_fit.slope * t_ref_rel, # Mean RA in radians
        'theta': dec_fit.intercept + dec_fit.slope * t_ref_rel, # Mean Dec in radians
        'eps_dot': ra_fit.slope,  # rad/day
        'theta_dot': dec_fit.slope, # rad/day
    }
    return attributable

def calculate_coefficients(attributable):
    """
    Step 2: Calculates the c-coefficients based on the attributable and Earth's state vector.
    These coefficients are the building blocks for the energy equations.
    """
    t_ref = attributable['t_ref_astropy']
    eps = attributable['epsilon']
    theta = attributable['theta']
    eps_dot = attributable['eps_dot']
    theta_dot = attributable['theta_dot']

    # Get Earth's and Sun's barycentric state vectors using Astropy's built-in ephemeris.
    # This replaces the need for an external SPK file.
    earth_posvel = get_body_barycentric_posvel('earth', t_ref)
    sun_posvel = get_body_barycentric_posvel('sun', t_ref)
    
    # Extract position and velocity, converting to the required units (AU and AU/day).
    pos_emb = earth_posvel[0].xyz.to(u.au).value
    vel_emb = earth_posvel[1].xyz.to(u.au / u.day).value

    pos_sun = sun_posvel[0].xyz.to(u.au).value
    vel_sun = sun_posvel[1].xyz.to(u.au / u.day).value

    # Heliocentric position of Earth (observer's position).
    P_oplus = (pos_emb - pos_sun)
    P_dot_oplus = (vel_emb - vel_sun)

    # Define the observation basis vectors (line-of-sight and sky plane).
    r_hat = np.array([np.cos(eps) * np.cos(theta),
                      np.sin(eps) * np.cos(theta),
                      np.sin(theta)])
    
    r_eps_hat = np.array([-np.sin(eps) * np.cos(theta),
                           np.cos(eps) * np.cos(theta),
                           0])

    r_theta_hat = np.array([-np.cos(eps) * np.sin(theta),
                            -np.sin(eps) * np.sin(theta),
                             np.cos(theta)])

    # Calculate c-coefficients as defined in the paper's equations.
    coeffs = {}
    coeffs['c0'] = np.dot(P_oplus, P_oplus)
    coeffs['c1'] = 2 * np.dot(P_dot_oplus, r_hat)
    coeffs['c2'] = eps_dot**2 * np.cos(theta)**2 + theta_dot**2 # This is eta^2
    
    c3_1 = 2 * np.dot(P_dot_oplus, r_eps_hat)
    c3_2 = 2 * np.dot(P_dot_oplus, r_theta_hat)
    coeffs['c3'] = eps_dot * c3_1 + theta_dot * c3_2

    coeffs['c4'] = np.dot(P_dot_oplus, P_dot_oplus)
    coeffs['c5'] = 2 * np.dot(P_oplus, r_hat)
    
    return coeffs, attributable

def construct_admissible_region(obs_data, h_mag, H_max=30.0):
    """
    Main function to construct and plot the admissible region.
    This orchestrates all steps of the methodology.
    """
    # Steps 1 & 2: Get attributable and coefficients
    attributable = compute_attributable(obs_data['times'], obs_data['ras'], obs_data['decs'])
    coeffs, attributable = calculate_coefficients(attributable)

    r_plot_range = np.linspace(0.0001, 1.5, 100000) # A dense range of r for plotting

    # --- Step 3: Establish and Plot All Boundaries ---

    plt.figure(figsize=(14, 9))
    
    # 1. Inner Boundary (Magnitude Limit)
    # This sets a minimum distance based on the object's brightness and assumed size.
    # H = h - 5*log10(r) (simplified, ignoring phase effects x0)
    r_H = 10**((h_mag - H_max) / 5.0)
    plt.axvline(x=r_H, color='green', linestyle='-.', lw=2, label=f'Inner Boundary (H_max={H_max}, r_H={r_H:.3f} AU)')
    plt.axvline(x=R_SI, color='red', linestyle='-.', lw=2, label=f'SoI Boundary (R_SI={R_SI:.3f} AU)')
    # 2. Geocentric Boundary
    # This boundary excludes objects that are satellites of Earth.
    eta_sq = coeffs['c2']
    G_r = (2 * K_GAUSS**2 * MU_EARTH) / r_plot_range - eta_sq * r_plot_range**2
    G_r[G_r < 0] = np.nan # The curve only exists where G(r) is non-negative
    r_dot_G = np.sqrt(G_r)
    
    
    plt.plot(r_plot_range, r_dot_G, 'b--', lw=2, label='Geocentric Boundary ($\mathcal{E}_{\oplus} = 0$)')
    plt.plot(r_plot_range, -r_dot_G, 'b--', lw=2)

    # 3. Heliocentric Boundary (The most complex part)
    # 3a. Construct and solve the V(r) polynomial to find the valid range interval.
    c0, c1, c2, c3, c4, c5 = coeffs.values()
    gamma = c4 - c1**2 / 4.0
    
    A = np.zeros(7) # Coefficients for the 6th-degree polynomial V(r)
    A[6] = c2**2
    A[5] = c2 * (2*c3 + c2*c5)
    A[4] = c3**2 + 2*c2*gamma + 2*c2*c3*c5 + c0*c2**2
    A[3] = 2*c3*gamma + c5*(c3**2 + 2*c2*gamma) + 2*c0*c2*c3
    A[2] = gamma**2 + 2*c3*c5*gamma + c0*(c3**2 + 2*c2*gamma)
    A[1] = c5*gamma**2 + 2*c0*c3*gamma
    A[0] = c0*gamma**2

    V_poly = np.polynomial.Polynomial(A) # numpy expects coefficients from low to high degree
    
    # The boundary condition is V(r) <= 4k^4
    limit = 4 * K_GAUSS**4
    roots = (V_poly - limit).roots()
    
    # Filter for real, positive roots which define the interval boundaries.
    real_positive_roots = sorted([r.real for r in roots if np.isreal(r) and r.real > r_H])
    
    if not real_positive_roots:
        print("Could not determine a clear valid range interval from V(r) analysis.")
        # Attempt to find a valid region by testing points
        test_r = np.linspace(r_H, 5, 200)
        v_vals = V_poly(test_r)
        valid_mask = v_vals <= limit
        if not np.any(valid_mask):
            print("No valid region found.")
            return
    
    # 3b. Trace the Heliocentric Boundary curve over the valid range.
    # We solve the quadratic energy equation for r_dot for each valid r.
    S_r = r_plot_range**2 + c5*r_plot_range + c0
    C_r = c2*r_plot_range**2 + c3*r_plot_range + c4
    
    discriminant_h = c1**2 - 4*(C_r - 2*K_GAUSS**2 / np.sqrt(S_r))
    discriminant_h[discriminant_h < 0] = np.nan # No real solution for r_dot
    
    r_dot_h1 = (-c1 + np.sqrt(discriminant_h)) / 2.0
    r_dot_h2 = (-c1 - np.sqrt(discriminant_h)) / 2.0

    # Mask out the regions that are not valid according to V(r)
    v_values = V_poly(r_plot_range)
    invalid_mask = (v_values > limit) | (r_plot_range < r_H)
    r_dot_h1[invalid_mask] = np.nan
    r_dot_h2[invalid_mask] = np.nan

    # Plot the final boundary and fill the admissible region
    plt.plot(r_plot_range, r_dot_h1, 'r-', lw=2, label='Heliocentric Boundary ($\mathcal{E}_{\odot} = 0$)')
    plt.plot(r_plot_range, r_dot_h2, 'r-', lw=2)
    
    # --- Correctly Fill the Admissible Region ---
    # The admissible region is INSIDE the red curve AND OUTSIDE the blue curve.
    # We fill the area between the upper red and upper blue curves.
    plt.fill_between(r_plot_range, r_dot_h1, r_dot_G, where=(r_dot_h1 > r_dot_G), color='orange', alpha=0.5, label='Admissible Region')
    # We fill the area between the lower red and lower blue curves.
    plt.fill_between(r_plot_range, r_dot_h2, -r_dot_G, where=(r_dot_h2 < -r_dot_G), color='orange', alpha=0.5)

    # --- Step 4: Final Plot Formatting ---
    plt.title('Admissible Region in (r, $\dot{r}$) Plane', fontsize=16)
    plt.xlabel('Geocentric Range, r (AU)', fontsize=12)
    plt.ylabel('Geocentric Range-Rate, $\dot{r}$ (AU/day)', fontsize=12)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Set plot limits dynamically based on the solution
    valid_r = r_plot_range[~invalid_mask]
    if len(valid_r) > 0:
        plt.xlim(0, valid_r.max() * 1.1)
    
    valid_y = np.concatenate([r_dot_h1, r_dot_h2])
    valid_y = valid_y[~np.isnan(valid_y)]
    if len(valid_y) > 0:
        y_max = np.max(np.abs(valid_y))
        plt.ylim(-y_max * 1.2, y_max * 1.2)
    else:
        plt.ylim(-0.1, 0.1) # Default if no solution found
    
    plt.show()

def initiate_admissible_region(RA, DEC, obs_times, apparent_magnitude):
    """
        Do admissible region
    """
    obs = {
        'times': obs_times,
        'ras': RA,
        'decs': DEC
    }

    construct_admissible_region(obs, apparent_magnitude)

