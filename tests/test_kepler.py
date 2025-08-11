import numpy as np
import pytest
from utils.iod import Kepler_DA, lagrange_coefficients, kepler_F
import numpy as np
from daceypy import DA, array
import daceypy.op as op

# --- CONSTANTS -------------------------------------------------------------
MU_EARTH = 3.986004418e5  # km^3 s^-2  (CODATA2014, ≈ Curtis Table A-1)

# Helper: create canonical circular LEO state (7000 km altitude above Earth's centre)
def _leo_state(radius_m: float, mu: float = MU_EARTH):
    r0 = np.array([radius_m, 0.0, 0.0])           # position on +x
    v_mag = np.sqrt(mu / radius_m)                # circular speed
    v0 = np.array([0.0,  v_mag, 0.0])             # velocity on +y (pro-grade)
    return r0, v0

# ---------------------------------------------------------------------------

MU_EARTH_KM = 3.986e5              # km³ s⁻²  (Curtis, Tbl A-2)

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


# ---------------------------------------------------------------------------
@pytest.mark.basic
def test_kepler_da_basic_interface_km():
    """Test Taken from Vallado p.95 Example 2.4."""
    r0, v0, r1_textbook, v1_textbook = _leo_state_km()


    #Initalise DA for r1 and v1 as DA variables
    DA.init(4,7)
    r0 = array([r0[i] + DA(i + 1) for i in range(3)])
    v0 = array([v0[i] + DA(i + 4) for i in range(3)])

    dt = 40*60
    r1, v1 = Kepler_DA(r0, v0, dt, mu=MU_EARTH_KM)
    # type & shape
    assert isinstance(r1.cons(), np.ndarray) and r1.cons().shape == (3,)
    assert isinstance(v1.cons(), np.ndarray) and v1.cons().shape == (3,)

    assert isinstance(r1, array)
    assert isinstance(v1, array)

    # magnitudes sensible
    assert np.isclose(op.vnorm(r1).cons(), op.vnorm(r1_textbook), atol=1e-9), f"Expect {op.vnorm(r1_textbook)} obtained {op.vnorm(r1).cons()}\n"        # >1 000 km from Earth-centre
    assert np.isclose(op.vnorm(v1).cons(), op.vnorm(v1_textbook), atol=1e-9), f"Circular Orbit so Radius should be unchanged"        # >1 000 km from Earth-centre
    assert np.isfinite(r1.cons()).all() and np.isfinite(v1.cons()).all()

# ---------------------------------------------------------------------------
@pytest.mark.regression
def test_kepler_da_full_orbit_regression_km():
    """
    Regression: propagate one full circular revolution and
    expect metre-level agreement (1 m ≈ 1e-3 km; 1 mm s⁻¹ ≈ 1e-6 km s⁻¹).
    """
    r0, v0, _, _ = _leo_state_km()
    DA.init(4,7)

    r0 = array([r0[i] + DA(i + 1) for i in range(3)])
    v0 = array([v0[i] + DA(i + 4) for i in range(3)])
    # Period T = 2π √(a³/μ)  with  a = r for circular orbit

    energy = v0.dot(v0) / 2 - (MU_EARTH_KM / op.vnorm(r0))
    a = - MU_EARTH_KM / (2*energy)
    period_sec = 2.0*np.pi*np.sqrt(a.cons()**3 / MU_EARTH_KM)

    r1, v1 = Kepler_DA(r0, v0, period_sec, mu=MU_EARTH_KM)

    #Test Propagator functionally.
    pos_err_km  = op.vnorm(r1-r0)
    vel_err_kms = op.vnorm(v1-v0)

    tol_pos_km  = 1e-3     # 1 metre
    tol_vel_kms = 1e-6     # 1 mm/s

    assert pos_err_km.cons()  <= tol_pos_km,  \
        f"Position error {pos_err_km.cons():.3e} km exceeds {tol_pos_km:.1e} km"
    assert vel_err_kms.cons() <= tol_vel_kms, \
        f"Velocity error {vel_err_kms.cons():.3e} km/s exceeds {tol_vel_kms:.1e} km/s"
    
    
    # 1 min, 10 min

@pytest.mark.basic
def test_kepler_F_basic():
    """
    Test Automatic Kepler: Obtain nomial dE
    Inputs derived form Example 3.2 in Curtis 
        Assumes r1 is at perigee
    """
    DA.init(4,2)
    a = (9600.0 + 21000.0) / 2.0
    sigma = 0.0 
    r1_norm = 9600.0
    dM = 3.6029 + DA(1)
    expected_dE = 3.4794

    dE = kepler_F(a, sigma, r1_norm, dM)

    assert isinstance(dE, float)
    assert np.isclose(dE, expected_dE, atol = 1e-9), (
        f"ΔE {dE:0.7f} rad differs from Curtis Example 3.1 value "
        f"{expected_dE:0.4f} rad")    


@pytest.mark.basic
def test_lagrange_basic_interface():
        r0, v0 = _leo_state(7000.0)
        a  = np.linalg.norm(r0)        # circular ⇒ a = r
        # Eccentric-anomaly increment from the analytical short-arc formula
        dt = 10
        
        dE = np.sqrt(MU_EARTH_KM / a ** 3) * dt
        sigma = r0.dot(v0) / op.sqrt(MU_EARTH_KM)
        energy = v0.dot(v0) / 2 - (MU_EARTH_KM / op.vnorm(r0))  #Specific orbital energy
        SMA = - MU_EARTH_KM / (2 * energy)  #Semi-major axis
        
        r1, v1 = lagrange_coefficients(SMA, dE, r0, v0, sigma, MU_EARTH_KM)
       # --- type & shape ---
        assert isinstance(r1, np.ndarray) and r1.shape == (3,)
        assert isinstance(v1, np.ndarray) and v1.shape == (3,)

        # --- magnitudes sensible ---
        assert np.linalg.norm(r1) > 1_000.0            # km
        assert np.linalg.norm(v1) > 1.0                # km/s
        assert np.isfinite(r1).all() and np.isfinite(v1).all()


# ----------------------------------------------------------------------
#     # 2.  ONE-REVOLUTION REGRESSION
# ----------------------------------------------------------------------
@pytest.mark.regression
def test_lagrange_one_revolution():
        """
        Propagate a circular orbit for one full period
        and demand metre-level agreement.
        """
        r0, v0, = _leo_state(7000.0)

        # Universal-variable solution for ΔE over one revolution is 2π
        dE = 2.0 * np.pi
        #Prepare variables
        sigma = r0.dot(v0) / op.sqrt(MU_EARTH_KM)
        energy = v0.dot(v0) / 2 - (MU_EARTH_KM / op.vnorm(r0))  #Specific orbital energy
        SMA = - MU_EARTH_KM / (2 * energy)  #Semi-major axis
        
        r1, v1 = lagrange_coefficients(SMA, dE, r0, v0, sigma, MU_EARTH_KM)

        # Compare with initial state
        pos_err_km = np.linalg.norm(r1 - r0)
        vel_err_kms = np.linalg.norm(v1 - v0)

        # 1 m → 1e-3 km ; 1 mm/s → 1e-6 km/s
        assert pos_err_km  <= 1.0e-6,  f"pos err {pos_err_km:.2e} km"
        assert vel_err_kms <= 1.0e-6,  f"vel err {vel_err_kms:.2e} km/s"