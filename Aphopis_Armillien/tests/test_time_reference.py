import pytest
import numpy as np
from Aphopis_Armillien.utils.time_reference import J0, zeroTo360, LST, equatorial_to_eclipitcJ2000, create_da_los_vectors, CC2COE, CC2MEE, MEE2CC, COE2CC, ROT1, ROT3


def test_J0():
    """
    Tests the provided J0(y, m, d) routine.

    Typical-case numbers come from Curtis (Example 5.4):
        12 May 2004  →  J0 = 2 453 137.5

    The tolerance is absolute, since the reference value is exact
    for the algorithm’s integer-arithmetic formulation.
    """

    # ---------------------------
    # 1. Typical (worked-example)
    # ---------------------------
    ref = 2_453_137.5          # Example 5.4 value (p. 215)
    calc = J0(2004, 5, 12)
    assert np.isclose(calc, ref, rtol=1e-6), (
        f"Expected {ref}, got {calc}"
    )

    # ----------------
    # 2. Return type
    # ----------------
    assert isinstance(calc, float), "J0 must return a float"

    # ------------------------------
    # 3. Boundary / error handling
    # ------------------------------
    with pytest.raises(AssertionError):
        J0(1899, 1, 1)          # year below lower bound
    with pytest.raises(AssertionError):
        J0(2100, 1, 1)          # year above upper bound
    with pytest.raises(AssertionError):
        J0(2025, 13, 1)         # invalid month
    with pytest.raises(AssertionError):
        J0(2025, 2, 32)         # invalid day


def test_zeroTo360():
    assert zeroTo360(0) == 0
    assert zeroTo360(360) == 360
    assert zeroTo360(720) == 360
    assert zeroTo360(-360) == 0
    assert zeroTo360(-720) == 0
    assert zeroTo360(-90) == 270
    assert zeroTo360(450) == 90


def test_LST():
    """
    Functional, boundary, and typical-case checks for LST().

    Typical-case data (Curtis Example 5.6):
        Date  : 3 Mar 2004
        UT    : 04 h 30 m 00 s
        EL    : 139° 47′ 00″  → 139.783333… deg
        Expected LST = 8.57688 deg
    """

    # ---------------------------
    # 1. Typical-case inputs
    # ---------------------------
    y, m, d = 2004, 3, 3
    ut      = (4, 30, 0)                     # (h, m, s)
    EL_deg  = 139 + 47/60 + 0/3600           # 139.783333…
    lst_exp = 8.57688                        # deg  :contentReference[oaicite:2]{index=2}

    # -------------------------------
    # 2. Call the function under test
    # -------------------------------
    lst_calc, epoch_calc = LST(y, m, d, ut, EL_deg)

    # -----------------------------------------------
    # 3. Numerical accuracy vs. worked example
    # -----------------------------------------------
    assert np.isclose(lst_calc, lst_exp, rtol=1e-6), (
        f"LST mismatch: expected {lst_exp}, got {lst_calc}"
    )

    # Verify the Julian epoch is J0 + UT/24
    UT_hours      = ut[0] + ut[1]/60 + ut[2]/3600
    j0            = J0(y, m, d)
    epoch_expected = j0 + UT_hours/24
    assert np.isclose(epoch_calc, epoch_expected, rtol=1e-10), (
        "Returned epoch is incorrect"
    )

    # --------------------------------------
    # 4. Types and value-range guarantees
    # --------------------------------------
    assert isinstance(lst_calc, float) and isinstance(epoch_calc, float)
    assert 0.0 <= lst_calc < 360.0, "LST must lie in [0°, 360°)"

    # -------------------------------------------------------
    # 5. Simple boundary / defensive-programming check
    # -------------------------------------------------------
    with pytest.raises(AssertionError):
        # Year below supported range should propagate J0()’s assertion
        LST(1899, 1, 1, (0, 0, 0), 0.0)

def test_equatorial_to_eclipitcJ2000():
    """
    Numerical, type/shape, and geometric checks for the
    equatorial→ecliptic rotation routine.

    Typical-case: Tokyo (lat = 35°40 N) on 3 Mar 2004 04:30 UT
                  LST = 8.59° (Curtis Ex. 5.6)
    """

    # --- 1. Build the worked-example inputs -----------------
    # Site latitude (φ) in decimal degrees
    lat_deg  = 35 + 40/60          # 35.6667°
    lat_rad  = np.deg2rad(lat_deg)

    # Spherical-earth radius (Curtis uses 6378 km in worked examples)
    Re_km    = 6378.0

    # Components relative to the earth’s rotation axis / equator
    dr       = Re_km * np.cos(lat_rad)      # km
    h_e      = Re_km * np.sin(lat_rad)      # km

    # Local sidereal time from the example (degrees → radians)
    lst_deg  = 8.59
    lst_rad  = np.deg2rad(lst_deg)

    # --- 2.  “Ground-truth” computation (independent) -------
    # Equatorial vector of the site
    r_eq = np.array([dr * np.cos(lst_rad),
                     dr * np.sin(lst_rad),
                     h_e])

    # Obliquity rotation (about +x axis) with ε = 23.4°
    eps      = np.deg2rad(23.4)
    R_x = np.array([[1,            0,           0],
                    [0,  np.cos(eps), np.sin(eps)],
                    [0, -np.sin(eps), np.cos(eps)]])

    r_expected = R_x @ r_eq        # ecliptic-frame vector (km)

    # --- 3. Call the user function --------------------------
    r_calc = equatorial_to_eclipitcJ2000(dr, lst_rad, h_e)

    # --- 4. Accuracy: element-wise vector comparison --------
    assert np.allclose(r_calc, r_expected, rtol=1e-6), (
        f"Vector mismatch:\nexpected {r_expected}\nactual   {r_calc}"
    )

    # --- 5. Type, shape, and invariants ---------------------
    assert isinstance(r_calc, np.ndarray), "Return type must be np.ndarray"
    assert r_calc.shape == (3,),          "Return shape must be (3,)"
    # Rotation should preserve magnitude, and x-component is invariant
    assert np.isclose(np.linalg.norm(r_calc),
                      np.linalg.norm(r_eq),
                      rtol=1e-10), "Rotation must preserve vector length"
    assert np.isclose(r_calc[0], r_eq[0], rtol=1e-6), (
        "x-component should remain unchanged by x-axis rotation"
    )

def test_create_da_los_vectors():
    """
    Accuracy, shape/type, unit-length, and assertion checks for the
    line-of-sight vector generator.

    Typical-case numbers come from Curtis Example 5.11 (Table 5.1).
    """

    # -------------------------------------------------
    # 1. Typical-case inputs (degrees ➜ radians)
    # -------------------------------------------------
    ra_deg  = np.array([43.537, 54.420, 64.318])   # 1×N row
    dec_deg = np.array([-8.7833, -12.074, -15.105])
    ra_rad  = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    # -------------------------------------------------
    # 2. Expected result from Eq. 5.57 (independent)
    # -------------------------------------------------
    expected = np.array([[0.716428, 0.568968, 0.418403],
                        [0.680745, 0.795312, 0.870076],
                        [-0.152698, -0.209175, -0.260589]])
    # -------------------------------------------------
    # 3. Call the function under test
    # -------------------------------------------------
    los = create_da_los_vectors(ra_rad, dec_rad)

    # -------------------------------------------------
    # 4. Numerical accuracy (element-wise)
    # -------------------------------------------------
    assert np.allclose(los, expected, atol=1e-4), (
        f"LOS vectors incorrect:\nexpected\n{expected}\nactual\n{los}"
    )

    # -------------------------------------------------
    # 5. Type, shape, and unit-length checks
    # -------------------------------------------------
    assert isinstance(los, np.ndarray),  "Return type must be np.ndarray"
    assert los.shape == (3, 3),          "Return shape must be (3, N)"
    norms = np.linalg.norm(los.astype(float), axis=0)
    assert np.allclose(norms, 1.0, rtol=1e-10), "Each LOS vector must be unit length"


def test_ROT1_ROT3():
    #based on curtis example 4.1
    
    RAAN = np.deg2rad(40)
    argp = np.deg2rad(60)
    inc = np.deg2rad(30)

    ROT  = ROT3(argp) @ ROT1(inc) @ ROT3(RAAN)
    
    answer = np.zeros((3,3))
    answer[0,0] = -0.099068
    answer[1,0] = 0.89593
    answer[2,0] = 0.43301
    answer[0,1] =  -0.94175
    answer[1,1] = -0.22496
    answer[2,1] = 0.25
    answer[0,2] = 0.32139
    answer[1,2] = -0.38302
    answer[2,2] = 0.86603

    answer = answer.T

    assert np.allclose(ROT, answer, atol=1e-3), (
        f"ROT mismatch: expected\n{answer},\ngot\n{ROT}"
    )


def test_CC2COE():
    #Test from Cutis ex 4.3

    r = np.array([-6045.0, -3490.0, 2500.0])
    v = np.array([-3.457, 6.618, 2.533])

    COE = CC2COE(r, v, mu=398600.4418)
    
    COE_expected = np.array([8788.0, 0.1712, np.deg2rad(153.2), np.deg2rad(255.3), np.deg2rad(20.07), np.deg2rad(28.45)])
    
    assert np.allclose(COE, COE_expected, rtol=1e-3), (f"CC2COE mismatch:\nexpected\n{COE_expected}\nactual\n{COE}")

def test_COE2CC():
    # Test from Curtis Ex 4.4

    COE = np.array([16725.20488, 1.4, np.deg2rad(30), np.deg2rad(40), np.deg2rad(60), np.deg2rad(30)])
    r, v = COE2CC(COE, mu=398600.4418)

    r_expected = np.array([-4040, 4815, 3629])
    v_expected = np.array([-10.39, -4.772, 1.744])
    
    assert np.allclose(r, r_expected, rtol=1e-3), (
        f"Position vector mismatch: expected {r_expected}, got {r}"
    )
    assert np.allclose(v, v_expected, rtol=1e-3), (
        f"Velocity vector mismatch: expected {v_expected}, got {v}"
    )

def test_CC2MEE():
    # Test from https://ai-solutions.com/_help_Files/orbit_element_types.htm#achr_modifiedequinoctial 
    r = np.array([-3410.673, 5950.957, -1788.627])
    v = np.array([1.893, -1.071, -7.176])

    MEE = CC2MEE(r, v, mu=398600.4418)
    MEE_expected = np.array([7070.766, 0.00180, -0.00170, 0.610, -0.980, 136.64])

    assert np.allclose(MEE, MEE_expected, rtol=1e-6), (
        f"CC2MEE mismatch: expected\n{MEE_expected}\n, got \n{MEE}\n"
    )

def test_MEE2CC():
    # Test from https://ai-solutions.com/_help_Files/orbit_element_types.htm#achr_modifiedequinoctial 
    
    MEE = np.array([7070.766, 0.00180, -0.00170, 0.610, -0.980, 136.64])
    r, v = MEE2CC(MEE, mu=398600.4418)

    r_expected = np.array([-3410.673, 5950.957, -1788.627])
    v_expected = np.array([1.893, -1.071, -7.176])

    assert np.allclose(r, r_expected, rtol=1e-6), (
        f"Position vector mismatch: expected {r_expected}, got {r}"
    )
    assert np.allclose(v, v_expected, rtol=1e-6), (
        f"Velocity vector mismatch: expected {v_expected}, got {v}"
    )
    # Note: The above tests assume the mu value is the gravitational parameter for Earth.

