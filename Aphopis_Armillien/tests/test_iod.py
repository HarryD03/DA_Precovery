import numpy as np
import pytest
from Aphopis_Armillien.utils.iod import Guass_8th_seed, f_g_series
import numpy as np
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray


# Textbook Example - Performance test
def test_Guass_8th_seed_basic():
    # Use simple, linearly independent vectors for observer positions and directions
    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          # Observer position [x components at all instances]
        [3430.2, 3460.1, 3490.1],          # Observer position [y components at all instances]
        [4078.5, 4078.5, 4078.5]           # Observer position [z components at all instances]
    ])

    obs_dir = np.array([            # Direction vectors of the observations
        [0.71643, 0.56897, 0.41841],            # Direction vector for x components
        [0.68074, 0.79531, 0.87007],            # Direction vector for y components
        [-0.15270,-0.20917,-0.26059]            # Direction vector for z components
    ])

    t = np.array([0.0, 118.10, 237.58])


    # Should not raise and should return two lists of length 3
    position, ranges = Guass_8th_seed(pos_obs, obs_dir, t, mu=3.986e5)
    r1_textbook_ex = np.array([
        6096.9,
        5907.5,
        3522.9])
    
    r2_textbook_ex = np.array([
        5659.1,
        6533.8,
        3270.1])
    
    r3_textbook_ex = np.array([
        5169.1,
        7107.0,
        2995.3])
    
    assert isinstance(position, np.ndarray)
    assert isinstance(ranges, np.ndarray)
    assert len(position) == 3
    assert len(ranges) == 3

    assert np.allclose(position[:,1], r2_textbook_ex, rtol=1e-3)
    assert np.allclose(position[:,0], r1_textbook_ex, rtol=1e-3)
    assert np.allclose(position[:,2], r3_textbook_ex, rtol=1e-3)

# Functional Example Test
def test_Guass_8th_seed_invalid_coeffs():
    # Use vectors that will produce NaN or inf in coefficients
    pos_obs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    obs_dir = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    t = np.array([0.0, 0.0, 0.0])

    with pytest.raises(AssertionError):
        Guass_8th_seed(pos_obs, obs_dir, t)

def test_Guass_8th_seed_no_real_roots():
    # Use vectors that will produce no positive real roots
    pos_obs = np.array([
        [-1.0e8, 0.0, 0.0],
        [0.0, -1.0e8, 0.0],
        [0.0, 0.0, -1.0e8]
    ])
    obs_dir = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    t = np.array([0.0, 1.0e4, 2.0e4])

    with pytest.raises(AssertionError):
        Guass_8th_seed(pos_obs, obs_dir, t)


# Functional tests 
def test_f_g_series_v0_zero_vector():
    # Provide a v0 that is a zero vector - see if works with no velcoity
    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.zeros(3)  # Zero vector
    dt = 1000.0
    f_order = 4
    g_order = 4
    f, g = f_g_series(r0, dt, f_order, g_order, v0=v0)
    # Should still return valid f and g values
    assert isinstance(f, float)
    assert isinstance(g, float)

def test_f_g_series_v0_nonzero_vector():
    # Provide a nonzero v0
    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.array([1.0e3, 2.0e3, 3.0e3])  # Nonzero vector
    dt = 1000.0
    f_order = 5
    g_order = 5
    f, g = f_g_series(r0, dt, f_order, g_order, v0=v0)
    assert isinstance(f, Union[float,int])
    assert isinstance(g, Union[float,int])

def test_f_g_series_v0_invalid_shape():
    # Provide a v0 that is not a 3D vector

    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.array([1.0, 2.0])  # Invalid shape
    dt = 1000.0
    with pytest.raises(AssertionError):
        f_g_series(r0, dt, 3, 3, v0=v0)

def test_f_g_series_IOD():
    # Provide no v0 vector - see if v0 is None part works.
    r0 = np.array([1.0e8, 0.0, 0.0])
    dt = 1000.0
    f_order = 2
    g_order = 3
    f, g = f_g_series(r0, dt, f_order, g_order)

    # Should still return valid f and g values
    assert isinstance(f, float)
    assert isinstance(g, float)

# ---------------------------------------------------------------------
#Performance tests (textbook examples)

def test_f_g_series_IOD_typical():
    # Textbook example with typical values - Orbital mechancis for Engineering Students Ex: 5: Guass example
    # IOD as no v0 is provided
    r0 = np.array([5659.1, 6533.8, 3270.1])
    dt = -118.10
    f_order = 2
    g_order = 3
    f, g = f_g_series(r0, dt, f_order, g_order, mu=3.986e5)
    # Should return valid f and g values
    # Replace 1.234 with the textbook value you want to check against
    f_textbook_value = 0.99648
    g_textbook_value = -117.97
    assert np.isclose(f, f_textbook_value, rtol=1e-3), f"Expected {f_textbook_value}, got {f}"
    assert np.isclose(g, g_textbook_value, rtol=1e-3), f"Expected {g_textbook_value}, got {g}"