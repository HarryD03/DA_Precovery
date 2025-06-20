import numpy as np
import pytest
from Aphopis_Armillien.utils.iod import Guass_8th_seed, f_g_series
import numpy as np
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray


# Mocking the DummyVec class to mimic the behavior of numpy arrays
class DummyVec:
    """Dummy vector class to mimic .dot and .cross for testing."""
    def __init__(self, arr):
        self.arr = np.asarray(arr)
    def dot(self, other):
        if isinstance(other, DummyVec):
            return float(np.dot(self.arr, other.arr))
        return float(np.dot(self.arr, other))
    def cross(self, other):
        if isinstance(other, DummyVec):
            return DummyVec(np.cross(self.arr, other.arr))
        return DummyVec(np.cross(self.arr, other))
    def __mul__(self, other):
        return DummyVec(self.arr * other)
    def __rmul__(self, other):
        return DummyVec(self.arr * other)
    def __add__(self, other):
        if isinstance(other, DummyVec):
            return DummyVec(self.arr + other.arr)
        return DummyVec(self.arr + other)
    def __sub__(self, other):
        if isinstance(other, DummyVec):
            return DummyVec(self.arr - other.arr)
        return DummyVec(self.arr - other)
    def __pow__(self, power):
        return DummyVec(self.arr ** power)
    def __repr__(self):
        return f"DummyVec({self.arr})"

def make_dummy_matrix(mat):
    """Convert a 2D numpy array to a matrix of DummyVecs (for .dot/.cross support)."""
    return np.array([[DummyVec(row) for row in mat]])

def test_Guass_8th_seed_basic():
    # Use simple, linearly independent vectors for observer positions and directions
    pos_obs = np.array([
        [1.0e8, 0.0, 0.0],          # Observer position [x components at all instances]
        [0.0, 1.0e8, 0.0],          # Observer position [y components at all instances]
        [0.0, 0.0, 1.0e8]           # Observer position [z components at all instances]
    ])
    obs_dir = np.array([            # Direction vectors of the observations
        [1.0, 0.0, 0.0],            # Direction vector for x components
        [0.0, 1.0, 0.0],            # Direction vector for y components
        [0.0, 0.0, 1.0]             # Direction vector for z components
    ])

    t = np.array([0.0, 1.0e4, 2.0e4])

    # Wrap as DummyVecs for .dot/.cross support
    pos_obs_dummy = np.array([DummyVec(row) for row in pos_obs])
    obs_dir_dummy = np.array([DummyVec(row) for row in obs_dir])

    # Should not raise and should return two lists of length 3
    position, ranges = Guass_8th_seed(pos_obs_dummy, obs_dir_dummy, t)
    assert isinstance(position, list)
    assert isinstance(ranges, list)
    assert len(position) == 3
    assert len(ranges) == 3

def test_Guass_8th_seed_multiple_roots(monkeypatch):
    # Use vectors that will likely produce multiple positive real roots
    pos_obs = np.array([
        [1.0e8, 0.0, 0.0],
        [0.0, 1.0e8, 0.0],
        [0.0, 0.0, 1.0e8]
    ])
    obs_dir = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0]
    ])
    t = np.array([0.0, 1.0e4, 2.0e4])

    pos_obs_dummy = np.array([DummyVec(row) for row in pos_obs])
    obs_dir_dummy = np.array([DummyVec(row) for row in obs_dir])

    # Patch print to suppress output
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    position, ranges = Guass_8th_seed(pos_obs_dummy, obs_dir_dummy, t)
    assert isinstance(position, list)
    assert isinstance(ranges, list)
    assert len(position) == 3
    assert len(ranges) == 3

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

    pos_obs_dummy = np.array([DummyVec(row) for row in pos_obs])
    obs_dir_dummy = np.array([DummyVec(row) for row in obs_dir])

    with pytest.raises(AssertionError):
        Guass_8th_seed(pos_obs_dummy, obs_dir_dummy, t)

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

    pos_obs_dummy = np.array([DummyVec(row) for row in pos_obs])
    obs_dir_dummy = np.array([DummyVec(row) for row in obs_dir])

    with pytest.raises(AssertionError):
        Guass_8th_seed(pos_obs_dummy, obs_dir_dummy, t)

        def test_f_g_series_basic():
            # Basic test with simple input, 2nd order expansion
            r0 = np.array([1.0e8, 0.0, 0.0])
            dt = 1000.0
            f_order = 3
            g_order = 3
            f, g = f_g_series(r0, dt, f_order, g_order)
            # f and g should be numpy arrays or floats (if summed)
            assert isinstance(f, (np.ndarray, float))
            assert isinstance(g, (np.ndarray, float))
            # For f_order < 5, should be sum (float)
            assert isinstance(f, float)
            assert isinstance(g, float)

        def test_f_g_series_full_order():
            # Test with full order (5), should return arrays
            r0 = np.array([1.0e8, 0.0, 0.0])
            dt = 1000.0
            f_order = 5
            g_order = 5
            f, g = f_g_series(r0, dt, f_order, g_order)
            assert isinstance(f, np.ndarray)
            assert isinstance(g, np.ndarray)
            assert f.shape == (5,)
            assert g.shape == (5,)
            # 0th order f should be 1.0, 1st order 0.0, 2nd order positive
            np.testing.assert_allclose(f[0], 1.0)
            np.testing.assert_allclose(f[1], 0.0)
            assert f[2] > 0

        def test_f_g_series_with_v0():
            # Provide a nonzero v0
            r0 = np.array([1.0e8, 0.0, 0.0])
            v0 = np.array([0.0, 1.0e3, 0.0])
            dt = 1000.0
            f_order = 5
            g_order = 5
            f, g = f_g_series(r0, dt, f_order, g_order, v0=v0)
            assert isinstance(f, np.ndarray)
            assert isinstance(g, np.ndarray)
            # 3rd and 4th order terms should be nonzero due to v0
            assert not np.allclose(f[3], 0)
            assert not np.allclose(g[4], 0)

        def test_f_g_series_invalid_r0_shape():
            # r0 wrong shape
            r0 = np.array([[1.0, 0.0, 0.0]])
            dt = 1000.0
            with pytest.raises(AssertionError):
                f_g_series(r0, dt, 3, 3)

        def test_f_g_series_zero_r0():
            # r0 is zero vector
            r0 = np.zeros(3)
            dt = 1000.0
            with pytest.raises(AssertionError):
                f_g_series(r0, dt, 3, 3)

        def test_f_g_series_invalid_orders():
            r0 = np.array([1.0e8, 0.0, 0.0])
            dt = 1000.0
            with pytest.raises(AssertionError):
                f_g_series(r0, dt, -1, 3)
            with pytest.raises(AssertionError):
                f_g_series(r0, dt, 3, -1)

        def test_f_g_series_invalid_dt():
            r0 = np.array([1.0e8, 0.0, 0.0])
            dt = "not_a_number"
            with pytest.raises(AssertionError):
                f_g_series(r0, dt, 3, 3)


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
    # Textbook example with typical values - Orbital mechancis for Engineering Students Ex:
    # IOD as no v0 is provided
    r0 = np.array([1.0e8, 0.0, 0.0])
    dt = 1000.0
    f_order = 3
    g_order = 3
    f, g = f_g_series(r0, dt, f_order, g_order)
    # Should return valid f and g values
    # Replace 1.234 with the textbook value you want to check against
    f_textbook_value = 1.234
    g_textbook_value = 1.234  
    assert np.isclose(f, f_textbook_value, rtol=1e-6), f"Expected {f_textbook_value}, got {f}"
    assert np.isclose(g, g_textbook_value, rtol=1e-6), f"Expected {g_textbook_value}, got {g}"

def test_f_g_series_textbook_example():
    # Textbook example with typical values - Orbital mechanics for Engineering Students Ex:
    # v0 is provided
    r0 = np.array([1.0e8, 0.0, 0.0])
    v0 = np.array([0.0, 1.0e3, 0.0])  # Nonzero velocity vector
    dt = 1000.0
    f_order = 3
    g_order = 3
    f, g = f_g_series(r0, dt, f_order, g_order, v0=v0)
    # Should return valid f and g values
    # Replace 1.234 with the textbook value you want to check against
    f_textbook_value = 1.234
    g_textbook_value = 1.234  
    assert np.isclose(f, f_textbook_value, rtol=1e-6), f"Expected {f_textbook_value}, got {f}"
    assert np.isclose(g, g_textbook_value, rtol=1e-6), f"Expected {g_textbook_value}, got {g}"