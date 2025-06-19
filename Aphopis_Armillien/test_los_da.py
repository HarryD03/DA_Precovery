import numpy as np
import pytest
from daceypy import array
from main import create_da_los_vectors

import daceypy.op as op


def test_create_da_los_vectors_basic():
    # Test with simple RA/DEC values (in radians)
    ra = np.array([[0.0, np.pi/2, np.pi]])  # shape (1, 3)
    dec = np.array([[0.0, 0.0, 0.0]])       # shape (1, 3)

    los = create_da_los_vectors(ra, dec)
    assert los.shape == (3, 3)
    # Check first vector: should be [1, 0, 0]
    np.testing.assert_allclose([float(los[0,0][0]), float(los[1,0][0]), float(los[2,0][0])], [1, 0, 0], atol=1e-8)
    # Second: [0, 1, 0]
    np.testing.assert_allclose([float(los[0,1][0]), float(los[1,1][0]), float(los[2,1][0])], [0, 1, 0], atol=1e-8)
    # Third: [-1, 0, 0]
    np.testing.assert_allclose([float(los[0,2][0]), float(los[1,2][0]), float(los[2,2][0])], [-1, 0, 0], atol=1e-8)

def test_create_da_los_vectors_dec_variation():
    # Test with nonzero declination (45 degrees)
    ra = np.array([[0.0]])
    dec = np.array([[np.pi/4]])
    los = create_da_los_vectors(ra, dec)
    # Should be [cos(45), 0, sin(45)]
    expected = [np.cos(np.pi/4), 0, np.sin(np.pi/4)]
    np.testing.assert_allclose([float(los[0,0][0]), float(los[1,0][0]), float(los[2,0][0])], expected, atol=1e-8)

def test_create_da_los_vectors_shape_asserts():
    # RA not a row vector
    ra = np.array([[0.0], [1.0]])
    dec = np.array([[0.0], [1.0]])
    with pytest.raises(AssertionError):
        create_da_los_vectors(ra, dec)
    # DEC not a row vector
    ra = np.array([[0.0, 1.0]])
    dec = np.array([[0.0], [1.0]])
    with pytest.raises(AssertionError):
        create_da_los_vectors(ra, dec)
    # Mismatched number of observations
    ra = np.array([[0.0, 1.0]])
    dec = np.array([[0.0]])
    with pytest.raises(AssertionError):
        create_da_los_vectors(ra, dec)

def test_create_da_los_vectors_with_daceypy_array():
    # Test with daceypy.array input
    ra = array([[0.0, np.pi/2]])
    dec = array([[0.0, 0.0]])
    los = create_da_los_vectors(ra, dec)
    assert los.shape == (3, 2)
    # Should behave as with numpy arrays for nominal values
    np.testing.assert_allclose([float(los[0,0][0]), float(los[1,0][0]), float(los[2,0][0])], [1, 0, 0], atol=1e-8)
    np.testing.assert_allclose([float(los[0,1][0]), float(los[1,1][0]), float(los[2,1][0])], [0, 1, 0], atol=1e-8)