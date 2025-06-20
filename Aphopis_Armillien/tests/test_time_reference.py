import pytest
import numpy as np
from Aphopis_Armillien.utils.time_reference import J0, zeroTo360, LST, equatorial_to_eclipitcJ2000, create_da_los_vectors

def test_J0():
    # Typical date
    assert J0(2000, 1, 1) == pytest.approx(2451544.5)
    # Edge cases
    with pytest.raises(AssertionError):
        J0(1800, 1, 1)  # Year out of range
    with pytest.raises(AssertionError):
        J0(2000, 13, 1)  # Month out of range
    with pytest.raises(AssertionError):
        J0(2000, 1, 32)  # Day out of range

def test_zeroTo360():
    assert zeroTo360(0) == 0
    assert zeroTo360(360) == 360
    assert zeroTo360(720) == 360
    assert zeroTo360(-360) == 0
    assert zeroTo360(-720) == 0
    assert zeroTo360(-90) == 270
    assert zeroTo360(450) == 90

def test_LST():
    # LST should return a float in [0, 360] and a float epoch
    # Test with textbook example
    result = LST(2000, 1, 1, 0, 0)
    if isinstance(result, tuple):
        lst, epoch = result
        assert 0 <= lst <= 360
        assert isinstance(epoch, float)
    else:
        assert isinstance(result, float)
        assert 0 <= result <= 360

def test_equatorial_to_eclipitcJ2000():
    # Test with typical values / Textbook example

    dr = 1.0
    lst = 0.0
    h_e = 0.0
    result = equatorial_to_eclipitcJ2000(dr, lst, h_e)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # Check if the result is a valid rotation (not NaN or Inf)
    assert np.all(np.isfinite(result))

def test_create_da_los_vectors():
    # Test with typical values / Textbook example
    # Right ascension and declination in Radians

    # Example values for RA and Dec
    ra = np.array([[0, 6, 12, 18]])  # Right ascension in radians
    dec = np.array([[0, np.pi/6, np.pi/3, np.pi/2]])  # Declination in radians

    result = create_da_los_vectors(ra, dec)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)  # 2D vectors for 4 observations
    # Check if the result is a valid direction (not NaN or Inf)
    assert np.all(np.isfinite(result))