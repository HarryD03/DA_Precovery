import pytest
import numpy as np
from Aphopis_Armillien.utils.time_reference import J0, zeroTo360, LST, equatorial_to_eclipitcJ2000

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
    assert zeroTo360(720) == 0
    assert zeroTo360(-90) == 270
    assert zeroTo360(450) == 90

def test_LST():
    # LST should return a float in [0, 360] and a float epoch
    result = LST(2000, 1, 1, 0, 0)
    if isinstance(result, tuple):
        lst, epoch = result
        assert 0 <= lst <= 360
        assert isinstance(epoch, float)
    else:
        assert isinstance(result, float)
        assert 0 <= result <= 360

def test_equatorial_to_eclipitcJ2000():
    dr = 1.0
    lst = 0.0
    h_e = 0.0
    result = equatorial_to_eclipitcJ2000(dr, lst, h_e)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # Check if the result is a valid rotation (not NaN or Inf)
    assert np.all(np.isfinite(result))