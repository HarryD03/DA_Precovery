"""
This module contains utilities for the propagation of DAIOD (Differential Algebra Initial Orbit Determination) algorithms via Automatic Domain Splitting (ADS).
It includes the implementation of the DAIOD algorithm without automatic domain splitting.
"""

from daceypy import DA, array, ADS
import numpy as np
from astropy import units as u


