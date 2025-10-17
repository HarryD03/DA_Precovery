
## Import Packages

from pyaudi import gdual_double as gdual
from pyaudi import sin, cos, tan

import numpy as np
import matplotlib as plt

m = 3                            # Order of the polynomial

x = gdual(0, "x", m)       #gduals so they behave like a taylor polynomials around the value 0.
y = gdual(0, "y", m)
z = gdual(0, "z", m)

# As defined in Bernz paper, addition, multiplication and division can be conducted on gdual numbers.

f = x*x + 2*tan(x/(y+1)) - sin(z) # This is a function, described as a taylor expansion of the other functions.
                                  # The sin, tan functions are the 'elementary' taylor polynomial functions described within the bernz text

print(f)                          # print taylor polynomial in console