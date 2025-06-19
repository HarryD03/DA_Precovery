from pyaudi import gdual_double as gdual, sin

x = gdual(0., 'x', 5)
print("sin(x) =", sin(x))