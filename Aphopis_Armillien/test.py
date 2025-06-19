from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Union, overload, Tuple

import numpy as np
from Tools.demo.sortvisu import steps
from daceypy import DA, array
from daceypy.ifunction import implicit_function
import daceypy.op as op
from numpy.typing import NDArray
from matplotlib import pyplot as plt


help(implicit_function)