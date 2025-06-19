from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Union, overload

import numpy as np
from Tools.demo.sortvisu import steps
from daceypy import DA, array
from daceypy.op import cos, sin, sqr, sqrt, vnorm
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from time import perf_counter


def main():

    # initialize DACE for 20th-order computations in 1 variable
    DA.init(20, 1)

    # initialize x as DA
    x = DA(1)

    # compute y = sin(x)
    y = x.sin()

    # print x and y to screen
    print("x\n" + str(x) + "\n")
    print("y = sin(x)\n" + str(y) + "\n")


if __name__ == "__main__":
    main()