import numpy as np
import matplotlib.pyplot as plt
from f_P import *
from f_Maxwell_Boltzmann import *


# Challenge A_3_1 --------------------
"""
Since the formula uses the absolute velocity of the particle, we are going to use the "max_boltz_dist3D"-function
"""
a = 0
b = np.inf
def mean_M_B(v, T, m, a, b):
    def f(vel, T, m):
        return v*max_boltz_dist3D(vel, T, m)
    return quad(f, a, b, args=(T, m))[0]

