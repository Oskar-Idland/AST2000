import numpy as np
import matplotlib.pyplot as plt
from f_P import *
from f_Maxwell_Boltzmann import *
from molmass import Formula
from scipy.constants import Avogadro as A

# Challenge A_3_1 --------------------
"""
Since the formula uses the absolute velocity of the particle, we are going to use the "max_boltz_dist3D"-function
"""
m = Formula('H2').mass/(A*1000)  # Mass of one H2 molecule In Kg
a = 0
b = np.inf
T = 3000 
def mean_M_B(T, m, a, b):
    def f(v, T, m):
        return v*max_boltz_dist3D(v, T, m)
    return quad(f, a, b, args=(T, m))[0]

v_avg = mean_M_B(T,m,a,b)
print(v_avg)