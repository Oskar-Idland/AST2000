import numpy as np
from scipy.stats import maxwell
from f_P import P3
from f_Maxwell_Boltzmann import max_boltz_dist1D, max_boltz_dist3D
import matplotlib.pyplot as plt
from scipy.constants import Avogadro as A
from molmass import Formula


# Challenge A_2_1 --------------------
N = 10**5  # Number of H2 molecules
T = 3000  # Temperature
m = Formula('H2').mass/(A*1000)  # Mass of one H2 molecule In Kg

a = -2.5*(10**4)
b = 2.5*(10**4)
n = 20000
vx = np.linspace(a, b, n)
Px = max_boltz_dist1D(vx, T, m)
plt.plot(vx, Px)
plt.xlabel("1D-Velocity of particle [m/s]")
plt.ylabel("Probability")
plt.title("Maxwell-Boltzmann distribution of H2 molecule")
plt.grid()
plt.show()

# Challenge A_2_3
a = 0
b = 3*1E4
x = np.linspace(a,b,n)
plt.plot(x,abs(P))
plt.show()




# Challenge A_2_3 --------------------
a3 = 0
b3 = 3*10**4
v = np.linspace(a3, b3, n)
P_abs = max_boltz_dist3D(v, T, m)
plt.plot(v, P_abs)
plt.xlabel("Absolute Velocity of particle [m/s]")
plt.ylabel("Probability")
plt.title("Maxwell-Boltzmann distribution of H2 molecule")
plt.grid()
plt.show()

"""
There is no conflict between these two plots since these are different plots. One is for a one-dimensional velocity
only, whereas the other is for the absolute velocity in all dimensions. 
"""
