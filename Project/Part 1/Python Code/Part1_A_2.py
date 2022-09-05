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
x = np.linspace(a, b, n)
P = max_boltz_dist1D(x, T, m)
plt.plot(x, P)
plt.xlabel("1D-Velocity of particle [m/s]")
plt.ylabel("Probability [%]")
plt.title("Maxwell-Boltzmann distribution of H2 molecule")
plt.grid()
plt.show()

# Challenge A_2_3
a = 0
b = 3*1E4
x = np.linspace(a,b,n)
plt.plot(x,abs(P))
plt.show()





