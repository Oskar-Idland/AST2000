import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit
import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)

N = 100000
T = 0.001
d_t = T/N
a = system.semi_major_axes[0]
e = system.eccentricities[0]
init_angle = system.aphelion_angles[0]
angles1 = np.linspace(-np.pi/16, np.pi/16, N)
d_theta1 = (angles1[-1]-angles1[0])/N
angles2 = np.linspace(7*np.pi/16, 9*np.pi/16, N)
d_theta2 = (angles2[-1]-angles2[0])/N


# Area Perihelion
A1 = 0
A2 = 0
for i in range(len(angles1)):
    r1 = (a * (1 - e ** 2)) / (1 + (e * np.cos(angles1[i] - init_angle)))
    A1 += 0.5 * r1 ** 2 * d_theta1
    r2 = (a * (1 - e ** 2)) / (1 + (e * np.cos(angles2[i] - init_angle)))
    A2 += 0.5 * r2 ** 2 * d_theta2

print(f"Aphelion Area: {A1:.4f} AU^2")
print(f"Perihelion Area: {A2:.4f} AU^2")

print(system.aphelion_angles[0])
