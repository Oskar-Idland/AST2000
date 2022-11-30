import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
username = "janniesc"
seed = 36874
code_launch_results = 83949
code_escape_trajectory = 74482
stable_orbit = 88515
system = SolarSystem(seed)
mission = SpaceMission(seed)
planet_idx = 1
# a = np.array([1, 2, 3, 4])
# b = np.array([1, 1, 2, 2])
# print(a*b)
#
# print(utils.AU_to_m(0.0009600889548249979))
# print(12_248_227 - 8_653_612)
# print(np.linalg.norm([-1.10860419, -6.6127185]))
# print(np.arctan2(-6.18378838, 2.59187393)/np.pi*180 + 360)
#
# t = 2*np.pi*0.0005265799873456048/4.038591791057868
# print(t)
planet_rad = system.radii[planet_idx]*1000
r0 = planet_rad + 75_000_000
r1 = planet_rad + 1_000_000
m_planet = system.masses[planet_idx] * 1.98847e30  # Planet mass
GM = (6.6743015 * 10 ** (-11))*m_planet  # Standard Gravitational parameter
v0_abs = np.sqrt(GM * m_planet / r0)  # Absolute velocity in the initial circular orbit
T_orb0 = (2 * np.pi * r0) / v0_abs  # Calculating orbital period
delta_v0 = -np.sqrt(GM/r1)*(np.sqrt((2*r0)/(r0+r1))-1)  # Calculating required boost to enter Hohmann transfer orbit
delta_v1 = -np.sqrt(GM / r0) * (np.sqrt((2 * r1) / (r0 + r1)) - 1)

print(r0)
print(r1)
print(v0_abs)
print(delta_v0)
print(delta_v1)
print(1*10**6)

print(100*64.25/4775244.860135423)