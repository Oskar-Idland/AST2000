import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
a = np.array([1, 2, 3, 4])
b = np.array([1, 1, 2, 2])
print(a*b)

print(utils.AU_to_m(0.0009600889548249979))
print(12_248_227 - 8_653_612)
print(np.linalg.norm([-1.10860419, -6.6127185]))
print(np.arctan2(-6.18378838, 2.59187393)/np.pi*180 + 360)

t = 2*np.pi*0.0005265799873456048/4.038591791057868
print(t)


