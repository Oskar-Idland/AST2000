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

d_t = 1
r = 1
d_theta = 1
d_A = 0.5*r**2*d_theta

# Area Perihelion


# Area Aphelion
