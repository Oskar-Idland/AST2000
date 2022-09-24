import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)
star_mass = system.star_mass
for i in range(len(system.initial_positions[0])):
    mass = system.masses[i]
    r = np.array([system.initial_positions[0][i],system.initial_positions[1][i]])
    print(abs(mass/np.linalg.norm(r)**2))
print(r)