import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

username = "janniesc"
seed = utils.get_seed(username)

# Solar System
system = SolarSystem(seed)
print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'
      .format(system.star_mass, system.star_radius))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))

# Spacecraft
mission = SpaceMission(seed)
home_planet_idx = 0  # The home planet always has index 0
print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))
    print(f"Planet mass is {system.masses[planet_idx]*1.989e30} kg")
    print(f"Planets rotational period is {system.rotational_periods[planet_idx]}")
    print(f"Planets radius is {system.radii[planet_idx]}\n")

system.print_info()
print(system.initial_positions)
