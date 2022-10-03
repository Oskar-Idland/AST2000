import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit, njit
import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission


username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)
sigma = constants.sigma

def find_star_luminosity(star_temp, star_radius):
    return 4*np.pi*sigma*(star_temp**4)*((star_radius*1000)**2)

def find_dist_flux(star_lum, radius):
    return star_lum/(4*np.pi*radius**2)

def find_tot_planet_energy(star_lum, orbit_radius, planet_radius):
    return star_lum*(2*np.pi*planet_radius**2)/(4*np.pi*orbit_radius**2)

def find_planet_temp(star_lum, rad_planet_orbit):
    flux = star_lum/(4*np.pi*rad_planet_orbit**2)
    return (flux/sigma)**(1/4)

def find_panel_area(req_power, efficiency, planet_flux):
    total_dE = req_power/efficiency
    return total_dE/planet_flux


flux_star = find_star_luminosity(system.star_temperature, system.star_radius)
temps_areas = np.zeros([8, 2])
for i in range(8):
    r_orbit = utils.AU_to_m(np.linalg.norm(system.initial_positions[:, i]))
    temps_areas[i][0] = find_planet_temp(flux_star, r_orbit)
    temps_areas[i][1] = find_panel_area(40, 0.12, find_dist_flux(flux_star, r_orbit))

print(temps_areas)
for i in range(len(temps_areas)):
    if 260 < temps_areas[i][0] < 390:
        print(f"Planet {i} has a temperature of {temps_areas[i][0]-273.15:.2f} deg C ({temps_areas[i][0]:.2f} deg Kelvin) and is in the habitable zone")
    else:
        print(f"Planet {i} has a temperature of {temps_areas[i][0]-273.15:.2f} deg C")
    print(f"We need a solar panel which is {temps_areas[i][1]:.2f} m^2 large\n")

for i in np.linspace(12.532, 12.5320125, 5):
    temp = find_planet_temp(flux_star, utils.AU_to_m(i))
    print(i, temp, (260 < temp < 390))

for i in np.linspace(28.197, 28.197025, 5):
    temp = find_planet_temp(flux_star, utils.AU_to_m(i))
    print(i, temp, (260 < temp < 390))

