import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit, njit
import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission


seed = 36874
system = SolarSystem(seed)
mission = SpaceMission(seed)
sigma = constants.sigma

def find_star_luminosity(star_temp, star_radius):
    """
    Finding total luminosity of the star
    :param star_temp: Star temperature in Kelvin
    :param star_radius: Star radius in kilometers
    :return: Luminosity of the star in Watts
    """
    return 4*np.pi*sigma*(star_temp**4)*((star_radius*1000)**2)

def find_dist_flux(star_lum, radius):
    """
    Calculates the flux based on the distance from the star
    :param star_lum: Star luminosity
    :param radius: Distance to star
    :return: Flux in Watts per square meter
    """
    return star_lum/(4*np.pi*radius**2)

def find_tot_planet_energy(star_lum, orbit_radius, planet_radius):
    """
    Finds total energy received by planet per second
    :param star_lum: Star Luminosity
    :param orbit_radius: Distance from star
    :param planet_radius: Radius of planet
    :return: Total Energy received per second in Watts
    """
    return star_lum*(2*np.pi*planet_radius**2)/(4*np.pi*orbit_radius**2)

def find_dist_temp(star_lum, rad):
    """
    Finds the temperature of an object based on its distance from the star
    :param star_lum: Star luminosity
    :param rad: Distance from star
    :return: Temperature in Kelvin
    """
    flux = star_lum/(4*np.pi*rad**2)
    return (flux/sigma)**(1/4)

def find_panel_area(req_power, efficiency, planet_flux):
    """
    Finds required solar panel area
    :param req_power: Required power in Watts
    :param efficiency: Efficiency of the solar panels
    :param planet_flux: Electromagnetic flux on the planet
    :return: Required solar panel size in square meters
    """
    total_dE = req_power/efficiency
    return total_dE/planet_flux


# Calculating temperatures and required solar panel sizes for all planets
flux_star = find_star_luminosity(system.star_temperature, system.star_radius)
print(f"Star Luminosity: {flux_star}")
temps_areas = np.zeros([8, 2])
for i in range(8):
    r_orbit = utils.AU_to_m(np.linalg.norm(system.initial_positions[:, i]))
    temps_areas[i][0] = find_dist_temp(flux_star, r_orbit)  # Finding temperature of each planet
    temps_areas[i][1] = find_panel_area(40, 0.12, find_dist_flux(flux_star, r_orbit))  # Finding required solar panel size for each planet
    print(i, f"{find_dist_flux(flux_star, r_orbit):.3f}")  # Printing Flux at each planet's distance from the star


# Calculating and printing temperatures of all planets and whether they are inside the habitable zone
for i in range(len(temps_areas)):
    if 260 < temps_areas[i][0] < 390:  # Checking whether planet is inside the habitable zone
        print(f"Planet {i} has a temperature of {temps_areas[i][0]-273.15:.2f} deg C ({temps_areas[i][0]:.2f} deg Kelvin) and is in the habitable zone")
    else:
        print(f"Planet {i} has a temperature of {temps_areas[i][0]-273.15:.1f} deg C ({temps_areas[i][0]:.1f} deg Kelvin)")
    print(f"We need a solar panel which is {temps_areas[i][1]:.3f} m^2 large\n")

for i in np.linspace(12.532, 12.5320125, 5):  # Looking for inner border of the habitable zone
    temp = find_dist_temp(flux_star, utils.AU_to_m(i))
    print(i, temp, (260 < temp < 390))

for i in np.linspace(28.197, 28.197025, 5):  # Looking for outer border of the habitable zone
    temp = find_dist_temp(flux_star, utils.AU_to_m(i))
    print(i, temp, (260 < temp < 390))


# Plotting Flux as function of distance from star
r = np.linspace(1, 66, 100000)
flux_r = find_dist_flux(flux_star, r)
temp_r = find_dist_temp(flux_star, r)
area_r = find_panel_area(40, 0.12, find_dist_flux(flux_star, utils.AU_to_m(r)))
plt.plot(r, flux_r)
plt.xlabel("Distance from Star [AU]")
plt.ylabel("Solar Energy Flux [W/$m^2]$")
plt.grid()
plt.savefig("../Figures/Radius_flux.png")
plt.clf()

# Plotting Temperature as function of distance from star
plt.plot(r, temp_r)
plt.xlabel("Distance from Star [AU]")
plt.ylabel("Temperature [K]$")
plt.grid()
plt.savefig("../Figures/Radius_temp.png")
plt.clf()

# Plotting Solar panel area as function of distance from star
plt.plot(r, area_r)
plt.xlabel("Distance from Star [AU]")
plt.ylabel("Size of Solar Panels [$m^2]$")
plt.grid()
plt.savefig("../Figures/Radius_area.png")
plt.clf()

# Plotting k-ratio of gravitational forces as function of distance from planet
orbit5_r = utils.AU_to_m(system.semi_major_axes[5])
planet5_m = system.masses[5]
star_m = system.star_mass
g_r = np.linspace(0.18e9, 0.5e9, 100000)
k = planet5_m*orbit5_r**2/(star_m*g_r**2)
for i in range(len(k)):
    if 9.999 < k[i] < 10.001:
        print(g_r[i])
plt.plot(g_r, k)
plt.xlabel("Distance from Planet [m]")
plt.ylabel("Ratio k")
plt.grid()
plt.savefig("../Figures/Grav_ratio_k.png")
plt.clf()

# Plotting distance required to resolve planet on photo
planet5_rad = system.radii[5]*1000
fov = utils.deg_to_rad(2.6)  # Field of view of 800 mm lens
pixels = np.linspace(1024, 12096, 100000)
dist = planet5_rad*pixels/fov
plt.plot(dist, pixels)
plt.xlabel("Distance from Planet [m]")
plt.ylabel("Required Pixels")
plt.grid()
plt.savefig("../Figures/Res_pixels.png")
plt.show()


