import numpy as np
from scipy import interpolate
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)
star_mass = system.star_mass
planet_mass = system.masses[0]

lambda0 = 656.3
phi1, phi2 = utils.deg_to_rad(mission.star_direction_angles)
sc_dop_shift0, sc_dop_shift1 = mission.measure_star_doppler_shifts()
star_dop_shift0, star_dop_shift1 = mission.star_doppler_shifts_at_sun
