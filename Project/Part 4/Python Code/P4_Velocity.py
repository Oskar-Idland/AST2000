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


def vf_launch_res():
    l_par = np.load("../../Part 1/Python Code/Launch_Parameters.npy")
    mission.set_launch_parameters(l_par[2], l_par[3], l_par[4], l_par[5], [l_par[6], l_par[7]], l_par[8])
    mission.launch_rocket(l_par[9])
    mission.verify_launch_result([l_par[0], l_par[1]])
    print(f"Launch Results Verified: {mission.launch_result_verified}")


vf_launch_res()
lambda0 = 656.3
phi1, phi2 = utils.deg_to_rad(mission.star_direction_angles)
sc_dop_shift0, sc_dop_shift1 = mission.measure_star_doppler_shifts()
star_dop_shift0, star_dop_shift1 = mission.star_doppler_shifts_at_sun
