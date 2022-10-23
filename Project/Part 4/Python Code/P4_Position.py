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


def create_orbit_func(planet_idx):
    # Loading numeric orbits from file
    data = np.load(f"../../Part 2/Python Code/Orbits/Planet_{planet_idx}.npz")
    t = data["time"]
    x = data["position"]
    v = data["velocity"]
    # INTERPOLATION
    position_function = interpolate.interp1d(t, x, axis=0, bounds_error=False, fill_value='extrapolate')
    velocity_function = interpolate.interp1d(t, v, axis=0, bounds_error=False, fill_value='extrapolate')
    return position_function, velocity_function


def locate_spacecraft(ref_p1_idx, ref_p2_idx, time):
    pos_sun = np.array([0, 0])
    pos_p1 = create_orbit_func(ref_p1_idx)[0](time)
    pos_p2 = create_orbit_func(ref_p2_idx)[0](time)
    dist_sun = mission.measure_distances()[-1]
    dist_p1 = mission.measure_distances()[ref_p1_idx]
    dist_p2 = mission.measure_distances()[ref_p2_idx]
    A = (-2*pos_sun[0] + 2*pos_p1[0])
    B = (-2*pos_sun[1] + 2*pos_p1[1])
    C = (dist_sun**2) - (dist_p1**2) - (pos_sun[0]**2) + (pos_p1[0]**2) - (pos_sun[1]**2) + (pos_p1[1]**2)
    D = (-2*pos_p1[0] + 2*pos_p2[0])
    E = (-2*pos_p1[1] + 2*pos_p2[1])
    F = (dist_p1**2) - (dist_p2**2) - (pos_p1[0]**2) + (pos_p2[0]**2) - (pos_p1[1]**2) + (pos_p2[1]**2)
    x_sc = (C*E - F*B)/(E*A - B*D)
    y_sc = (C*D - A*F)/(B*D - A*E)
    return np.array([x_sc, y_sc])


vf_launch_res()
position_SC = locate_spacecraft(3, 6, 1)
print(position_SC)
