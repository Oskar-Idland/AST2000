import numpy as np
from scipy import interpolate
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

# Initializing system
seed = 36874
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

def calc_vel_dop(dopplershift, lambda0):
    c = constants.c
    return dopplershift*c/lambda0

def find_velocity(lambda0, phi1, phi2, sc_ds1, sc_ds2, star_ds1, star_ds2):
    v_sc1 = calc_vel_dop(sc_ds1, lambda0)  # Velocities of the spacecraft relative to reference stars
    v_sc2 = calc_vel_dop(sc_ds2, lambda0)
    v_star1 = calc_vel_dop(star_ds1, lambda0)  # Velocities of our star relative to reference stars
    v_star2 = calc_vel_dop(star_ds2, lambda0)
    v_rel = [v_sc1-v_star1, v_sc2-v_star2]  # Relative velocity of spacecraft to star in u-coordinate system
    coeff = 1/np.sin(phi1-phi2)
    u_to_cart = np.array([[np.sin(phi2), -np.sin(phi1)], [-np.cos(phi2), np.cos(phi1)]])  # Matrix to convert from u-coordinate system to cartesian system
    vel_cart = coeff*np.matmul(u_to_cart, v_rel)  # Converting to cartesian coordinates
    return utils.m_pr_s_to_AU_pr_yr(vel_cart)  # Changing units


vf_launch_res()

lambda0 = mission.reference_wavelength
phi1, phi2 = utils.deg_to_rad(mission.star_direction_angles)  # Angles to reference stars
sc_dop_shift1, sc_dop_shift2 = mission.measure_star_doppler_shifts()  # Doppler shifts at spacecraft
star_dop_shift1, star_dop_shift2 = mission.star_doppler_shifts_at_sun  # Doppler shift at star in solar system

print(find_velocity(lambda0, phi1, phi2, sc_dop_shift1, sc_dop_shift2, star_dop_shift1, star_dop_shift2))

