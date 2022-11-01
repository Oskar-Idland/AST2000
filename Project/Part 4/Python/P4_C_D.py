import numpy as np
from scipy import interpolate
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from Part_A import find_angle

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
    print(f"Dist Sun: {dist_sun}")
    print(f"Dist Planet 1: {dist_p1}")
    print(f"Dist Planet 2: {dist_p2}")
    A = (-2*pos_sun[0] + 2*pos_p1[0])
    B = (-2*pos_sun[1] + 2*pos_p1[1])
    C = (dist_sun**2) - (dist_p1**2) - (pos_sun[0]**2) + (pos_p1[0]**2) - (pos_sun[1]**2) + (pos_p1[1]**2)
    D = (-2*pos_p1[0] + 2*pos_p2[0])
    E = (-2*pos_p1[1] + 2*pos_p2[1])
    F = (dist_p1**2) - (dist_p2**2) - (pos_p1[0]**2) + (pos_p2[0]**2) - (pos_p1[1]**2) + (pos_p2[1]**2)
    x_sc = (C*E - F*B)/(E*A - B*D)
    y_sc = (C*D - A*F)/(B*D - A*E)
    return np.array([x_sc, y_sc])


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
mission.take_picture()

lambda0 = mission.reference_wavelength
phi1, phi2 = utils.deg_to_rad(mission.star_direction_angles)  # Angles to reference stars
sc_dop_shift1, sc_dop_shift2 = mission.measure_star_doppler_shifts()  # Doppler shifts at spacecraft
star_dop_shift1, star_dop_shift2 = mission.star_doppler_shifts_at_sun  # Doppler shift at star in solar system
print(sc_dop_shift1, sc_dop_shift2, star_dop_shift1, star_dop_shift2)



position_SC = locate_spacecraft(3, 6, utils.s_to_yr(885.21))
velocity_SC = find_velocity(lambda0, phi1, phi2, sc_dop_shift1, sc_dop_shift2, star_dop_shift1, star_dop_shift2)
angle_SC = find_angle("sky_picture.png")
print(f"Position: {position_SC}")
print(f"Velocity: {velocity_SC}")
print(f"Angle: {angle_SC}")

mission.verify_manual_orientation(position_SC, velocity_SC, angle_SC)
