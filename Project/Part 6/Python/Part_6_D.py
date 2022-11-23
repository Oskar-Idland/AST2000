import numpy as np
import sys
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
# sys.path.append("../../Part 1/Python Code")
# from Rocket_launch import launch_rocket  # It works when running, even if it shows an error in the editor!
from numba import njit

# Initializing system
username = "janniesc"
seed = 36874
code_launch_results = 83949
code_escape_trajectory = 74482
stable_orbit = 88515
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [stable_orbit])
G = 4 * np.pi ** 2
planet_idx = 1

# FOR SIMPLICITY WE WILL ONLY MOVE ON THE XY-PLANE

dt = 8.666669555556518e-05
time0 = 2762487 * dt
orbital_height0 = 75_000_000
orbital_angle0 = 0

# Putting Spacecraft into high stable orbit
# shortcut.place_spacecraft_in_stable_orbit(time0, orbital_height0, orbital_angle0, planet_idx)


def decrease_orbit(r0, r1, land_seq, planet_idx):
    m_planet = system.masses(planet_idx) * 1.98847e30
    G_SI = 6.6743015 * 10 ** (-11)
    v0_abs = np.sqrt(G_SI * m_planet / r0)
    T_orb0 = (2 * np.pi * r0) / v0_abs


def find_actual_coordinates(curr_coords, time_elapsed, planet_idx=1, cartesian=False):
    """
    Calculates at which coordinates the input position was at time 0.
    :param curr_coords: Current coordinates in spherical form in radians (or cartesian if cartesian is set to True) (Phi is 0 at z-axis)
    :param time_elapsed: Point of time from start in years
    :param planet_idx: Planet index
    :return: Spherical coordinates of the position we are over now at time 0.
    """
    # If current coordinates are cartesian
    if cartesian:
        x = curr_coords[0]
        y = curr_coords[1]
        z = curr_coords[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))
        curr_coords = np.array([r, theta, phi])

    r_planet = system.radii[planet_idx] * 1000
    omega = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)
    time_elapsed_sec = utils.yr_to_s(time_elapsed)
    new_theta = (curr_coords[1] - (omega * time_elapsed_sec)) % (2 * np.pi)
    new_coords = np.array([r_planet, new_theta, curr_coords[2]])
    return new_coords


def find_new_coordinates(curr_coords, time_ahead, planet_idx=1, cartesian=False):
    """
    Calculates at which coordinates the input position will be at a future point in time.
    :param curr_coords: Current coordinates in spherical form in radians (or cartesian if cartesian is set to True) (Phi is 0 at z-axis)
    :param time_ahead: Time interval in seconds (from now) how long in the future the coordinate prediction will be
    :param planet_idx: Planet index
    :return: Spherical coordinates (in radians) of the position we are over now at a time [time_interval] from now.
    """
    # If current coordinates are cartesian
    if cartesian:
        x = curr_coords[0]
        y = curr_coords[1]
        z = curr_coords[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))
        curr_coords = np.array([r, theta, phi])

    r_planet = system.radii[planet_idx] * 1000
    omega = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)
    time_elapsed_sec = utils.yr_to_s(time_ahead)
    new_theta = (curr_coords[1] + (omega * time_elapsed_sec)) % (2 * np.pi)
    new_coords = np.array([r_planet, new_theta, curr_coords[2]])
    return new_coords



if __name__ == "__main__":
    for i in range(0, 2*314):
        ri = i/100
        r, theta, phi = find_new_coordinates([1e8, ri, np.pi/4], 1)
        print(f"r: {r:.2f}, Theta: {theta/np.pi:.2f} pi, Phi: {phi/np.pi:.2f} pi")
