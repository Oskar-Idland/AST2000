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
G = 6.6743015e-11
planet_idx = 1

m_planet = system.masses(planet_idx) * 1.98847e30
g = G*m_planet/(system.radii[1]*1000)
rho_atm = system.atmospheric_densities[planet_idx]
C_D = 1
omega_atm = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)


def cart_to_spherical(x, y, z):
    """
    Conversion from cartesian coordinates to spherical coordinates
    :param x: x-coordinate
    :param y: y-coordinate
    :param z: z-coordinate
    :return: Spherical coordinates r, theta, phi (phi measured from z-axis)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos2(z, r)
    phi = np.arccos2(x, np.sqrt(x ** 2 + y ** 2))
    return r, theta, phi


def spherical_to_cart(r, theta, phi):
    """
    Conversion from spherical coordinates to cartesian coordinates
    :param r: radius from center
    :param theta: theta-angle
    :param phi: phi-angle
    :return: Cartesian coordinates x, y, z
    """
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def find_w(current_pos, planet_idx=1):
    """
    Finds velocity of atmosphere based on the position relative to the planet
    :param current_pos: Current position relative to the planet in cartesian coordinates
    :return: velocity of the atmosphere at the given position in cartesian coordinates
    """
    r, theta, phi = cart_to_spherical(current_pos[0], current_pos[1], current_pos[2])  # Converting to spherical coordinates
    omega = (2*np.pi)/(system.rotational_periods[planet_idx]*24*3600)  # Angular velocity of planet
    abs_w = omega*r  # Calculating absolute velocity of atmosphere
    dir_w = np.array([-np.sin(theta), np.cos(theta), 0])  # Calculating direction of velocity of atmosphere
    return abs_w*dir_w


def drag(pos, vel, area_parachute=0):
    """
    Calculates drag force based on current position, velocity and parachute area
    :param pos: Current position in cartesian coordinates
    :param vel: Current velocity in cartesian coordinates
    :param area_parachute: Area of parachutes in m^2 (default 0 so that we don't have to set the parameter when the parachute isn't deployed.)
    :return: Current drag force on the lander in cartesian coordinates
    """
    C_D = 1
    area_sc = mission.spacecraft_area
    vel_norm = vel/np.linalg.norm(vel)
    w = find_w(pos)
    v_drag = vel-w
    F_D_abs = (1 / 2) * rho_atm * C_D * (area_sc + area_parachute) * np.linalg.norm(v_drag) ** 2
    F_D = - F_D_abs*vel_norm
    return F_D


def terminal_velocity(area_parachute):
    """
    Calculates absolute terminal velocity based on area of parachutes/lander
    :param area_parachute: Area of parachutes im m^2
    :return: Absolute terminal velocity
    """
    area_sc = mission.spacecraft_area
    m = mission.spacecraft_mass
    rho_atm = system.atmospheric_densities[planet_idx]
    C_D = 1
    v = np.sqrt(2*m*g/(rho_atm*C_D*(area_sc + area_parachute)))
    return v


def parachute_area(vel_term):
    """
    Calculates required parachute area of lander for a given terminal velocity
    :param vel_term: Terminal velocity
    :return: Total area of parachutes
    """
    m = mission.spacecraft_mass
    rho_atm = system.atmospheric_densities[planet_idx]
    C_D = 1
    area_tot = 2*m*g/(rho_atm*C_D*(vel_term**2))
    area_parachutes = area_tot-mission.spacecraft_area
    return area_parachutes



