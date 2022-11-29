import numpy as np
import sys
import ast2000tools.utils as utils
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
# sys.path.append("../../Part 1/Python Code")
# from Rocket_launch import launch_rocket  # It works when running, even if it shows an error in the editor!
from numba import njit

# Initializing mission, system and shortcuts
username = "janniesc"
seed = 36874
code_launch_results = 83949
code_escape_trajectory = 74482
stable_orbit = 88515
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [stable_orbit])
shortcut1 = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])
planet_idx = 1
G = 6.6743015e-11

# FOR SIMPLICITY WE WILL ONLY MOVE ON THE XY-PLANE

# Defining some variables from earlier parts
dt = 8.666669555556518e-05
time0 = 2762487 * dt
orbital_height0 = 1_000_000
orbital_angle0 = 0

# Putting Spacecraft into low stable orbit (requires verification of launch and orientation first)
launch_angle = 260.483012
t_launch = 2752487 * dt
shortcut1.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t_launch, 3_000_000, launch_angle, 392_000)
fuel_consumed, t_after_launch, r_after_launch, v_after_launch = shortcut1.get_launch_results()
mission.verify_manual_orientation(r_after_launch, v_after_launch, 37.01285168461271)
shortcut.place_spacecraft_in_stable_orbit(time0, orbital_height0, orbital_angle0, planet_idx)


m_planet = system.masses[planet_idx] * 1.98847e30
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


def drag(pos, vel, densities, area_parachute=0):
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
    r = round(np.linalg.norm(pos))
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


def plot_planet():
    planet_radius = system.radii[1]*1000
    angles = np.linspace(0, 2*np.pi, 2000)
    x = planet_radius*np.cos(angles)
    y = planet_radius*np.sin(angles)
    plt.plot(x, y)


if __name__ == "__main__":
    densities = np.load("../../Part 6/Densities.npy")
    landing_seq = mission.begin_landing_sequence()  # Creating landing sequence instance
    t0, pos0, vel0 = landing_seq.orient()
    boost0_strength = 200
    vel0_norm = vel0/np.linalg.norm(vel0)
    landing_seq.boost(-boost0_strength*vel0_norm)  # Decreasing tangential velocity to initiate fall into atmosphere

    N = 1000
    pos_list = np.zeros([N, 3])
    for i in range(N):
        ti, posi, veli = landing_seq.orient()
        pos_list[i] = posi
        print(f"Velocity: {np.linalg.norm(veli)}")
        print(f"Position: {np.linalg.norm(posi)/1000}")
        landing_seq.fall(2)

    t0, pos0, vel0 = landing_seq.orient()
    boost0_strength = 100
    vel0_norm = pos0 / np.linalg.norm(pos0)
    landing_seq.boost(boost0_strength * vel0_norm)

    pos_list1 = np.zeros([N, 3])
    for i in range(N):
        ti, posi, veli = landing_seq.orient()
        pos_list1[i] = posi
        print(f"Velocity: {np.linalg.norm(veli)}")
        print(f"Position: {np.linalg.norm(posi) / 1000}")
        landing_seq.fall(1)

    t0, pos0, vel0 = landing_seq.orient()
    boost0_strength = -3000
    vel0_norm = vel0 / np.linalg.norm(vel0)
    landing_seq.boost(boost0_strength * vel0_norm)

    pos_list2 = np.zeros([N, 3])
    for i in range(N):
        ti, posi, veli = landing_seq.orient()
        pos_list2[i] = posi
        print(f"Velocity: {np.linalg.norm(veli)}")
        print(f"Position: {np.linalg.norm(posi) / 1000}")
        landing_seq.fall(0.5)

    plot_planet()
    plt.plot(pos_list[:, 0], pos_list[:, 1])
    plt.plot(pos_list1[:, 0], pos_list1[:, 1])
    plt.plot(pos_list2[:, 0], pos_list2[:, 1])
    plt.axis("equal")
    plt.show()


