import numpy as np
import sys
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
# sys.path.append("../../Part 1/Python Code")
# from Rocket_launch import launch_rocket  # It works when running, even if it shows an error in the editor!
from numba import njit

# Initializing mission, system and shortcuts
seed = 36874
code_launch_results = 83949
code_escape_trajectory = 74482
stable_orbit = 88515
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [stable_orbit])
shortcut1 = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])
planet_idx = 1

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


def cart_to_spherical(x, y, z):
    """
    Conversion from cartesian coordinates to spherical coordinates
    :param x: x-coordinate
    :param y: y-coordinate
    :param z: z-coordinate
    :return: Spherical coordinates r, theta (theta measured from z-axis), phi
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))
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


def decrease_orbit(r0, r1, land_seq, planet_idx=1):
    """
    Performs maneuver to decrease orbital altitude to given altitude using Hohmann transfer
    :param r0: Radius of initial (higher) orbit [m]
    :param r1: Radius if target (lower) orbit [m]
    :param land_seq: Landing sequence instance
    :param planet_idx: Planet Index
    :return: None
    """
    m_planet = system.masses[planet_idx] * 1.98847e30  # Planet mass
    GM = (6.6743015 * 10 ** (-11))*m_planet  # Standard gravitational parameter
    v0_abs = np.sqrt(GM * m_planet / r0)  # Absolute velocity in the initial circular orbit
    T_orb0 = (2 * np.pi * r0) / v0_abs  # Calculating orbital period
    delta_v0 = -np.sqrt(GM/r1)*(np.sqrt((2*r0)/(r0+r1))-1)  # Calculating required boost to enter Hohmann transfer orbit
    delta_v1 = -np.sqrt(GM / r0) * (np.sqrt((2 * r1) / (r0 + r1)) - 1)  # Calculating required boost to enter circular target orbit

    t, pos, vel = land_seq.orient()
    vel_norm = vel / np.linalg.norm(vel)
    land_seq.boost(delta_v0 * vel_norm)  # Boosting to go from the initial corculat orbit into the Hohmann transfer orbit.

    land_seq.fall(T_orb0/4)  # Falling for 1/4 of the orbital time (until we are at the periapsis)

    t, pos, vel = land_seq.orient()
    vel_norm = vel / np.linalg.norm(vel)
    land_seq.boost(delta_v1 * vel_norm)  # Boosting to go from the Hohmann transfer orbit into the circular target orbit.
    t, pos, vel = land_seq.orient()
    return t, pos, vel


def find_actual_coordinates(curr_coords, time_elapsed, planet_idx=1, cartesian=False):
    """
    Calculates at which coordinates the input position was at time 0.
    :param curr_coords: Current coordinates in spherical form in radians (or cartesian if cartesian is set to True) (Phi is 0 at z-axis)
    :param time_elapsed: Point of time from start of landing sequence in seconds
    :param planet_idx: Planet index
    :return: Spherical coordinates of the position we are over now at time 0.
    """
    if cartesian:  # Converting to spherical if current coordinates are cartesian
        x = curr_coords[0]
        y = curr_coords[1]
        z = curr_coords[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))
        curr_coords = np.array([r, theta, phi])

    r_planet = system.radii[planet_idx] * 1000  # Planet radius
    omega = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)  # Angular velocity of atmosphere
    new_phi = (curr_coords[2] + (omega * time_elapsed))  # Calculating new theta angle
    new_coords = np.array([r_planet, curr_coords[1], new_phi])  # New coordinates
    return new_coords


def find_new_coordinates(curr_coords, time_ahead, planet_idx=1, cartesian=False):
    """
    Calculates at which coordinates the input position will be at a future point in time.
    :param curr_coords: Current coordinates in spherical form in radians (or cartesian if cartesian is set to True) (Phi is 0 at z-axis)
    :param time_ahead: Time interval in seconds (from now) how long in the future the coordinate prediction will be
    :param planet_idx: Planet index
    :return: Spherical coordinates (in radians) of the position we are over now at a time [time_interval] from now.
    """
    if cartesian: # Converting to spherical if current coordinates are cartesian
        x = curr_coords[0]
        y = curr_coords[1]
        z = curr_coords[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))
        curr_coords = np.array([r, theta, phi])

    r_planet = system.radii[planet_idx] * 1000  # Planet radius
    omega = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)  # Angular velocity of atmosphere
    new_theta = (curr_coords[1] + (omega * time_ahead)) % (2 * np.pi)  # Calculating new theta angle
    new_coords = np.array([r_planet, new_theta, curr_coords[2]])  # New Coordinates
    return new_coords


def verify_constant_orbit_height(land_seq, max_diff=100):
    """
    Verifying that the orbital altitude stays the same
    :param land_seq: Landing sequence instance
    :param max_diff: Maximum difference in orbital altitude (in meters)
    :return: Boolean whether orbital height is stable
    """
    r0_list = []  # List for orbital altitude measurements
    for i in range(1_000):  # Measuring orbital altitude in short intervals to determine average altitude
        t0, pos0, vel0 = land_seq.orient()
        r0_list += [np.linalg.norm([pos0[0], pos0[1], pos0[2]])]
        land_seq.fall(100)

    land_seq.fall(1_000_000)  # Orbiting for a while before checking again

    r1_list = []  # List for orbital altitude measurements - Same method as before
    for i in range(1_000):  # Measuring orbital altitude in short intervals to determine average altitude
        t1, pos1, vel1 = land_seq.orient()
        r1_list += [np.linalg.norm([pos1[0], pos1[1], pos1[2]])]
        land_seq.fall(100)

    r0_avg = sum(r0_list)/len(r0_list)  # Determining average height based on measurements for measurement series 0
    r1_avg = sum(r1_list) / len(r1_list)  # Determining average height based on measurements for measurement series 1

    print(f"Difference is {r1_avg-r0_avg:.2f} m")  # Printing difference
    if abs(r1_avg-r0_avg) > max_diff:  # Checking whether difference is larger than max_diff (given maximum difference)
        print("Difference in radius of the orbit is too large!")
        return False
    else:
        print("Orbit is stable!")
        return True


def image_landing_site(land_seq, idx=0, planet_idx=1):
    t, pos_cart, vel_cart = land_seq.orient()
    # r, theta, phi = cart_to_spherical(pos_cart[0], pos_cart[1], pos_cart[2])
    act_coords = find_actual_coordinates(pos_cart, t, cartesian=True)
    pos = f"Landing_site{idx}: {act_coords[0]}, {act_coords[1]}, {act_coords[2]}\n"
    with open("landing_area_coords.txt", "a") as file:
        file.write(pos)
    land_seq.look_in_direction_of_planet(planet_idx)
    land_seq.take_picture(f"landing_area{idx}.xml")




if __name__ == "__main__":
    landing_seq = mission.begin_landing_sequence()  # Creating landing sequence instance
    # print(verify_constant_orbit_height(landing_seq))  # Verifying stability of orbital height
    landing_seq.start_video()
    picture_num = 15  # Number of landing site pictures to be taken
    picture_delay = 60  # Delay between when pictures are taken in seconds
    with open("landing_area_coords.txt", "w") as file:  # Creating and clearing file with landing site coordinates
        file.write("Landing Site Index, Radius [m], Theta [rad], Phi [rad]\n")  # Writing header to file
    for idx in range(picture_num):
        image_landing_site(landing_seq, idx=idx)
        landing_seq.fall(picture_delay)
    # landing_seq.fall(10000)
    landing_seq.finish_video()
