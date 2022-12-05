# This file is based on the file Part_5_B, but most of the main code has been rewritten into a function

import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from numba import njit
import time as ti
from Part_5_A import trajectory
sys.path.append("../../Part 1/Python Code")
from Rocket_launch import launch_rocket  # It works when running, even if it shows an error in the editor!
from ast2000tools.shortcuts import SpaceMissionShortcuts

'''
----Results----
Total time of travel:           4.75 years
The best solution found had a speed of 5.877665391162627 and  a velocity vector [-2.43231345 -5.35077581]

'''

# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
mission = SpaceMission(seed)
system = SolarSystem(seed)
star_m = system.star_mass
code_launch_results = 83949
code_escape_trajectory = 74482
shortcut1 = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])



@njit
def find_closest_orbit(planet_trajectory1, planet_trajectory2):
    '''
    Finds the time and distance where to bodies are the closest \n
    Main use is to find the best time to launch the shuttle \n
    Returns smallest distance and time index
    '''

    dist_arr = planet_trajectory2-planet_trajectory1
    idx = 0
    least_value = np.linalg.norm(dist_arr[0])
    for i in range(1, len(dist_arr)):
        if np.linalg.norm(dist_arr[i]) < least_value:
            least_value = np.linalg.norm(dist_arr[i])
            idx = i + 5000
    return dist_arr[idx], idx


# @njit
def check_close_enough(sc_traj, planet_traj, planet_mass, time_index0, time_index1):
    '''
    Just for a quick check of how close the shuttle is to the target
    '''
    min_distance = 1E20  # Arbitrary large integer
    index = 0
    for t in range(time_index0, (time_index1-1)):
        distance = np.linalg.norm(planet_traj[t, :] - sc_traj[int(t - time_index0), :])
        if distance < min_distance:
            min_distance = distance
            index = t

    distance_to_star = np.linalg.norm(sc_traj[int(index-time_index0)])  # Calculating how close the shuttle needs to be
    l = distance_to_star * np.sqrt(planet_mass / (10 * star_m))
    if min_distance < l:
        close_enough = True
        print("Shuttle is close enough! Min Distance [AU]:")
        print(min_distance)

    else:
        close_enough = False

    print("Distance")
    print(min_distance, close_enough)
    plt.plot(sc_traj[:7550, 0], sc_traj[:7550, 1], c="k")
    plt.scatter(sc_traj[index-time_index0, 0], sc_traj[index-time_index0, 1])
    plt.scatter(planet_traj[index, 0], planet_traj[index, 1])

    return close_enough, index, min_distance


# @njit
def find_velocity_for_trajectory(r0, t0, t1, dt, median_angle, angle_span, median_velocity, velocity_span, dest_planet_orbit, dest_planet_mass, time_index0, time_index1):
    angles = np.linspace(median_angle - angle_span, median_angle + angle_span, 11)  # Creating arrays with different angles and velocities to iterate over and test
    velocities = np.linspace(median_velocity - velocity_span, median_velocity + velocity_span, 1)

    good_vel_ang = []  # List of angles and velocities which give a good enough trajectory

    for angle in angles:  # Iterating over initial angles
        print("\nAngle:")
        print(angle * 180 / np.pi)
        for velocity in velocities:  # Iterating over initial velocities
            print("Velocity:")
            print(velocity)
            v0 = velocity * np.array([np.cos(angle), np.sin(angle)])
            print(v0)
            final_time, sc_velocity, sc_position = trajectory(t0, t1, dt, v0, r0)  # Calculating trajectory of the current initial angle and velocity
            close_enough, index, dist = check_close_enough(sc_position, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)  # Checking if the calculated trajectory comes close enough to the planet
            if close_enough:
                good_vel_ang.append([angle, velocity, dist])  # Appending to list if close enough to planet

    if len(good_vel_ang) == 0:
        print('No results were good enough!')

    else:
        # Iterating over satisfactory trajectories and picking the one which comes closest to the planet
        best_result = good_vel_ang[0]
        for i in range(len(good_vel_ang)):
            angle, vel, dist1 = good_vel_ang[i]
            rocket_vel = vel * np.array([np.cos(angle), np.sin(angle)])
            print(f"Good enough initial values are Angle, Abs. Velocity, Velocity, Distance")
            print(utils.rad_to_deg(angle), vel, rocket_vel, dist1)
            if dist1 < best_result[2]:
                best_result = good_vel_ang[i]

        print(f"\nThe best solution found had an angle, speed and minimum distance of")
        print(utils.rad_to_deg(best_result[0]), best_result[1], best_result[2])  # Printing the best initial values

        fin_time, sc_vel, sc_pos = trajectory(t0, t1, dt, best_result[1] * np.array([np.cos(best_result[0]), np.sin(best_result[0])]), r0)  #  Recalculating the best trajectory and returning it
        return sc_pos, sc_vel

if __name__ == "__main__":
    AA = ti.time()
    launch_planet_idx = 0
    dest_planet_idx = 1
    dest_planet_mass = system.masses[dest_planet_idx]

    # Loading previously calculated planet orbits from a file
    with np.load("../../Part 2/Python Code/planet_trajectories.npz") as file:
        time = file['times']
        planet_positions = file['planet_positions']
    launch_planet_orbit = np.transpose(planet_positions[:, launch_planet_idx, :])  # Orbit for the launch planet
    dest_planet_orbit = np.transpose(planet_positions[:, dest_planet_idx, :])  # Orbit for the destination planet

    # Finding optimal time for launch
    min_dist_planets, time_index0 = find_closest_orbit(launch_planet_orbit, dest_planet_orbit)  # Calculating min distance between planets, best time for launch and estimated time for reaching destination
    dt = time[1]
    t0 = time_index0 * dt
    time_index1 = int(time_index0 + 10_000)
    t1 = time_index1 * dt

    # Using the rocket launch shortcut to place our spacecraft into escape trajectory
    launch_angle = 200
    shortcut1.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t0, 3_000_000, launch_angle, 392_000)
    r0, v0, t0 = launch_rocket(mission.spacecraft_mass, 392_000, 6_000_000, t_orbit_launch=t0, launch_angle=launch_angle, printing=False, store=False)

    # Setting some approximate initial values for the trajectory
    median_angle = np.arctan2(v0[1], v0[0])
    angle_span = 0.1 * np.pi / 180
    median_velocity = np.linalg.norm(v0)
    velocity_span = 1e-25
    print(v0)
    print((median_angle/np.pi*180) + 360)

    # Plotting
    plt.plot(launch_planet_orbit[time_index0:(time_index0 + 5_000), 0], launch_planet_orbit[time_index0:(time_index0 + 5_000), 1])
    plt.scatter(launch_planet_orbit[time_index0, 0], launch_planet_orbit[time_index0, 1], c="r")
    plt.plot(dest_planet_orbit[(time_index0 + 5000):(time_index0 + 10000), 0], dest_planet_orbit[(time_index0 + 5000):(time_index0 + 10000), 1])

    # Finding best initial velocity for a trajectory from our initial position
    find_velocity_for_trajectory(r0, t0, t1, dt, median_angle, angle_span, median_velocity, velocity_span, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)

    # Plotting
    plt.axis("equal")
    plt.show()
    BB = ti.time()
    print(f"The program took {(BB - AA):.2f} seconds")
