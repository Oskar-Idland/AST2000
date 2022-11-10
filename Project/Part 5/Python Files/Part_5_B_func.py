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
            idx = i
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
    plt.plot(sc_traj[:10000, 0], sc_traj[:10000, 1], c="k")
    plt.scatter(planet_traj[index, 0], planet_traj[index, 1])

    return close_enough, index, min_distance


# @njit
def find_velocity_for_trajectory(r0, t0, t1, dt, median_angle, angle_span, median_velocity, velocity_span, dest_planet_orbit, dest_planet_mass, time_index0, time_index1):
    # median_angle = 250.226 * np.pi / 180   # Setting Middle, max and min values for angles and velocities
    # angle_span = 0.001 * np.pi / 180
    # median_velocity = 5.12745  # Absolute value!
    # velocity_span = 0.0001
    angles = np.linspace(median_angle - angle_span, median_angle + angle_span, 11)  # Creating arrays with different angles and velocities to iterate over and test
    velocities = np.linspace(median_velocity - velocity_span, median_velocity + velocity_span, 11)

    good_vel_ang = []

    for angle in angles:
        print("\nAngle:")
        print(angle * 180 / np.pi)
        for velocity in velocities:
            print("Velocity:")
            print(velocity)
            v0 = velocity * np.array([np.cos(angle), np.sin(angle)])
            # A1 = ti.time()
            final_time, sc_velocity, sc_position = trajectory(t0, t1, dt, v0, r0)
            # sc_position = utils.m_to_AU(sc_position1)
            # B1 = ti.time()
            close_enough, index, dist = check_close_enough(sc_position, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)
            # C1 = ti.time()
            # print(B1-A1)
            # print(C1-B1)
            if close_enough:
                good_vel_ang.append([angle, velocity, dist])

    if len(good_vel_ang) == 0:
        print('No results were good enough!')

    else:
        best_result = good_vel_ang[0]
        for i in range(len(good_vel_ang)):
            angle, vel, dist1 = good_vel_ang[i]
            rocket_vel = vel * np.array([np.cos(angle), np.sin(angle)])
            print(f"Good enough initial values are Angle, Abs. Velocity, Velocity, Distance")
            print(utils.rad_to_deg(angle), vel, rocket_vel, dist1)
            if dist1 < best_result[2]:
                best_result = good_vel_ang[i]

        print(f"\nThe best solution found had an angle, speed and minimum distance of")
        print(utils.rad_to_deg(best_result[0]), best_result[1], best_result[2])

        fin_time, sc_vel, sc_pos = trajectory(t0, t1, dt, best_result[1] * np.array([np.cos(best_result[0]), np.sin(best_result[0])]), r0)
        return sc_pos, sc_vel

if __name__ == "__main__":
    AA = ti.time()
    launch_planet_idx = 0
    dest_planet_idx = 1
    dest_planet_mass = system.masses[dest_planet_idx]

    with np.load("../../Part 2/Python Code/planet_trajectories.npz") as file:
        time = file['times']
        planet_positions = file['planet_positions']
    launch_planet_orbit = np.transpose(planet_positions[:, launch_planet_idx, :])
    dest_planet_orbit = np.transpose(planet_positions[:, dest_planet_idx, :])  # Creating orbits for the planets

    median_angle = 250.226 * np.pi / 180  # Setting Middle, max and min values for angles and velocities
    angle_span = 0.001 * np.pi / 180
    median_velocity = 5.12745  # Absolute value!
    velocity_span = 0.0001

    min_dist_planets, time_index00 = find_closest_orbit(launch_planet_orbit, dest_planet_orbit)  # Calculating min distance between planets, best time for launch and estimated time for reaching destination
    dt = time[1]
    time_index1 = int(time_index00 + 5 / dt)
    t00 = time_index00 * dt

    r0, v01, t0, total_time = launch_rocket(mission.spacecraft_mass, 392_000, 6_000_000, t_orbit_launch=t00, printing=False, store=False)

    time_index0 = int(total_time/dt)
    t1 = time_index1 * dt

    plt.plot(launch_planet_orbit[time_index0:(time_index1 - 1), 0], launch_planet_orbit[time_index0:(time_index1 - 1), 1])
    plt.scatter(launch_planet_orbit[time_index0, 0], launch_planet_orbit[time_index0, 1], c="r")
    plt.scatter(launch_planet_orbit[time_index1, 0], launch_planet_orbit[time_index1, 1], c="k")
    plt.plot(dest_planet_orbit[time_index0:(time_index1 - 1), 0], dest_planet_orbit[time_index0:(time_index1 - 1), 1])
    plt.scatter(dest_planet_orbit[time_index0, 0], dest_planet_orbit[time_index0, 1], c="r")
    plt.scatter(dest_planet_orbit[time_index1, 0], dest_planet_orbit[time_index1, 1], c="k")

    find_velocity_for_trajectory(r0, total_time, t1, dt, median_angle, angle_span, median_velocity, velocity_span, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)
    plt.axis("equal")
    plt.show()
    BB = ti.time()
    print(f"The program took {(BB - AA):.2f} seconds")
