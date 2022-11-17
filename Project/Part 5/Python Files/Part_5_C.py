import numpy as np
import sys
import time as ti
import ast2000tools.utils as utils
from numba import njit
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
sys.path.append("../../Part 1/Python Code")
from Rocket_launch import launch_rocket  # It works when running, even if it shows an error in the editor!
from Part_5_A import trajectory
from Part_5_B_func import *
from ast2000tools.shortcuts import SpaceMissionShortcuts
seed = 36874
code_launch_results = 83949
code_escape_trajectory = 74482

# Initializing system
username = "janniesc"
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])
G = 4 * np.pi ** 2

def find_orbit_inj_velocity(planet_idx, dist):
    mass = system.masses[planet_idx]
    return np.sqrt((G*mass/dist))


def interplanetary_travel(r0, v0, t0, t1_approx, dt, max_dev, dest_planet_idx, dest_planet_orbit, reference_traj, reference_vel, l):
    inter_trav = mission.begin_interplanetary_travel()
    t, r, v = inter_trav.orient()  # Orienting ourselves after launch
    plt.scatter(r[0], r[1])
    plt.show()
    distance = mission.measure_distances()[dest_planet_idx]
    dest_planet_mass = system.masses[dest_planet_idx]

    # Creating Reference Trajectory to destination planet
    v_req = reference_vel[0]
    delta_v = v_req - v
    print("HELLO")
    print(v, v_req, delta_v)
    inter_trav.boost(delta_v)
    print(v_req)
    print(inter_trav.orient()[2])

    # Coasting until we are close enough to planet
    coast_time = 0.00001
    while distance > l:
        inter_trav.coast(coast_time)
        t_curr, r_curr, v_curr = inter_trav.orient()
        if np.linalg.norm(reference_traj[int((t_curr-t0)/dt)] - r_curr) > max_dev:
            v_curr_abs = np.linalg.norm(v_curr)
            heading = np.arctan(v_curr[1]/v_curr[0]) + np.pi
            v_req1 = find_velocity_for_trajectory(r_curr, t_curr, t1_approx, dt, heading, 10 * np.pi / 180, v_curr_abs, 0.001, dest_planet_orbit, dest_planet_mass, int(t_curr/dt), int(t1_approx/dt))
            print(heading / np.pi * 180)
            print(v_curr)
            delta_v1 = v_req1-v_curr
            inter_trav.boost(delta_v1)

    # Checking when we are at the correct position for orbit injection
    t_111, r111, v111 = inter_trav.orient
    r222 = dest_planet_orbit[int(t_111/dt)]
    v222 = (dest_planet_orbit[int(t_111/dt)+1] - dest_planet_orbit[int(t_111/dt)])/dt
    while np.dot((r111-r222), (v111-v222)) > 0.01:
        t_111, r111, v111 = inter_trav.orient
        r222 = dest_planet_orbit[int(t_111 / dt)]
        v222 = (dest_planet_orbit[int(t_111 / dt) + 1] - dest_planet_orbit[int(t_111 / dt)]) / dt

    # Adjusting velocity
    v_req_orbit = find_orbit_inj_velocity(dest_planet_idx, np.linalg.norm(mission.measure_distances()[dest_planet_idx]))
    delta_v = np.linalg.norm(v222)-v_req_orbit
    heading = np.arccos(v222[0]/np.linalg.norm(v222))
    inter_trav.boost(delta_v*np.array([np.cos(heading), np.sin(heading)]))
    return inter_trav.orient




if __name__ == "__main__":
    AA = ti.time()
    launch_planet_idx = 0  # Defining launch and destination planet
    dest_planet_idx = 1
    dest_planet_mass = system.masses[dest_planet_idx]
    launch_angle = 260.483012

    with np.load("../../Part 2/Python Code/planet_trajectories.npz") as file:  # Loading Orbits from earlier simulations
        time = file['times']
        planet_positions = file['planet_positions']
    launch_planet_orbit = np.transpose(planet_positions[:, launch_planet_idx, :])
    dest_planet_orbit = np.transpose(planet_positions[:, dest_planet_idx, :])  # Creating orbits for the planets


    min_dist_planets, time_index0 = find_closest_orbit(launch_planet_orbit, dest_planet_orbit)  # Calculating min distance between planets, best time for launch and estimated time for reaching destination
    dt = time[1]  # Defining time of launch and an approximation when we will be reaching our destination
    time_index1 = int(time_index0 + (3 / dt))
    t1_approx = time_index1 * dt
    t0 = time_index0 * dt


    # Launching spacecraft to find position, velocity and time after the launch + Verifying launch results and orientation
    # launch_rocket(mission.spacecraft_mass, 392_000, 6_000_000, t_orbit_launch=t0, launch_angle=launch_angle, printing=False, store=True)
    shortcut.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t0, 3_000_000, launch_angle, 392_000)
    fuel_consumed, t0, r0, v0 = shortcut.get_launch_results()
    mission.verify_launch_result(r0)
    mission.verify_manual_orientation(r0, v0, 37.01285168461271)
    print(f"EXIT VELOCITY: {np.linalg.norm(v0)}, {v0}")


    # Plotting orbits of planets and trajectory of spacecraft during simulated travel to destination
    # plt.plot(launch_planet_orbit[time_index0:(time_index1 - 1), 0], launch_planet_orbit[time_index0:(time_index1 - 1), 1])
    plt.plot(launch_planet_orbit[time_index0:(time_index0 + 100), 0], launch_planet_orbit[time_index0:(time_index0 + 100), 1])
    # plt.scatter(launch_planet_orbit[time_index0, 0], launch_planet_orbit[time_index0, 1], c="r")
    # plt.scatter(launch_planet_orbit[time_index1, 0], launch_planet_orbit[time_index1, 1], c="k")
    # plt.plot(dest_planet_orbit[time_index0:(time_index1 - 1), 0], dest_planet_orbit[time_index0:(time_index1 - 1), 1])
    # plt.plot(dest_planet_orbit[time_index0:(time_index0 + 5000), 0], dest_planet_orbit[time_index0:(time_index0 + 5000), 1])
    # plt.scatter(dest_planet_orbit[time_index0, 0], dest_planet_orbit[time_index0, 1], c="r")
    # plt.scatter(dest_planet_orbit[time_index1, 0], dest_planet_orbit[time_index1, 1], c="k")

    reference_traj, reference_vel = find_velocity_for_trajectory(r0, t0, t1_approx, dt, (launch_angle * np.pi / 180), (0.00001 * np.pi / 180), np.linalg.norm(v0), 0.000001, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)

    plt.axis("equal")
    # plt.show()

    # Defining some variables for interplanetary travel
    max_deviation = 0.0001
    dest_planet_idx = 1
    l = 1

    # Starting interplanetary travel
    interplanetary_travel(r0, v0, t0, t1_approx, dt, max_deviation, dest_planet_idx, dest_planet_orbit, reference_traj, reference_vel, l)

    BB = ti.time()
    print(BB-AA)


