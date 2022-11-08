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

# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)


def interplanetary_travel(r0, v0, t0, t1_approx, dt, max_dev, dest_planet_idx):
    inter_trav = mission.begin_interplanetary_travel()
    t, r, v = inter_trav.orient()  # Orienting ourselves after launch
    distance = mission.measure_distances()[dest_planet_idx]

    sim_trajectory = trajectory(t, t1_approx, dt, v, r)  # Creating Reference Trajectory to destination planet
    v_req = calculate_rocket_velocity(sim_trajectory)  # TODO: FIX!!!
    delta_v = v_req - v
    inter_trav.boost(delta_v)

    coast_time = 0.1
    while distance > l:
        inter_trav.coast(coast_time)
        t_curr, r_curr, v_curr = inter_trav.orient()
        if abs(sim_trajectory[t_curr] - r_curr) > max_dev:
            v_req1 = calculate_rocket_velocity()
            delta_v1 = v_req1-v_curr
            inter_trav.boost(delta_v1)



if __name__ == "__main__":
    AA = ti.time()
    launch_planet_idx = 0
    dest_planet_idx = 1
    dest_planet_mass = system.masses[dest_planet_idx]*1.989e30

    with np.load("../../Part 2/Python Code/planet_trajectories.npz") as file:
        time = file['times']
        planet_positions = file['planet_positions']
    launch_planet_orbit = np.transpose(planet_positions[:, launch_planet_idx, :])
    dest_planet_orbit = np.transpose(planet_positions[:, dest_planet_idx, :])  # Creating orbits for the planets

    min_dist_planets, time_index00 = find_closest_orbit(launch_planet_orbit, dest_planet_orbit)  # Calculating min distance between planets, best time for launch and estimated time for reaching destination
    dt = time[1]
    time_index1 = int(time_index00 + 5 / dt)
    t00 = time_index00 * dt

    r0, v01, t0, total_time = launch_rocket(mission.spacecraft_mass, 392_000, 6_000_000, t_orbit_launch=t00, printing=False, store=False)

    time_index0 = int(total_time/dt)
    t1 = time_index1 * dt

    find_velocity_for_trajectory(r0, total_time, t1, dt, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)
    BB = ti.time()
    print(BB-AA)


t1_approx = 6
max_deviation = 0.1
dest_planet_idx = 1

r0, v0, t0 = launch_rocket(store=True)  # TODO: Add parameters
interplanetary_travel(r0, v0, t0, t1_approx, dt, max_deviation, dest_planet_idx)

