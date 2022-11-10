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
G = 4 * np.pi ** 2

def find_orbit_inj_velocity(planet_idx, dist):
    mass = system.masses[planet_idx]
    return np.sqrt((G*mass/dist))




def interplanetary_travel(r0, v0, t0, t1_approx, dt, max_dev, dest_planet_idx, dest_planet_orbit, reference_traj, reference_vel, l):
    inter_trav = mission.begin_interplanetary_travel()
    t, r, v = inter_trav.orient()  # Orienting ourselves after launch
    distance = mission.measure_distances()[dest_planet_idx]
    dest_planet_mass = system.masses[dest_planet_idx]

    # Creating Reference Trajectory to destination planet
    v_req = reference_vel[0]
    delta_v = v_req - v
    inter_trav.boost(delta_v)
    print(v_req)
    print(inter_trav.orient[2])

    # Coasting until we are close enough to planet
    coast_time = 0.001
    while distance > l:
        inter_trav.coast(coast_time)
        t_curr, r_curr, v_curr = inter_trav.orient()
        if abs(reference_traj[t_curr/dt] - r_curr) > max_dev:
            v_curr_abs = np.lialg.norm(v_curr)
            heading = np.arccos(v_curr[0]/v_curr_abs)
            v_req1 = find_velocity_for_trajectory(r_curr, t_curr, t1_approx, dt, heading, 0.001 * np.pi / 180, v_curr_abs, 0.001, dest_planet_orbit, dest_planet_mass, t_curr/dt, t1_approx/dt)
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
    launch_planet_idx = 0
    dest_planet_idx = 1
    dest_planet_mass = system.masses[dest_planet_idx]

    with np.load("../../Part 2/Python Code/planet_trajectories.npz") as file:
        time = file['times']
        planet_positions = file['planet_positions']
    launch_planet_orbit = np.transpose(planet_positions[:, launch_planet_idx, :])
    dest_planet_orbit = np.transpose(planet_positions[:, dest_planet_idx, :])  # Creating orbits for the planets

    min_dist_planets, time_index00 = find_closest_orbit(launch_planet_orbit, dest_planet_orbit)  # Calculating min distance between planets, best time for launch and estimated time for reaching destination
    dt = time[1]
    time_index1 = int(time_index00 + 5 / dt)
    t00 = time_index00 * dt

    r0, v01, launch_duration, t0 = launch_rocket(mission.spacecraft_mass, 392_000, 6_000_000, t_orbit_launch=t00, printing=False, store=True)

    time_index0 = int(t0/dt)
    t1_approx = time_index1 * dt

    reference_traj, reference_vel = find_velocity_for_trajectory(r0, t0, t1_approx, dt, (250.226 * np.pi / 180), (0.001 * np.pi / 180), 5.12745, 0.0001, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)

    max_deviation = 0.1
    dest_planet_idx = 1
    l = 1

    interplanetary_travel(r0, v01, t0, t1_approx, dt, max_deviation, dest_planet_idx, dest_planet_orbit, reference_traj, reference_vel, l)

    BB = ti.time()
    print(BB-AA)


