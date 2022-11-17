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
stable_orbit = 88515# insert code here

# Initializing system
username = "janniesc"
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut1 = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])
shortcut = SpaceMissionShortcuts(mission, [stable_orbit])
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

    # Boosting after entering space to have required velocity for trajectory to destination planet
    v_req = reference_vel[0]
    delta_v = v_req - v
    inter_trav.boost(delta_v)

    # Coasting until we are close enough to planet
    # Checking deviation from simulated orbit at regular intervals and performing correctional burn if necessary
    coast_time = 0.01
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

    # Checking (almost) continuously if we are at the correct position for orbit injection
    t_111, r111, v111 = inter_trav.orient
    r222 = dest_planet_orbit[int(t_111/dt)]
    v222 = (dest_planet_orbit[int(t_111/dt)+1] - dest_planet_orbit[int(t_111/dt)])/dt
    while np.dot((r111-r222), (v111-v222)) > 0.01:
        t_111, r111, v111 = inter_trav.orient
        r222 = dest_planet_orbit[int(t_111 / dt)]
        v222 = (dest_planet_orbit[int(t_111 / dt) + 1] - dest_planet_orbit[int(t_111 / dt)]) / dt
        inter_trav.coast(0.000001)

    # Calculating and adjusting velocity to enter orbit
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

    # Loading Orbits from earlier simulations
    with np.load("../../Part 2/Python Code/planet_trajectories.npz") as file:
        time = file['times']
        planet_positions = file['planet_positions']
    launch_planet_orbit = np.transpose(planet_positions[:, launch_planet_idx, :])
    dest_planet_orbit = np.transpose(planet_positions[:, dest_planet_idx, :])  # Creating orbits for the planets

    # Calculating min distance between planets, best time for launch and estimated time for reaching destination
    min_dist_planets, time_index0 = find_closest_orbit(launch_planet_orbit, dest_planet_orbit)
    dt = time[1]  # Defining time of launch and an approximation when we will be reaching our destination
    time_index1 = int(time_index0 + 10_000)
    t1 = time_index1 * dt
    t0 = time_index0 * dt


    # Shortcut to launch rocket from planet
    shortcut1.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t0, 3_000_000, launch_angle, 392_000)
    fuel_consumed, t0, r0, v0 = shortcut1.get_launch_results()
    mission.verify_launch_result(r0)
    mission.verify_manual_orientation(r0, v0, 37.01285168461271)
    # print(f"Exit Velocity: {np.linalg.norm(v0)}, {v0}")

    # Finding a trajectory and necessary velocity after reaching space to end up at our destination planet
    reference_traj, reference_vel = find_velocity_for_trajectory(r0, t0, t1, dt, (launch_angle * np.pi / 180), (0.00001 * np.pi / 180), np.linalg.norm(v0), 0.000001, dest_planet_orbit, dest_planet_mass, time_index0, time_index1)

    # Plotting orbits of planets and trajectory of spacecraft during simulated travel to destination
    plt.plot(launch_planet_orbit[time_index0:(time_index0 + 10000), 0], launch_planet_orbit[time_index0:(time_index0 + 10000), 1], label="Launch Planet orbit")
    plt.plot(dest_planet_orbit[time_index0:(time_index0 + 10000), 0], dest_planet_orbit[time_index0:(time_index0 + 10000), 1], label="Destination Planet orbit")
    plt.axis("equal")
    plt.legend()
    plt.xlabel("x-position [AU]")
    plt.ylabel("y-position [AU]")
    plt.savefig("../Figures/inter_trav.png")
    plt.show()

    # Defining some variables for interplanetary travel
    int_trav = mission.begin_interplanetary_travel()
    r_planet_orb = np.linalg.norm(dest_planet_orbit[time_index1])
    max_deviation = 0.0001
    l = r_planet_orb*np.sqrt(dest_planet_mass/(10*system.star_mass))
    print(f"l: {l}")

    # Starting interplanetary travel
    # t_in_orbit, r_in_orbit, v_in_orbit = interplanetary_travel(r0, v0, t0, t1, dt, max_deviation, dest_planet_idx, dest_planet_orbit, reference_traj, reference_vel, l)

    # Shortcut to stable orbit
    orbital_height = 75_000_000
    orbit_rad = utils.m_to_AU(system.radii[dest_planet_idx]*1000 + orbital_height)
    orbital_angle = 0
    shortcut.place_spacecraft_in_stable_orbit(t1, orbital_height, orbital_angle, dest_planet_idx)

    v = np.sqrt(G*dest_planet_mass/orbit_rad)
    r_orb = orbital_height + system.radii[dest_planet_idx]*1000

    print(f"v_orb: {np.linalg.norm([3.54142, -1.94128])}")
    print(f"r_orb: {utils.m_to_AU(np.linalg.norm(r_orb))}")

    BB = ti.time()
    print(BB-AA)


