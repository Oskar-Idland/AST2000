import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

G = 4 * np.pi ** 2

# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)
star_mass = system.star_mass
sum_planet_mass = np.sum(system.masses)


def trajectory(t: float, T: float, dt: float, v0: np.ndarray, r0: np.ndarray):
    '''
    Finds the trajectory of the space shuttle \n
    Returns the final time (yr), velocity(AU) and position(AU)
    '''
    N = int((T - t) / dt)
    r = np.zeros((N, 2))
    r[0] = r0
    v = np.zeros((N, 2))
    v[0] = v0

    '''
    We simplify the system by adding all positions, velocities and masses of the planets and star
    '''
    m_system = star_mass + sum_planet_mass
    r_system = np.array([0.0, 0.0])
    v_system = np.array([0.0, 0.0])
    com = np.array([0.0, 0.0])
    tot_momentum = 0
    for body in range(system.number_of_planets):  # Finds 
        planet_mass = system.masses[body]
        planet_position = system.initial_positions[:, body]
        com += planet_mass * planet_position

        planet_velocity = system.initial_velocities[:, body]
        momentum = planet_mass * planet_velocity
        tot_momentum += momentum
    com *= 1 / m_system
    r_system = com
    v_system = tot_momentum / m_system
    # Integration loop
    # As the shuttle mass is so small in comparison to the mass of the system, we disregard the gravitational force from the shuttle on the system.  
    for i in range(N - 1):
        # Calculating the acceleration as a N-body system
        r_shuttle_system = r[i] - r_system
        a = -G * (m_system) / (np.linalg.norm(r_shuttle_system) ** 3) * r_shuttle_system

        # Calculating the acceleration of each body individually
        # a = -G *(star_mass)/(np.linalg.norm(r[i])) * r[i] # Acceleration from star
        # for p in range(8):
        #     r_shuttle_p = r[i] - system.initial_positions[:,p]
        #     a += -G *(system.masses[p])/(np.linalg.norm(r_shuttle_p)) * r_shuttle_p  

        vh = v[i] + a * dt / 2
        r[i + 1] = r[i] + vh * dt
        r_system = r_system + v_system * dt  # To be commented out when calculating each body individually.

        r_shuttle_system = r[i + 1] - r_system  # To be commented out when calculating each body individually.

        # Calculating the acceleration of each body individually
        # a = -G *(star_mass)/(np.linalg.norm(r[i])) * r[i] # Acceleration from star
        # for p in range(8):
        #     r_shuttle_p = r[i] - system.initial_positions[:,p]
        #     a += -G *(system.masses[p])/(np.linalg.norm(r_shuttle_p)) * r_shuttle_p

        # Calculating the acceleration as a N-body system
        a = -G * (m_system) / (np.linalg.norm(r_shuttle_system) ** 3) * r_shuttle_system
        v[i + 1] = vh + a * dt / 2

    return T, v, r


if __name__ == "__main__":
    planet_file = np.load('../../Part 2/Python Code/planet_trajectories.npz')
    dt = planet_file['times'][1]  # Nice to have the same time index as used by SpaceMission
    planet_r = utils.km_to_AU(system.radii[0])
    v0 = np.array([1, 1])
    r0 = np.array([planet_r + system.initial_positions[0, 0], 0])
    t, v, r = trajectory(0, 2, dt, v0, r0)
    plt.scatter(r[0, 0], r[0, 1], label='Beginning')
    plt.plot(r[:, 0], r[:, 1], label="Shuttle trajectory")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.grid()
    plt.axis('equal')
    plt.legend()
    print(system.initial_positions[:, 0])
    print(system.initial_positions[:, 1])
    plt.show()
