import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from numba import njit
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
initial_positions = system.initial_positions
masses = system.masses
planet_file = np.load('../../Part 2/Python Code/planet_trajectories.npz')
planet_positions = planet_file['planet_positions']



@njit
def trajectory(t: float, T: float, dt: float, v0: np.ndarray, r0: np.ndarray):
    '''
    Finds the trajectory of the space shuttle \n
    Returns the final time (yr), velocity(AU) and position(AU)
    '''
    N = int((T-t)/dt)
    r = np.zeros((N, 2))
    r[0] = r0
    v = np.zeros((N, 2))
    v[0] = v0

    '''
    We simplify the system by adding all positions, velocities and masses of the planets and star
    '''

    # Integration loop
    # As the shuttle mass is so small in comparison to the mass of the system, we disregard the gravitational force from the shuttle on the system.
    for i in range(N-1):
        # Calculating the acceleration of each body individually
        a = -G *(star_mass)/(np.linalg.norm(r[i])**3) * r[i] # Acceleration from star
        for p in range(8):
            time_index = int(t/dt + i)
            r_shuttle_p = r[i] - planet_positions[:, p, time_index]
            a += -G *(masses[p])/(np.linalg.norm(r_shuttle_p)**3) * r_shuttle_p

        vh = v[i] + a * dt / 2
        r[i+1] = r[i] + vh * dt

        # Calculating the acceleration of each body individually
        a = -G *(star_mass)/(np.linalg.norm(r[i+1])**3) * r[i+1] # Acceleration from star
        for p in range(8):
            time_index = int(t/dt + i+1)
            r_shuttle_p = r[i+1] - planet_positions[:, p, time_index]
            a += -G *(masses[p])/(np.linalg.norm(r_shuttle_p)**3) * r_shuttle_p
            
        v[i+1] = vh + a * dt / 2

    return T, v, r


if __name__ == "__main__":
    planet_file = np.load('../../Part 2/Python Code/planet_trajectories.npz')
    # Nice to have the same time index as used by SpaceMission
    dt = planet_file['times'][1]
    planet_r = utils.km_to_AU(system.radii[0])
    v0 = np.array([0, 1])
    r0 = np.array([planet_r + system.initial_positions[0, 0], 0])
    t, v, r = trajectory(0, 7, dt, v0, r0)
    plt.scatter(r[0, 0], r[0, 1], label='Beginning')
    plt.plot(r[:, 0], r[:, 1], label="Shuttle trajectory")
    
    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
    radius = utils.km_to_AU(system.star_radius)
    x = radius * np.cos( angle ) 
    y = radius * np.sin( angle ) 
    plt.plot(x,y, label = 'Star')
    
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.grid()
    plt.axis('equal')
    plt.legend()
    print(system.initial_positions[:, 0])
    print(system.initial_positions[:, 1])
    plt.show()
