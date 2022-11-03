import numpy as np
import scipy as sp  
import os
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from numba import njit
from Part_5_A import trajectory

# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)

@njit
def find_closest_distance(planet_trajectory1, planet_trajectory2):
    '''
    Finds the time and distance where to bodies are the closest \n
    Main use is to find the best time to launch the shuttle \n
    Returns smallest distance and time index
    '''
    smallest_distance = np.linalg.norm(planet_trajectory1[:,0] - planet_trajectory2[:,0])
    time_index = 0
    for t in range(len(time)):
        distance = np.linalg.norm(planet_trajectory1[:,t] - planet_trajectory2[:,t])
        if distance < smallest_distance:
            smallest_distance = distance
            time_index = t    
            
    return smallest_distance, time_index

def check_close_enough():
    '''
    Just for a quick check of how close the shuttle is to the target
    '''
    min_distance = 10000; index = 0
    for t in range(time_index, final_time_index):
        distance = np.linalg.norm(target_planet_trajectory[:,t] - shuttle_position[int(t - time_index),:])
        if distance < min_distance:
            min_distance = distance
            index = t
    

    planet_m = system.masses[1]
    star_m = system.star_mass
    distance_to_star = np.linalg.norm(shuttle_position[int(index - time_index)])
    l = distance_to_star*np.sqrt(planet_m/10*star_m)
    
    if min_distance < l:
        print(f'Min distance: {min_distance}')
        print('Close enough!!')
        plt.scatter(shuttle_position[int(index - time_index), 0], shuttle_position[int(index - time_index),1], label = 'Shuttle position close enough')
        plt.scatter(target_planet_trajectory[0,index], target_planet_trajectory[1,index], label = 'Target planet position close enough')
        print(f'Total time of travel is {(final_time_index-index)*dt: .2f} years')  
        return True    
        


if __name__ == "__main__":
    planet_file = np.load('planet_trajectories.npz')
    planet_positions = planet_file['planet_positions']
    time = planet_file['times']
    home_planet_trajectory = planet_positions[:,0,:]
    target_planet_trajectory = planet_positions[:,1,:]
    smallest_distance, time_index = find_closest_distance(home_planet_trajectory, target_planet_trajectory)
    dt = time[1]  # Nice to have the same time index as used by SpaceMission. Multiplying by 100 to get faster plot

    r_home_target =  target_planet_trajectory[:,time_index] - home_planet_trajectory[:,time_index] # vector from home planet to target planet we can use to direct the initial velocity vector
    
    t = time_index*dt
    T = time_index*dt + 5
    angle_r_home_target = (np.arctan(r_home_target[1]/r_home_target[0])) + np.pi 
    # Finds the angle of the vector pointing from home to target. We use this to calculate the start position of our rocket launch
    angle = np.pi * 5/4
    
    planet_r = utils.km_to_AU(system.radii[0])
    '''
    To place the rocket at the closest position between the planets and on the surface of the home planet. we find the angle of the vector pointing between the planets. We use this angle and polar coordinates to place the rocket at the correct position.
    '''
    
    r0 = home_planet_trajectory[:,time_index] + np.array([0,-planet_r]) # Sets the rocket on the bottom of the planet
    
    
    final_time_index = int(T/dt)
    plt.plot(home_planet_trajectory[0,time_index:final_time_index], home_planet_trajectory[1,time_index:final_time_index], label = 'Home planet trajectory')
    plt.plot(target_planet_trajectory[0,time_index:final_time_index], target_planet_trajectory[1,time_index:final_time_index], label = 'Target planet trajectory')
    
    # angle = 261.75*np.pi/180
    # v0 = np.array([np.cos(angle), np.sin(angle)]) * 4.75
    # final_time, shuttle_velocity, shuttle_position = trajectory(t, T, dt, v0, r0)
    # plt.plot(shuttle_position[:25000,0], shuttle_position[:25000,1])
    # check_close_enough()


    angles = np.linspace(260 * np.pi/180, 265 * np.pi/180, 5)
    velocities = np.linspace(4.70, 4.80, 5)
    finished = False
    for angle in angles:
        for velocity in velocities:
            v0 = np.array([np.cos(angle), np.sin(angle)]) * velocity
            final_time, shuttle_velocity, shuttle_position = trajectory(t, T, dt, v0, r0)
            if check_close_enough():
                print('success')
                good_enough_velocity = velocity     
                good_enough_angle = angle
                print(f'{good_enough_angle*180/np.pi} degrees were good enough')
                print(f'{good_enough_velocity} was fast enough')
                plt.plot(shuttle_position[:25000,0], shuttle_position[:25000,1], label = 'Shuttle trajectory')
                finished = True
                
            if finished:
                break
        if finished:
            break
                

                




    # min_angle = 261.7750520*np.pi/180
    # max_angle = 261.7750524*np.pi/180 
    
    # angles = np.linspace(min_angle, max_angle, 10)
    # for i in range(len(angles)):
    #     v0 = np.array([np.cos(angles[i]), np.sin(angles[i])]) * 4.75 - 0.00000005
    #     final_time, shuttle_velocity, shuttle_position = trajectory(t, T, dt, v0, r0)
    #     plt.plot(shuttle_position[:25000,0], shuttle_position[:25000,1], color =  colors[i])
    #     check_distance_from_target()
        
    # min_vel = 4.75 - 0.00005
    # max_vel = 4.75 - 0.00006
    # velocities = np.linspace(min_vel, max_vel, 10)
    # angle = (max_angle + min_angle) / 2
    # for v in velocities:
    #     v0 = np.array([np.cos(angle), np.sin(angle)])  * v
    #     final_time, shuttle_velocity, shuttle_position = trajectory(t, T, dt, v0, r0)
    #     plt.plot(shuttle_position[:25000,0], shuttle_position[:25000,1])
    #     check_close_enough()
    # for i in range(time_index, final_time_index, 10):
        
    #     plt.scatter(home_planet_trajectory[0,i], home_planet_trajectory[1,i], label = f'Home planet at time index {i}', color = 'green')
    #     plt.scatter(target_planet_trajectory[0,i], target_planet_trajectory[1,i], label = f'Target planet at time index {i}', color = 'green')
    #     plt.scatter(shuttle_position[int(i - time_index),0], shuttle_position[int(i - time_index),1], label = f'Shuttle position at time index {i}', color = 'pink')
        
    
    
    
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.legend(loc = 'upper right')
    plt.show()
    
    
    
    '''
    Plotting the trajectory of home and target planet and where they are the closest
    '''
    plt.plot(home_planet_trajectory[0,:], home_planet_trajectory[1,:])
    plt.plot(target_planet_trajectory[0,:], target_planet_trajectory[1,:])
    plt.scatter(home_planet_trajectory[0,time_index], home_planet_trajectory[1,time_index], label = 'Home planet')
    plt.scatter(target_planet_trajectory[0,time_index], target_planet_trajectory[1,time_index], label = 'Target planet')
    plt.axis('equal')
    plt.legend()
    plt.show()