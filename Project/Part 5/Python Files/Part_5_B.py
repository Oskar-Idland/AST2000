import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from numba import njit
from Part_5_A import trajectory

π =np.pi


'''
----Results----
Total time of travel:           4.75 years
The best solution found had a speed of 5.877665391162627 and  a velocity vector [-2.43231345 -5.35077581]

'''

# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
planet_m = system.masses[1]
star_m = system.star_mass

@njit
def find_closest_orbit(planet_trajectory1, planet_trajectory2):
    '''
    Finds the time and distance where to bodies are the closest \n
    Main use is to find the best time to launch the shuttle \n
    Returns smallest distance and time index
    '''
    smallest_distance = np.linalg.norm(planet_trajectory1[:, 0] - planet_trajectory2[:, 0])
    time_index = 0
    for t in range(len(time)):
        distance = np.linalg.norm(planet_trajectory1[:, t] - planet_trajectory2[:, t])
        if distance < smallest_distance:
            smallest_distance = distance
            time_index = t

    return smallest_distance, time_index


@njit
def min_distance_shuttle_target():
    min_distance = 1E20
    index = 0
    for t in range(time_index, final_time_index):
        distance = np.linalg.norm(
            target_planet_trajectory[:, t] - shuttle_position[int(t - time_index), :])
        if distance < min_distance:
            min_distance = distance
            index = t

    return index, final_time_index, min_distance


def check_close_enough():
    '''
    Just for a quick check of how close the shuttle is to the target
    '''

    min_distance = 1E20 # Arbitrary large integer 
    index = 0
    for t in range(time_index, final_time_index):
        distance = np.linalg.norm(target_planet_trajectory[:, t] - shuttle_position[int(t - time_index), :])
        if distance < min_distance:
            min_distance = distance
            index = t


    # Rask versjon av løkken over, men returnerer feil index
    # index, final_time_index, min_distance = min_distance_shuttle_target()

    # plt.scatter(shuttle_position[int(index - time_index), 0], shuttle_position[int(
    #     index - time_index), 1])

    # plt.scatter(target_planet_trajectory[0, index],
    #             target_planet_trajectory[1, index])


    distance_to_star = np.linalg.norm(
        shuttle_position[int(index - time_index)])
    l = distance_to_star*np.sqrt(planet_m/(10*star_m))
    print(f'\t \t Min distance: {min_distance: .2e}')
    close_enough = False
    if min_distance < l:
        close_enough = True
        print('Close enough!!')
        plt.scatter(shuttle_position[int(index - time_index), 0], shuttle_position[int(
            index - time_index), 1], label='Shuttle position close enough')
        plt.scatter(target_planet_trajectory[0, index], target_planet_trajectory[1,
                    index], label='Target planet position close enough',)
        print(
            f'Total time of travel is {(index - time_index)*dt: .2f} years, and begins at year {time_index*dt}')
        return close_enough, index
    
    else:
        return close_enough, index

def calculate_rocket_velocity(speed, angle, time):
    '''
    As the file containing the planets velocity is in cartesian coordinates we must convert our calculated vector into cartesian coordinates as well. Then subract this from our calculated vector from the simulation to get the actual value the rocket needs. 
    '''
    planet_vel = np.load(os.path.join('Orbits\Planet_1.npz'))['velocity']
    time_index = int(time/dt)
    planet_vel = planet_vel[time_index, :] # Velocity vector of home planet
    x = speed*np.cos(angle)
    y = speed*np.sin(angle)
    good_enough_vel = np.array([x,y])
    rocket_vel = good_enough_vel - planet_vel
    return rocket_vel


if __name__ == "__main__":
    planet_file = np.load(os.path.join('planet_trajectories.npz'))
    planet_positions = planet_file['planet_positions']
    time = planet_file['times']
    home_planet_trajectory = planet_positions[:, 0, :]
    home_planet_initial_velocity = system.initial_velocities[:, 0]
    target_planet_trajectory = planet_positions[:, 1, :]
    smallest_distance, time_index = find_closest_orbit(
        home_planet_trajectory, target_planet_trajectory)
    # Nice to have the same time index as used by SpaceMission. Multiplying by 100 to get faster plot
    dt = time[1]

    # vector from home planet to target planet we can use to direct the initial velocity vector
    r_home_target = target_planet_trajectory[:,
                                             time_index] - home_planet_trajectory[:, time_index]

    t = time_index*dt       # Converts index to years
    T = time_index*dt + 5   # Converts index to years

    # Finds the angle of the vector pointing from home to target. We use this to calculate the start position of our rocket launch

    planet_r = utils.km_to_AU(system.radii[0])
    '''
    To place the rocket at the closest position between the planets and on the surface of the home planet. we find the angle of the vector pointing between the planets. We use this angle and polar coordinates to place the rocket at the correct position.
    '''

    # Sets the rocket on the bottom of the planet
    r0 = home_planet_trajectory[:, time_index] + np.array([0, -planet_r])

    final_time_index = int(T/dt)
    plt.plot(home_planet_trajectory[0, time_index:final_time_index],
             home_planet_trajectory[1, time_index:final_time_index], label='Home planet trajectory')
    plt.plot(target_planet_trajectory[0, time_index:final_time_index],
             target_planet_trajectory[1, time_index:final_time_index], label='Target planet trajectory')

    median_angle = 267.5 * np.pi / 180 
    span_angle = 1 * np.pi / 180
    median_velocity = 9
    span_velocity = .25
    angles = np.linspace(median_angle - span_angle, median_angle + span_angle, 17)
    velocities = np.linspace(median_velocity - span_velocity, median_velocity + span_velocity, 21)
    finished = False
    good_enough_values = []
    
    for angle in angles:
        print(f'{angle*180/np.pi: .3f}')
        for velocity in velocities:
            print(f'\t{velocity}')
            v0 = np.array([np.cos(angle), np.sin(angle)]) * velocity
            final_time, shuttle_velocity, shuttle_position = trajectory(
                t, T, dt, v0, r0)
            close_enough, index = check_close_enough()
            if close_enough:
                good_enough_velocity = velocity
                good_enough_angle = angle
                print(f'{good_enough_angle*180/np.pi: .2f} degrees were good enough')
                print(f'{good_enough_velocity: .2f} was fast enough')
                # finished = True
                plt.plot(
                shuttle_position[:10000, 0], shuttle_position[:10000, 1])
                good_enough_values.append((good_enough_velocity, good_enough_angle))

            if finished:
                break
        if finished:
            break

    
    if len(good_enough_values) == 0:
        print('No results were good enough!')
    
    else:        
        best_result = good_enough_values[0]
        for result in good_enough_values:
            vel, angle = result
            rocket_vel = calculate_rocket_velocity(vel, angle, t)
            rocket_speed = np.linalg.norm(rocket_vel)
            print(rocket_speed)
            if rocket_speed < best_result[0]:
                best_result = [rocket_speed, rocket_vel]
                
        print(f'The best solution found had a speed of {best_result[0]} and  a velocity vector {best_result[1]}')
        
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.show()
    
    

    '''
    Plotting the trajectory of home and target planet and where they are the closest
    '''
    # plt.plot(home_planet_trajectory[0,:], home_planet_trajectory[1,:])
    # plt.plot(target_planet_trajectory[0,:], target_planet_trajectory[1,:])
    # plt.scatter(home_planet_trajectory[0,time_index], home_planet_trajectory[1,time_index], label = 'Home planet')
    # plt.scatter(target_planet_trajectory[0,time_index], target_planet_trajectory[1,time_index], label = 'Target planet')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()
