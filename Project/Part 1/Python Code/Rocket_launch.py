import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.constants import G
from scipy import interpolate
import time
from numba import njit
from snarvei_del1 import launch_rocket_shortcut

A = time.time()

username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)
N = 15_000_000
data = np.load(f"../../Part 2/Python Code/Orbits/Planet_{0}.npz")
t = data["time"]
dt = t[1]-t[0]
planet_idx = 0


def plot_circle(x, y, r):
    # Function to plot circle
    t_circ = np.linspace(0, 2*np.pi, 100)
    plt.plot(x+r*np.cos(t_circ), y+r*np.sin(t_circ), linewidth=2, label="Planet")


def create_orbit_func(planet_idx):
    # Loading numeric orbits from file
    data = np.load(f"../../Part 2/Python Code/Orbits/Planet_{planet_idx}.npz")
    t = data["time"]
    x = data["position"]
    v = data["velocity"]
    # INTERPOLATION
    position_function = interpolate.interp1d(t, x, axis=0, bounds_error=False, fill_value='extrapolate')
    velocity_function = interpolate.interp1d(t, v, axis=0, bounds_error=False, fill_value='extrapolate')
    return position_function, velocity_function


def chg_coords(planet_idx, coord_in, vel_in, elapsed_time_s, orbit_launch_time):
    """
    Changes coordinates from home planet system to solar system coordinate system and units
    :param planet_idx: Index of Home planet
    :param s_coord_in: Coordinates from launch simulation in meters
    :param s_vel_in: Velocity from launch simulation in meters/second
    :param elapsed_time_in: Elapsed time in seconds
    :return: Tuple with Position, Velocity and time in the Solar system coordinates and units
    """
    elapsed_time_yrs = utils.s_to_yr(elapsed_time_s)
    total_time = elapsed_time_yrs + orbit_launch_time
    pos_func, vel_func = create_orbit_func(planet_idx)
    coord_out = pos_func(orbit_launch_time + elapsed_time_yrs)[0] + utils.m_to_AU(coord_in)
    vel_out = vel_func(orbit_launch_time + elapsed_time_yrs) + utils.m_pr_s_to_AU_pr_yr(vel_in)
    return coord_out, vel_out, elapsed_time_yrs, total_time


@njit
def engine_performance(thrust, fuel_cons, m_init, speed_boost, dt=0.001):
    v = 0
    i = 0
    while v < speed_boost:
        m = m_init-((i + 0.5)*dt*fuel_cons)
        a = thrust/m
        v = v + a*dt
        i += 1
    return i*dt*fuel_cons


@njit
def integrate(pos, vel, dry_mass, wet_mass, thrust_force, fuel_consumption, mass_home_planet, dt):
    for i in range(N-1):
        # Using Leapfrog method
        wet_mass_i = wet_mass-(fuel_consumption*dt*i)

        r = np.linalg.norm(pos[i])
        thrust = thrust_force*pos[i]/r
        Fg = -G*mass_home_planet*wet_mass_i*pos[i]/r**3
        acc = (Fg+thrust)/wet_mass_i

        vh = vel[i] + acc * dt / 2
        pos[i+1] = pos[i] + vh*dt

        r = np.linalg.norm(pos[i+1])
        thrust = thrust_force*pos[i+1]/r
        Fg = -G*mass_home_planet*wet_mass_i*pos[i+1]/r**3
        acc = (Fg+thrust)/wet_mass_i

        vel[i+1] = vh + acc*dt/2

        # Checking if we run out of fuel
        if wet_mass <= dry_mass:
            end_i = i
            status = 1
            break

        # Checking if we reached escape velocity
        if np.linalg.norm(vel[i]) >= np.sqrt(2*G*mass_home_planet/r):
            exit_coords = np.array([pos[i][0], pos[i][1]])
            exit_vel = np.array([vel[i][0], vel[i][1]])
            end_i = i
            status = 0
            break

    return pos, vel, exit_coords, exit_vel, end_i, wet_mass_i, status


def launch_rocket(dry_mass, fuel_mass, thrust_force, estimated_time=1000, dt=0.001, N=200_000, thrust_per_box=5.290110991665214e-11, mass_flow_rate_per_box=2.413509643703512e-15, planet_index=0, t_orbit_launch=0, launch_angle=0, printing=False, store=True):
    num_of_boxes = thrust_force / thrust_per_box
    fuel_consumption = mass_flow_rate_per_box * num_of_boxes  # Kg/s
    wet_mass = dry_mass + fuel_mass
    mass_home_planet = system.masses[planet_index]*1.989e30
    rotational_period = system.rotational_periods[planet_index]  # In days
    radius_home_planet = system.radii[planet_index]*1000
    planet_theta = np.pi # 2*np.pi*t_orbit_launch/utils.s_to_yr(rotational_period*24*3600)
    tang_vel_planet = 2*np.pi*radius_home_planet/(rotational_period*24*3600)

    """
    # Creating and initialising arrays
    pos = np.zeros([N, 2])
    vel = np.zeros([N, 2])
    pos[0] = [np.cos(planet_theta)*radius_home_planet, np.sin(planet_theta)*radius_home_planet]
    vel[0] = [-np.sin(planet_theta)*tang_vel_planet, np.cos(planet_theta)*tang_vel_planet]

    # Simulating
    pos, vel, exit_coords, exit_vel, end_i, wet_mass_i, status = integrate(pos, vel, dry_mass, wet_mass, thrust_force, fuel_consumption, mass_home_planet, dt)

    # Changing coordinates to solar coordinate system
    sol_sys_coords, sol_sys_vel, sol_sys_time, total_time = chg_coords(planet_index, exit_coords, exit_vel, end_i*dt, t_orbit_launch)
    

    if store:
        # Verifying and storing results
        launch_position = create_orbit_func(planet_index)[0](t_orbit_launch) + utils.m_to_AU(radius_home_planet)*np.array([np.cos(planet_theta), np.sin(planet_theta)])
        plt.scatter(utils.AU_to_m(launch_position[0]), utils.AU_to_m(launch_position[1]))
        # verify_store_launch(launch_position, sol_sys_coords, thrust_force, fuel_mass, fuel_consumption, t_orbit_launch, estimated_time, dt)

    if printing:
        if status == 0:
            print("SPACE!!!")
            # print(f"Final position: x: {int(exit_coords[0])} m, y: {int(exit_coords[1])} m")
            # print(f"Final velocity: v_x: {int(exit_vel[0])} m/s, v_y: {int(exit_vel[1])} m/s")
            print(f"Final mass of spacecraft: {wet_mass:.2f} Kg")
            print(f"Remaining fuel: {wet_mass_i - dry_mass} Kg")
            print(f"Remaining Burn Time: {(wet_mass_i - dry_mass) / (fuel_consumption * 60):.2f} min")
            print(f"Duration of Launch: {int(end_i * dt // 60)} min {int(np.round(end_i * dt % 60, decimals=0))} sec\n")

        elif status == 1:
            print(f"Ran out of fuel after {end_i * dt} seconds :/")

        # Plotting
        # plt.plot(pos[:end_i, 0], pos[:end_i, 1], color="k")
        plt.plot(pos[:10000, 0], pos[:10000, 1], color="k", label="Launch")
        # plt.scatter(create_orbit_func(planet_index)[0](t_orbit_launch)[0] + radius_home_planet * np.cos(planet_theta), create_orbit_func(planet_index)[0](t_orbit_launch)[1] + radius_home_planet * np.sin(planet_theta), c="r")  # Plotting launch position
        plt.xlabel("x-position")
        plt.ylabel("y-position")
        plt.grid()
        plt.savefig("../Figures/Launch_plot.png")
        plt.show()

        print("\nIn solar system coordinate system:")
        print(f"Position: ({sol_sys_coords[0]:E}, {sol_sys_coords[1]:E}) AU")
        print(f"Velocity: ({sol_sys_vel[0]:E}, {sol_sys_vel[1]:E}) AU/Year")
        print(f"Elapsed Time: {sol_sys_time:E} Years\n")
        print(f"Number of Boxes: {num_of_boxes:e}")
        print(f"Mass flow rate: {fuel_consumption} Kg/s\n")
        print(f"Launch Results verified: {mission.launch_result_verified}")
    """

    height_above_suface = 0

    pos_after_launch, vel_after_launch, time, fuel_consumed = launch_rocket_shortcut(thrust_force, fuel_consumption, t_orbit_launch, height_above_suface, launch_angle, fuel_mass)

    return pos_after_launch, vel_after_launch, time


def verify_store_launch(launch_position, final_position, thrust_force, fuel_mass, fuel_consumption, t_orbit_launch1, estimated_time, dt):
    print(f"Launch_position_ver: {launch_position}")
    print(f"Launch_time_ver: {t_orbit_launch1}")
    mission.set_launch_parameters(thrust_force, fuel_consumption, fuel_mass, estimated_time, launch_position, t_orbit_launch1)
    mission.launch_rocket()
    mission.verify_launch_result(final_position)

    # Saving Launch parameters for later use
    launch_parameters = np.array([final_position[0], final_position[1], thrust_force, fuel_consumption, fuel_mass, estimated_time, launch_position[0], launch_position[1], t_orbit_launch1, dt])
    np.save("Launch_Parameters.npy", launch_parameters)


if __name__ == "__main__":
    # Defining variables


    dry_mass = mission.spacecraft_mass  # 1100 Kg
    fuel_mass = 392_000  # Guess!
    thrust_force = 6_000_000  # Newton # Needs to be at least wet_mass*9.81
    thrust_per_box = 5.290110991665214e-11  # 8.113886899686883e-11
    mass_flow_rate_per_box = 2.413509643703512e-15
    estimated_time = 1000
    launch_time = 200
    print(system.radii[0])
    print(utils.m_to_AU(49_587_900))
    launch_angle = 250
    print(launch_rocket(dry_mass, fuel_mass, thrust_force, estimated_time, dt, N, thrust_per_box, mass_flow_rate_per_box, planet_idx, launch_time, launch_angle=launch_angle, printing=True, store=True))

    B = time.time()
    print(f"\nThe program took {(B-A):.2f} seconds")
