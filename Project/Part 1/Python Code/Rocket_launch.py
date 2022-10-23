import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.constants import G
from scipy import interpolate


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
    pos_func, vel_func = create_orbit_func(planet_idx)
    coord_out = pos_func(orbit_launch_time + elapsed_time_yrs) + utils.m_to_AU(coord_in)
    vel_out = vel_func(orbit_launch_time + elapsed_time_yrs) + utils.m_pr_s_to_AU_pr_yr(vel_in)
    return coord_out, vel_out, elapsed_time_yrs


# Defining variables
username = "janniesc"
seed = utils.get_seed(username)
N = 200_000
dt = 0.01
system = SolarSystem(seed)
mission = SpaceMission(seed)
planet_idx = 0

dry_mass = mission.spacecraft_mass  # 1100 Kg
fuel_mass = 1_500  # Guess! Need at least 198'472 Kg of fuel to reach space (with no fuel left)
thrust_force = 50_000  # Newton # Needs to be at least wet_mass*9.81
thrust_per_box = 8.113886899686883e-11
mass_flow_rate_per_box = 2.413509643703512e-15
num_of_boxes = thrust_force/thrust_per_box
fuel_consumption = mass_flow_rate_per_box*num_of_boxes  # Kg/s

wet_mass = dry_mass + fuel_mass
mass_home_planet = system.masses[0]*1.989e30
rotational_period = system.rotational_periods[0]  # In days
radius_home_planet = system.radii[0]*1000
end_i = 0
t_orbit_launch = 0  # In Years
planet_theta = 2*np.pi*t_orbit_launch/utils.s_to_yr(rotational_period*24*3600)
tang_vel_planet = 2*np.pi*radius_home_planet/(rotational_period*24*3600)


# Creating and initialising arrays
pos = np.zeros([N, 2])
vel = np.zeros([N, 2])
# acc = np.zeros([N, 2])
pos[0] = [np.cos(planet_theta)*radius_home_planet, np.sin(planet_theta)*radius_home_planet]
vel[0] = [-np.sin(planet_theta)*tang_vel_planet, np.cos(planet_theta)*tang_vel_planet]

print(f"Wetmass: {wet_mass}")
# Integration loop
for i in range(N-1):
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

    # wet_mass = wet_mass - (fuel_consumption * dt)




    # Checking if we run out of fuel
    if wet_mass <= dry_mass:
        end_i = i
        print(f"Ran out of fuel after {end_i*dt} seconds :/")
        break

    # Checking if we reached escape velocity
    if np.linalg.norm(vel[i]) >= np.sqrt(2*G*mass_home_planet/r):
        exit_coords = np.array([pos[i][0], pos[i][1]])
        exit_vel = np.array([vel[i][0], vel[i][1]])
        end_i = i
        elapsed_time = end_i*dt
        print("SPACE!!!")
        print(f"Final position: x: {int(exit_coords[0])} m, y: {int(exit_coords[1])} m")
        print(f"Final velocity: v_x: {int(exit_vel[0])} m/s, v_y: {int(exit_vel[1])} m/s")
        print(f"Time elapsed: {(elapsed_time/60):.2f} min")
        print(f"Final mass of spacecraft: {wet_mass:.2f} Kg")
        print(f"Remaining fuel: {wet_mass-dry_mass} Kg")
        print(f"Remaining Burn Time: {(wet_mass-dry_mass)/(fuel_consumption*60):.2f} min")
        break


# Plotting
# plt.plot(pos[:end_i, 0], pos[:end_i, 1], color="k")
plt.plot(pos[:10000, 0], pos[:10000, 1], color="k")
# plt.scatter(create_orbit_func(planet_idx)[0](t_orbit_launch)[0] + radius_home_planet*np.cos(planet_theta), create_orbit_func(planet_idx)[0](t_orbit_launch)[1] + radius_home_planet*np.sin(planet_theta), c="r")  # Plotting launch position
plt.xlabel("x-position")
plt.ylabel("y-position")
# plt.axis("equal")
plt.grid()
plt.savefig("../Figures/Launch_plot.png")
plt.show()

# Changing coordinates to solar coordinate system
sol_sys_coords, sol_sys_vel, sol_sys_time = chg_coords(planet_idx, exit_coords, exit_vel, elapsed_time, t_orbit_launch)

print("\nIn solar system coordinate system:")
print(f"Position: ({sol_sys_coords[0]:E}, {sol_sys_coords[1]:E}) AU")
print(f"Position: ({sol_sys_vel[0]:E}, {sol_sys_vel[1]:E}) AU/Year")
print(f"Elapsed Time: {sol_sys_time:E} Years\n")
print(f"Number of Boxes: {num_of_boxes:e}")
print(f"Mass flow rate: {fuel_consumption} Kg/s\n")
# Verifying results
launch_position = create_orbit_func(planet_idx)[0](t_orbit_launch) + utils.m_to_AU(radius_home_planet)*np.array([np.cos(planet_theta), np.sin(planet_theta)])
mission.set_launch_parameters(thrust_force, fuel_consumption, fuel_mass, 1200, launch_position, t_orbit_launch)
mission.launch_rocket()
mission.verify_launch_result(sol_sys_coords)

