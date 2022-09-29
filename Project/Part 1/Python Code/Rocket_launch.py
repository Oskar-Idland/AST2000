import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.constants import G



username = "janniesc"
seed = utils.get_seed(username)
N = 200000
dt = 0.01

system = SolarSystem(seed)
mission = SpaceMission(seed)
planet_idx = 0
dry_mass = mission.spacecraft_mass
fuel_mass = 20000  # Guess!
fuel_consumption = 50  # Kg/s
thrust_force = 600000  # Newton
wet_mass = dry_mass + fuel_mass
mass_home_planet = system.masses[0]*1.989e30
rotational_period = system.rotational_periods[0]  # In days
radius_home_planet = system.radii[0]*1000
end_i = 0

planet_theta = 0


tang_vel_planet = 2*np.pi*(radius_home_planet)/(rotational_period*24*3600)
pos = np.zeros([N, 2])
vel = np.zeros([N, 2])
acc = np.zeros([N, 2])

pos[0] = [np.cos(planet_theta)*radius_home_planet, np.sin(planet_theta)*radius_home_planet]
vel[0] = [-np.sin(planet_theta)*tang_vel_planet, np.cos(planet_theta)*tang_vel_planet]


for i in range(N-1):
    r = np.linalg.norm(pos[i])
    pos_unit = pos[i]/r
    # vel_unit = vel[i]/np.linalg.norm(vel[i])
    theta = np.arccos(pos[i][0]/r)
    thrust = np.array(thrust_force*pos_unit)
    g = -G*mass_home_planet*wet_mass/r**2 * pos_unit
    acc[i] = (g+thrust)/wet_mass
    vel[i+1] = vel[i] + acc[i]*dt
    pos[i+1] = pos[i] + vel[i]*dt
    wet_mass = wet_mass-(fuel_consumption*dt)

    if wet_mass <= dry_mass:
        end_i = i
        print(f"Ran out of fuel after {end_i*dt} seconds :/")
        break

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
        break

plt.plot(pos[:end_i, 0], pos[:end_i, 1], color="k")
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.axis("equal")
plt.grid()
plt.savefig("../Figures/Launch_plot.png")
plt.show()


# Changing coordinates to solar coordinate system
def chg_coords(planet_idx, coord_in, vel_in, elapsed_time_in):
    """
    Changes coordinates from home planet system to solar system coordinate system and units
    :param planet_idx: Index of Home planet
    :param coord_in: Coordinates from launch simulation in meters
    :param vel_in: Velocity from launch simulation in meters/second
    :param elapsed_time_in: Elapsed time in seconds
    :return: Tuple with Position, Velocity and time in the Solar system coordinates and units
    """
    planet_coords = system.initial_positions[:, planet_idx]
    planet_vel = system.initial_velocities[:, planet_idx]
    meters_per_au = 1.495978707*10**11
    sec_per_yr = 365*24*3600
    coord_out = planet_coords + (coord_in/meters_per_au) + planet_vel*utils.s_to_yr(elapsed_time_in)
    vel_out = planet_vel + (vel_in*(sec_per_yr/meters_per_au))
    elapsed_time_out = elapsed_time_in/sec_per_yr
    return coord_out, vel_out, elapsed_time_out


sol_sys_coords, sol_sys_vel, sol_sys_time = chg_coords(planet_idx, exit_coords, exit_vel, elapsed_time)

print("\nIn solar system coordinate system:")
print(f"Position: ({sol_sys_coords[0]:E}, {sol_sys_coords[1]:E}) AU")
print(f"Position: ({sol_sys_vel[0]:E}, {sol_sys_vel[1]:E}) AU/Year")
print(f"Elapsed Time: {sol_sys_time:E} Years\n")

mission.set_launch_parameters(600000, 50, 20000, 360, [system.initial_positions[0, 0] + (utils.m_to_AU(radius_home_planet)), 0], 0)
mission.launch_rocket()
mission.verify_launch_result(sol_sys_coords)