import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.constants import G


username = "janniesc"
seed = utils.get_seed(username)
N = 200000
dt = 0.02

system = SolarSystem(seed)
mission = SpaceMission(seed)
dry_mass = mission.spacecraft_mass
fuel_mass = 20000  # Guess!
fuel_consumption = 50  # Kg/s
thrust_force = 600000  # Newton
wet_mass = dry_mass + fuel_mass
mass_home_planet = system.masses[0]*1.989e30
rotational_period = system.rotational_periods[0]  # In days
radius_home_planet = system.radii[0]*1000
end_i = 0

tang_vel_planet = 2*np.pi*(radius_home_planet)/(rotational_period*24*3600)
pos = np.zeros([N, 3])
vel = np.zeros([N, 3])
acc = np.zeros([N, 3])

pos[0] = [0, radius_home_planet, 0]
vel[0] = [-tang_vel_planet, 0, 0]


for i in range(N-1):
    r = np.linalg.norm(pos[i])
    theta = np.arccos(pos[i][1]/r)
    thrust = np.array([-thrust_force*np.sin(theta), thrust_force*np.cos(theta), 0])
    g = G*mass_home_planet*wet_mass/r**2 * np.array([np.sin(theta), -np.cos(theta), 0])
    acc[i] = (g+thrust)/wet_mass
    vel[i+1] = vel[i] + acc[i]*dt
    pos[i+1] = pos[i] + vel[i]*dt
    wet_mass = wet_mass-(fuel_consumption*dt)


    if wet_mass <= dry_mass:
        print(i)
        end_i = i
        break
    if np.linalg.norm(vel[i]) >= np.sqrt(2*G*mass_home_planet/r):
        print("SPACE!!!")
        print(f"Final position: x: {int(pos[i][0])} m, y: {int(pos[i][1])} m, z: {int(pos[i][2])} m")
        print(f"Final velocity: v_x: {int(vel[i][0])} m/s, v_y: {int(vel[i][1])} m/s, v_z: {int(vel[i][2])} m/s")
        print(f"Time elapsed: {(dt * i / 60):.2f} min")
        print(f"Remaining fuel: {wet_mass-dry_mass} Kg")
        end_i = i
        break

plt.plot(pos[:end_i, 0], pos[:end_i, 1])
plt.show()



