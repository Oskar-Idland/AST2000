import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.constants import G


username = "janniesc"
seed = utils.get_seed(username)
N = 10000
dt = 0.01

system = SolarSystem(seed)
mission = SpaceMission(seed)
dry_mass = mission.spacecraft_mass
fuel_mass = 1000  # Guess!
fuel_consumption = 5  # Kg/s
thrust_force = 100000000000  # Newton
wet_mass = dry_mass + fuel_mass
mass_home_planet = system.masses[0]*1.989e30
rotational_period = system.rotational_periods[0]  # In days
radius_home_planet = system.radii[0]

tang_vel_planet = 2*np.pi*(radius_home_planet*1000)/(rotational_period*24*3600)
pos = np.zeros([N, 3])
vel = np.zeros([N, 3])
acc = np.zeros([N, 3])

pos[0] = [0, radius_home_planet*1000, 0]
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
    print(pos[i])
    print(theta)
    print(thrust)
    print("")

    if wet_mass <= dry_mass:
        print(i)
        break

plt.plot(pos[0], pos[1])
plt.show()



