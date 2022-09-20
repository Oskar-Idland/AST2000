import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
print('Imports done')

#CONSTANTS
G = 4*(np.pi**2)  #Gravitational Constant

#CALCULATED DATA
T = 13*20  #You want to find the planetary position from t=0 to t=T. How could you make an educated guess for T?
# Look at our solar system which planet has the same distance from the sun.
# Takes approx 12.81542 years


#SIMULATION PARAMETERS
N = 300  #Number of time steps
dt = T/N  #calculate time step from T and N

@jit(nopython = True)                       #Optional, but recommended for speed, check the numerical compendium
def integrate(T, dt, N, S_x0, S_y0, S_vx0, S_vy0, P_x0, P_y0, P_vx0, P_vy0, G, sun_mass, planet_mass):
    print('Integration started')
    t = np.linspace(0.0, T, N)
    x_Star = np.zeros((N, 2))
    x_Planet = np.zeros((N, 2))
    v_Star = np.zeros((N, 2))
    v_Planet = np.zeros((N, 2))

    x_Planet[0] = [P_x0, P_y0]
    v_Planet[0] = [P_vx0, P_vy0]
    x_Star[0] = [S_x0, S_y0]
    v_Star[0] = [S_vx0, S_vy0]

    for i in range(N-1):
        '''
        During each iteration, you should calculate the sum of the forces on the
        planet and star in question at the given time-step and apply it to your system
        through your favorite integration method – Euler Cromer, Runge Kutta 4,
        or Leapfrog will do the trick.  Each have their own advantages and
        disadvantages; refer to the Numerical Compendium to learn about these.
        '''
        # Using Leapfrog method
        r_norm_Planet = np.linalg.norm(x_Planet[i])
        r_norm_Star = np.linalg.norm(x_Star[i])

        a_Planet = -G * sun_mass * x_Planet[i] / (r_norm_Planet ** 3)
        a_Star = -G * planet_mass * x_Star[i] / (r_norm_Planet ** 3)

        vh_Planet = v_Planet[i] + a_Planet[i]*dt/2
        vh_Star = v_Star[i] + a_Star[i]*dt/2

        x_Planet[i+1] = x_Planet[i] + vh_Planet*dt
        x_Star[i+1] = x_Star[i] + vh_Star*dt

        r_norm_Planet = np.linalg.norm(x_Planet[i+1])
        r_norm_Star = np.linalg.norm(x_Star[i+1])

        a_Planet = -G * sun_mass * x_Planet[i+1] / (r_norm_Planet ** 3)
        a_Star = -G * planet_mass * x_Star[i+1] / (r_norm_Star ** 3)

        v_Planet[i+1] = vh_Planet + a_Planet*dt/2
        v_Star[i+1] = vh_Star + a_Star*dt/2
    return t, x_Planet, v_Planet, x_Star, v_Star

print('Defining integration function complete')
# Initializing system
username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)
star_mass = system.star_mass
planet_mass = system.masses[0] 

P_x0 = system.initial_positions[0,0]
P_y0 = system.initial_positions[1,0]
P_vx0 = system.initial_velocities[0,0] 
P_vy0 = system.initial_velocities[1,0]
com = 1/(star_mass + planet_mass) * (planet_mass*np.array([P_x0,P_y0]))  # Calculating the center of mass assuming star is in origin

# Shifting the planet and star position such that the center of mass is in origin
P_x0 -= com[0]
P_y0 -= com[1]
S_x0 = com[0]
S_y0 = com[1]
S_vx0 = 0
S_vy0 = 0
print('System initialized and initial values defined')
t, x_Planet, v_Planet, x_Star, v_Star  = integrate(T, dt, N, S_x0, S_y0, S_vx0, S_vy0, P_x0, P_y0, P_vx0, P_vy0, G, star_mass, planet_mass)
print('Integration complete')

plt.plot(x_Planet[:,0], x_Planet[:,1], label = 'Planet Orbit')
plt.plot(x_Star[:,0], x_Star[:,1], label = 'Star Orbit')
plt.scatter([0],[0], label = 'Center of mass')
plt.legend()
plt.show()