import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit
import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

username = "janniesc"
seed = utils.get_seed(username)
system = SolarSystem(seed)
mission = SpaceMission(seed)

#CONSTANTS
G = 4*(np.pi**2)  #Gravitational Constant

#CALCULATED DATA
T = 13*20  #You want to find the planetary position from t=0 to t=T. How could you make an educated guess for T?
# Look at our solar system which planet has the same distance from the sun.
# Takes approx 12.81542 years


#SIMULATION PARAMETERS
N = 400_000  #Number of time steps
dt = T/N  #calculate time step from T and N

@jit(nopython = True)                       #Optional, but recommended for speed, check the numerical compendium
def integrate(T, dt, N, x0, y0, vx0, vy0, G, sun_mass):
    t = np.linspace(0.0, T, N)
    x = np.zeros((N, 2))
    v = np.zeros((N, 2))

    x[0] = [x0, y0]
    v[0] = [vx0, vy0]

    for i in range(N-1):
        '''
        During each iteration, you should calculate the sum of the forces on the
        planet in question at the given time-step and apply it to your system
        through your favorite integration method – Euler Cromer, Runge Kutta 4,
        or Leapfrog will do the trick.  Each have their own advantages and
        disadvantages; refer to the Numerical Compendium to learn about these.
        '''
        # Using Leapfrog method
        r_norm = np.linalg.norm(x[i])
        a = -G * sun_mass * x[i] / (r_norm ** 3)
        vh = v[i] + a*dt/2
        x[i+1] = x[i] + vh*dt
        r_norm = np.linalg.norm(x[i+1])
        a = -G * sun_mass * x[i+1] / (r_norm ** 3)
        v[i+1] = vh + a*dt/2
    return t, x, v


def find_analytical_orbit(N_points, system, planet_index): ### find which input parameters you need for your planet to calculate the analytical orbit, these should be imported from ast2000tools
    a = system.semi_major_axes[planet_index]
    e = system.eccentricities[planet_index]
    init_angle = system.aphelion_angles[planet_index]
    angles = np.linspace(0,2.*np.pi,N_points)
    r = (a*(1-e**2))/(1+(e*np.cos(angles-init_angle)))      #use analytical expression
    x_analytic = a*e + r*np.cos(angles-init_angle)          #convert to x-coordinates
    y_analytic = r*np.sin(angles-init_angle)                #convert to y-coordinates
    return x_analytic, y_analytic




# RUNNING THE SIMULATION
num_planet_orbits = []
an_planet_orbits = []
position_function = []
velocity_function = []

for planet_idx in range(8):
    # ORBITAL DATA
    x0 = system.initial_positions[0, planet_idx]        # x-position at t = 0
    y0 = system.initial_positions[1, planet_idx]        # y-position at t = 0
    vx0 = system.initial_velocities[0, planet_idx]      # x-velocity at t = 0
    vy0 = system.initial_velocities[1, planet_idx]      # y-velocity at t = 0
    sun_mass = system.star_mass                         # Mass of your sun
    # note: the above values may be imported directly from ast2000tools.SolarSystem
    t, x, v = list(integrate(T, dt, N, x0, y0, vx0, vy0, G, sun_mass))  # numerical orbit
    num_planet_orbits.append([t, x, v])
    x_analytic, y_analytic = find_analytical_orbit(N, system, planet_idx)  # Find analytic orbit first to check your numerical calculation
    an_planet_orbits.append([x_analytic, y_analytic])


    #INTERPOLATION
    position_function = interpolate.interp1d(t, x, axis=0, bounds_error=False,
    fill_value='extrapolate')

    velocity_function = interpolate.interp1d(t, v, axis=0, bounds_error=False,
    fill_value='extrapolate')

    '''
    Assuming that the "x" and "v" arrays each contain pairs of elements,
    "position_function" and "velocity_function" will both be functions that return
    a two-element array of x and y coordinates/velocities.
    
    As they are functions, they are used via being called.  For example, if you want
    to find the planet's position after 0.1 years, you simply run:
    
                        pos = position_function(0.1)
    
    This will return an array with an x and a y coordinate:
    
                                pos = [x, y]
    
    If you search for times outside the range of this function (in other words,
    searching for times after T) you may want to consider a solution involving
    the modulo operator – you may find it handy!
    '''
    x = np.array(x)
#PLOTTING THE ORBIT
    plt.plot(x[:, 0], x[:, 1], "--", linewidth=1.5) #analytic orbit first to check your numerical calculation
    plt.plot(x_analytic, y_analytic, linewidth=1.5)
plt.axis("equal")
plt.xlabel("x-position [AU]")
plt.ylabel("y-position [AU]")
plt.savefig("../Figures/Orbit_plots.png")
plt.show()


print(an_planet_orbits[0])
