import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit, njit
import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import time
start = time.time()

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
N = 3_000_000  #Number of time steps
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


def A_kepler(planet_orbits, start_idx, stop_idx, planet_idx):
    A = 0
    S = 0
    orbits_arr = np.array(planet_orbits)
    for i in range(start_idx, stop_idx):
        r = np.linalg.norm((orbits_arr[planet_idx, :, i]+orbits_arr[planet_idx, :, i+1])/2)
        h = np.linalg.norm(orbits_arr[planet_idx, :, i+1]-orbits_arr[planet_idx, :, i])
        d_theta = np.arcsin(h/r)
        A += 0.5 * r**2 *d_theta
        S += h
    return A, S



# RUNNING THE SIMULATION
num_planet_orbits = []
an_planet_orbits = []
position_function = []
velocity_function = []
verification_positions = np.zeros([2, 8, N])

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

    verification_positions[0, planet_idx] = x[:, 0]
    verification_positions[1, planet_idx] = x[:, 1]


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


    P_Newton = np.sqrt((4*(np.pi**2)*(system.semi_major_axes[planet_idx]**3))/(G*(system.star_mass + system.masses[planet_idx])))
    P_Kepler = np.sqrt(system.semi_major_axes[planet_idx]**3)
    print(f"Planet {planet_idx} -------------------")
    print(f"Newton: {P_Newton:.8f} Yrs, Kepler: {P_Kepler:.8f} Yrs")
    print(f"Difference: {abs(P_Kepler-P_Newton):.8f} Yrs")
    # print(f"Starting Position: {position_function(0)}")
    x = np.array(x)

    """
    # Uncomment to get Numeric period. Pretty much the same as Newtons. 
    # Increases computing time from ~30s to ~225s
    
    # Determining Period Numerically
    start_pos = position_function(0)
    for i in range(int(1/dt), N):
        a = position_function(i*dt)
        if np.linalg.norm(a-start_pos) < 0.01:
            print(f"Numeric Period: {i*dt:.4f} Yrs\n\n")
            break
    """
    # Finding period analytically
    e = system.eccentricities[planet_idx]
    a = system.semi_major_axes[planet_idx]
    b = a*np.sqrt((1-e**2))
    h = np.cross(x[0], v[0])
    period = 2*np.pi*a*b/h
    print(f"Analytic Period: {period:.8f} Yrs")
    diff1 = abs(period-P_Kepler)
    diff2 = abs(period-P_Newton)
    print(f"Difference Kepler: {100/period*diff1} %")
    print(f"Difference Newton: {100 / period * diff2} %\n\n")



# PLOTTING THE ORBIT
    plt.plot(x[:, 0], x[:, 1], "--", linewidth=1.5)  # Numeric orbit
    plt.plot(x_analytic, y_analytic, linewidth=1.5)  # Analytic Orbit
plt.scatter(0, 0, c="k")
plt.axis("equal")
plt.xlabel("x-position [AU]")
plt.ylabel("y-position [AU]")
plt.savefig("Part 2/Figures/Orbit_plots.png")
plt.show()

A1, S1 = A_kepler(an_planet_orbits, 0, 1, 0)
A2, S2 = A_kepler(an_planet_orbits, 100_000, 100_001, 0)
V1 = S1 / (1 * dt)
V2 = S2 / (1 * dt)
print(f"Planet {0}")
print(f"Area Aphelion: {A1:.8f} AU^2, Distance Aphelion: {S1:E} AU, Speed Aphelion: {V1:.8f} AU/Year")
print(f"Area Perihelion: {A2:.8f} AU^2, Distance Perihelion: {S2:E} AU, Speed Perihelion: {V2:.8f} AU/Year")
print(f"Diff Area: {(A2 - A1):E} AU^2, Diff Distance: {(S2 - S1):E} AU, Diff Speed: {(V2 - V1):.8f} AU/Year\n")
end = time.time()
print(f"The simulation took {end-start:.4f} seconds")


# Verifying the Orbits
mission.verify_planet_positions(T, verification_positions)
mission.generate_orbit_video(np.linspace(0, T, N), verification_positions)
