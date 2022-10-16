import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from Initialize_System import *
#CONSTANTS
G = 4*(np.pi**2)  #Gravitational Constant

def E(mu,v_vec,r_vec,M):
    '''
    Calculates the energy in the system \n
    mu is the reduced mass \n
    v = Planet_vel - Star_vel \n
    r = Planet_pos - Star_pos \n
    M = Planet_mass + Star_mass
    '''
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    return 1/2*mu*v**2 - (G*M*mu)/r

def P(r,v,mu):
    '''
    Calculates the angular momentum of the system \n
    r = Planet_pos - Star_pos \n
    v = Planet_vel - Star_vel \n
    mu is the reduced mass
    '''
    return np.cross(r,mu*v)

@njit
def integrate(T, dt, N, S_x0, S_y0, S_vx0, S_vy0, P_x0, P_y0, P_vx0, P_vy0, G, sun_mass, planet_mass):
    t = np.linspace(0.0, T, N)
    r_Star = np.zeros((N, 2))
    r_Planet = np.zeros((N, 2))
    v_Star = np.zeros((N, 2))
    v_Planet = np.zeros((N, 2))

    r_Planet[0] = [P_x0, P_y0]
    v_Planet[0] = [P_vx0, P_vy0]
    r_Star[0] = [S_x0, S_y0]
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
        r = r_Planet[i] - r_Star[i]
        r_norm = np.linalg.norm(r)
        # r_norm_Planet = np.linalg.norm(r_Planet[i])
        # r_norm_Star = np.linalg.norm(r_Star[i])

        a_Planet = -G * sun_mass * r / (r_norm ** 3)
        a_Star = G * planet_mass * r / (r_norm ** 3)

        vh_Planet = v_Planet[i] + a_Planet*dt/2
        vh_Star = v_Star[i] + a_Star*dt/2

        r_Planet[i+1] = r_Planet[i] + vh_Planet*dt
        r_Star[i+1] = r_Star[i] + vh_Star*dt

        r =   r_Planet[i+1] - r_Star[i+1]
        r_norm = np.linalg.norm(r)

        # r_norm_Planet = np.linalg.norm(r_Planet[i+1])
        # r_norm_Star = np.linalg.norm(r_Star[i+1])

        a_Planet = -G * sun_mass * r / (r_norm ** 3)
        a_Star = G * planet_mass * r / (r_norm ** 3)

        v_Planet[i+1] = vh_Planet + a_Planet*dt/2
        v_Star[i+1] = vh_Star + a_Star*dt/2

    return t, r_Planet, v_Planet, r_Star, v_Star

def main(): # Putting the execution of calculations in main function


    #CALCULATED DATA
    T = 2*20  #You want to find the planetary position from t=0 to t=T. How could you make an educated guess for T?
    # Look at our solar system which planet has the same distance from the sun.
    # Takes approx 12.81542 years


    #SIMULATION PARAMETERS
    N = 30000  #Number of time steps
    dt = T/N  #calculate time step from T and N

                        #Optional, but recommended for speed, check the numerical compendium
    # Functions
    

    # Initializing system
    username = "janniesc"
    seed = utils.get_seed(username)
    system = SolarSystem(seed)
    mission = SpaceMission(seed)
    star_mass = system.star_mass
    Planet = 2
    planet_mass = system.masses[Planet] 
    star_radius_au = system.star_radius * 6.684587*1e-9

    print(planet_mass)
    P_x0 = system.initial_positions[0,Planet]
    P_y0 = system.initial_positions[1,Planet]
    P_vx0 = system.initial_velocities[0,Planet] 
    P_vy0 = system.initial_velocities[1,Planet]
    S_vx0 = -(P_vx0*planet_mass) / star_mass
    S_vy0 = -(P_vy0*planet_mass) / star_mass

    com = 1/(star_mass + planet_mass) * (planet_mass*np.array([P_x0,P_y0]))  # Calculating the center of mass assuming star is in origin

    # Shifting the planet and star position such that the center of mass is in origin
    P_x0 -= com[0]
    P_y0 -= com[1]
    S_x0 = -com[0]
    S_y0 = -com[1]


    t, r_Planet, v_Planet, r_Star, v_Star  = integrate(T, dt, N, S_x0, S_y0, S_vx0, S_vy0, P_x0, P_y0, P_vx0, P_vy0, G, star_mass, planet_mass)

    r = (P_x0**2 + P_y0**2)**0.5
    t = np.linspace(0, 2*np.pi, 1000)
    x = r*np.cos(t)
    y = r*np.sin(t)
    plt.plot(x,y, label = 'Perfectly round planet orbit ')
    r = (S_x0**2 + S_y0**2)**0.5
    x = r*np.cos(t)
    y = r*np.sin(t)
    plt.plot(x,y, label = 'Perfectly round star orbit')
    plt.plot(r_Planet[:,0], r_Planet[:,1], label = 'Planet Orbit', linestyle = '--')
    plt.plot(r_Star[:,0], r_Star[:,1], label = 'Star Orbit', linestyle = '--')
    plt.scatter([0],[0], label = 'Center of mass')
    plt.scatter(r_Planet[0][0], r_Planet[0][1], label = 'Planet start position')
    plt.scatter(r_Planet[-1][0],r_Planet[-1][1], label = 'Planet end position' )
    plt.xlabel("x-position [AU]")
    plt.ylabel("y-position [AU]")
    plt.legend()
    plt.axis('equal')

    plt.show()

    M = planet_mass + star_mass
    mu = (planet_mass * star_mass)/M
    v = v_Planet - v_Star
    r = r_Planet - r_Star 
    resolution = 1000
    t = [i for i in range(int(resolution))]

    energy = [E(mu, v[i], r[i], M) for i in range(N) if i % (N/resolution) == 0]
    angular_momentum = [P(r[i], v[i],mu) for i in range(N) if i % (N/resolution) == 0]

            
    print(f'The greatest value is {max(energy)/min(energy): .10e} times greater than the smallest')
    print(f'The relative error of the energy is {(max(energy) - min(energy))/np.mean(energy)*100: .2e}%')
    # Plotting the angular momentum and energy 
    fig, ax1 = plt.subplots()


    P_plot = ax1.plot(t,angular_momentum, label="Angular Momentum", c = 'blue')
    ax1.tick_params(axis='y')
    ax1.set_ylabel('Angular Momentum')

    ax2 = ax1.twinx()
    E_plot = ax2.plot(t,energy, label = 'Energy', c = 'red')
    ax2.tick_params(axis='y')
    ax2.set_ylabel('Energy')

    plt.grid()
    plt.tight_layout()
    lns = P_plot + E_plot
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels)
    plt.show()

if __name__ == "__main__":
    main()