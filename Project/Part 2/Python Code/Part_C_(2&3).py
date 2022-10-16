from Part_C import integrate
from Initialize_System import *
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np


#CONSTANTS
G = 4*(np.pi**2)  #Gravitational Constant

#CALCULATED DATA
T = 5*20  #You want to find the planetary position from t=0 to t=T. How could you make an educated guess for T?
# Look at our solar system which planet has the same distance from the sun.
# Takes approx 12.81542 years


#SIMULATION PARAMETERS
N = 30000  #Number of time steps
dt = T/N  #calculate time step from T and N


Planet = 2
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

def radial_velocity():
    t, r_Planet, v_Planet, r_Star, v_Star  = integrate(T, dt, N, S_x0, S_y0, S_vx0, S_vy0, P_x0, P_y0, P_vx0, P_vy0, G, star_mass, planet_mass)

    angle = np.pi/4 # Arbitrary angle
    v_pec = .0000035 # Arbitrary velocity
    v_Star = np.array([np.linalg.norm(v) for v in v_Star]) #Changing the shape of the array from (3000,2) -> (3000,)
    v_rad = v_Star * np.sin(angle) - v_pec 
    m = np.max(v_rad)
    noise = np.random.normal(0, 1/5*m, N) #Gaussian probability distribution with mean = 0 stddev = 1/5 of the max v_rad
    plt.plot(t, v_rad + noise, c = 'black')
    plt.title('Radial Velocity')
    plt.show() 
    # plt.savefig(os.path.join("Figures\Radial Velocity"), format = 'pdf')
    # with open("Radial Velocity", 'w') as file: 
    #     file.write('time [Yr]               Velocity [AU]\n')
    #     for i in range(len(t)):
    #         file.write(f'{t[i]: <22} {v_rad[i]+noise[i]: >22}\n') 
    # TODO Undo commenting before delivery
    
def read_radial_velocity(filename):
    '''
    Reads radial velocity data from file \n
    Make sure not to have an empty line at the end \n
    Make sure the first line isn't data \n
    Make sure the values are split by spaces \n
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        N = len(lines)-1
        t = np.zeros(N)
        v_rad = np.zeros(N)
        for i in range(1,N):
            velocity, time = lines[i].split()
            v_rad[i], t[i] = float(velocity), float(time)
    return t, v_rad

# t, v_rad = read_radial_velocity('Radial Velocity')
# plt.plot(t,v_rad)
# plt.show()

# TODO: Remove comment before delivery 
# radial_velocity() 




def radial_velocity_analysis(t,v_rad):
    k = 400

    plt.plot(t[k:-k], v_rad[k:-k], label = 'Velocity with noise')


    t = t[int(k/2):int(-k/2):k]
    v_filtered = np.zeros_like(t)
    for i in range(int(len(v_rad) / k - 1)):
        v_filtered[i] = sum(v_rad[i*k:(i+1)*k + 1])


    v_filtered *= 1/(k)
    plt.plot(t, v_filtered, label = 'Cleansed data')
    plt.legend()
    # plt.savefig("Figures\\Other_Group_Radial_Velocity")
    plt.show()
    
t, v_rad = read_radial_velocity('velocitydata.txt')
radial_velocity_analysis(t,v_rad)
v_rad, t = read_radial_velocity('velocitydata3.txt')
radial_velocity_analysis(t,v_rad)


# TODO Ta bort før innlevering



