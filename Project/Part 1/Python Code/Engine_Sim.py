import numpy as np
from C_Particle import Particle
from scipy.constants import Avogadro as A
from molmass import Formula
from C_Box import Box
import time
import ast2000tools.constants as constants
start = time.time()

"""
Simplifications and Assumptions:
* Fuel tanks are equipped with pure H2 gas
* Temperature and number of particles are kept constant
* Gravitational effects are to be neglected
* Particles have no spacial extension - No particle-particle collisions!
* When particles collide with a wall, the collision is perfectly elastic
"""


seed = 8
m_h2 = constants.m_H2  # Mass of one H2 molecule In Kg
L = 1E-6 
T = 3*1E3
N = int(1e3)
timesteps = 1000
nozzle_pos = np.array([0, 0, -L/2])  # Nozzle positioned directly under the box
error_factor = 1/2
Box = Box(L, nozzle_pos)
# Initialize the array storing all particle objects

def adv_p(p):
    return p.advance()

adv_p = np.vectorize(adv_p)

# Creating an array with particles
a = time.time()
particles = np.array([Particle(m_h2, T, seed*i, Box) for i in range(N)])
b = time.time()
print(f'Particle creation took {b - a} s')
# Advancing all particles 1000 times 
a = time.time()
[[list(map(adv_p, particles))] for _ in range(timesteps)]
b = time.time()
print(f'Advancing particles took {b - a} s')

exited_particles = 0
for particle in particles:
    exited_particles += particle.num_exited  # Adding up the momentum of all exited particles
m_dot = exited_particles * constants.m_H2/1e-9  # Mass flow rate
print(f"Mass flow rate: {m_dot}")


a = time.time()
momentum = 10*error_factor*m_h2*np.sum(np.array(list(map(lambda p: p.v_exit, particles))))
print('Momentum: ' + str(momentum)) # Multiplying by error factor to correct for the momentum being counted twice as much as it should
b = time.time()
print(f'Momentum calculation took {b - a} s')
thrust = momentum/1e-9
print(f"Thrust: {thrust}")
num_of_boxes = 50_000/thrust
print(f"Number of boxes: {num_of_boxes:e}")
print(f"Total Mass Flow rate: {m_dot * num_of_boxes}")
end = time.time()
print(f'The program used {end - start} s')