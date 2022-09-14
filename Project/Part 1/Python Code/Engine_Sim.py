import numpy as np
from C_Particle import Particle
from scipy.constants import Avogadro as A
from molmass import Formula
from C_Box import Box
import time
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
m = Formula('H2').mass/(A*500) # Mass of one H2 molecule In Kg
L = 1E-6 
T = 3*1E4
N = int(1e3)
timesteps = 1000
nozzle_pos = np.array([0,0,-L/2]) # Nozzle positioned directly under the box 
error_factor = 1/2
Box = Box(L, nozzle_pos)
# Initialize the array storing all particle objects

a = time.time()
particles = np.array([Particle(m,T,seed*i,Box) for i in range(N)])
b = time.time()
print(f'Particle creation took {b - a} s')
# Advancing all particles 1000 times 
a = time.time()
[[list(map(lambda p: p.advance(), particles))] for _ in range(timesteps)]
b = time.time()
print(f'Advancing particles took {b - a} s')

a = time.time()
print('Momentum: ' + str(10*error_factor*m*np.sum(np.array(list(map(lambda p: p.v_exit, particles)))))) # Multiplying by error factor to correct for the momentum being counted twice as much as it should
b = time.time()
print(f'Momentum calculation took {b - a} s')
end = time.time()
print(f'The program used {end - start} s')