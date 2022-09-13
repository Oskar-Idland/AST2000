import numpy as np
from C_Particle import Particle
from scipy.constants import Avogadro as A
from molmass import Formula
from C_Box import Box



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
T = 3*1E3
N = int(1e3)
timesteps = 1000
nozzle_pos = np.array([0,0,-L/2]) # Nozzle positioned directly under the box 
error_factor = 1/2
Box = Box(L, nozzle_pos)
# Initialize the array storing all particle objects

particles = np.array([Particle(m,T,seed*i,Box) for i in range(N)])
# Advancing all particles 1000 times 

[[list(map(lambda p: p.advance(), particles))] for _ in range(timesteps)]


print('Momentum: ' + str(error_factor*m*np.sum(np.array(list(map(lambda p: p.v_exit, particles)))))) # Multiplying by error factor to correct for the momentum being counted twice as much as it should