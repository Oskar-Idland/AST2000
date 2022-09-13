import numpy as np
import matplotlib.pyplot as plt
import ast2000tools as ast
import random as rand
from C_Particle import *
from scipy.constants import Avogadro as A
from molmass import Formula

"""
Simplifications and Assumptions:
* Fuel tanks are equipped with pure H2 gas
* Temperature and number of particles are kept constant
* Gravitational effects are to be neglected
* Particles have no spacial extension - No particle-particle collisions!
* When particles collide with a wall, the collision is perfectly elastic
"""
seed = 8
m = Formula('H2').mass/(A*1000) # Mass of one H2 molecule In Kg
L = 1E-6 
T = 3*1E3
N = 100 
nozzle_pos = np.array([0,0,-L/2]) # Nozzle positioned directly under the box 
Box = Box(L, nozzle_pos)
particles = np.array([Particle(m,T,seed*_,Box) for _ in range(N)])
# Check for particle positions and velocities making sense. 
# print(list(map(lambda p: (p.name, p.position,np.linalg.norm(p.velocity)),particles)))

[map(lambda p: p.advance(), particles) for _ in range(1000)]
print(particles[0].p_exit, particles[1].p_exit)
print(list(map(lambda p: p.p_exit, particles)))
box_cordx = [-L/2, L/2, L/2, -L/2, -L/2]
box_cordy = [-L/2, -L/2, L/2, L/2, -L/2]
plt.plot(box_cordx,box_cordy)
[particles[0].advance() for _ in range(100)]
print(particles[0].position)
px = particles[0].position[0::3]
py = particles[0].position[1::3]
# plt.plot(([-Box.nozzle_rad, Box.nozzle_pos[-1]], [Box.nozzle_rad, Box.nozzle_pos[-1]]), color = 'red')
plt.plot(px,py)
plt.show()
