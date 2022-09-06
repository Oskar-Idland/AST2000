import numpy as np
import matplotlib.pyplot as plt
import ast2000tools as ast
import random
from C_Particle import *

"""
Simplifications and Assumptions:
* Fuel tanks are equipped with pure H2 gas
* Temperature and number of particles are kept constant
* Gravitational effects are to be neglected
* Particles have no spacial extension - No particle-particle collisions!
* When particles collide with a wall, the collision is perfectly elastic
"""

L = 1E-7 
T = 3*1E3
N = 100 
