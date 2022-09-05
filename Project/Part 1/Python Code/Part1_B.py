import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random


# We choose the chamber to be a cube of length L
L = 10**(-6)  # Length of chamber in meters
T = 3*10**3  # Temperature in Kelvin
N = 100  # Number of particles
seed = 1024
random.seed(a=seed, version=2)  # Sets a seed for the random number generator

particle_vel = np.zeros([[] for i in range(0, N)])
particle_pos = np.zeros([N, 3])


