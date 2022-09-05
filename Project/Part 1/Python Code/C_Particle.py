# In this file we will define and declare all functions and classes
import numpy as np

# FUNCTIONS -------------------


# CLASSES ---------------------

class Particle:
    particle_num = 0
    def __init__(self, timesteps = 0.001):
        self.name = f"Particle_{Particle.particle_num}"
        Particle.particle_num += 1
        self.timesteps = timesteps
        self.positions = np.zeros()
        self.velocities;
        self.inside_box = True 
