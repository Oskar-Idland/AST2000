# In this file we will define and declare all functions and classes
import numpy as np
import random
from scipy.constants import Boltzmann as k
# FUNCTIONS -------------------
from C_Box import Box

# CLASSES ---------------------

class Particle:
    particle_num = 0

    def __init__(self, m, T, seed, box):
        self.name = f"Particle_{Particle.particle_num}"
        Particle.particle_num += 1

        self.inside_box = True
        self.v_sigma = np.sqrt(m / k * T)
        self.v_mu = 0
        random.seed(a=seed, version=2)
        self.box = box

        self.position = np.array([random.uniform(-self.box.length/2, self.box.length/2) for _ in range(3)])
        self.velocity = np.array([random.gauss(self.v_mu, self.v_sigma) for _ in range(3)])

    def advance(self, timestep=1e-13):
        np.append(self.velocity, self.velocity[-1])  # Appending same velocity
        np.append(self.position, self.position[-1] + self.velocity[-1] * timestep)  # Continuing movement in same direction

        for dimension in range(3):  # Checking for each dimension whether the position is outside the length of the box
            if abs(self.position[-2][dimension]) >= self.box.length/2:
                self.wall_collision(dimension)

    def wall_collision(self, dimension):
        side = np.sign(self.position[-1][dimension])  # Checking which side of the box we are on (positive or negative)

        if self.exiting_nozzle(side, dimension):  # Checking if the particle is going through the nozzle
            self.inside_box = False
            # HAVE TO IMPLEMENT NEW PARTICLE BEING GENERATED HERE!!! ------------------------------
        else:
            self.velocity[-1][dimension] = -self.velocity[-1][dimension]  # Changing direction of velocity in correct dimension
            self.position[-1][dimension] = side * self.box.length - self.position[-2][dimension]  # Changing position to position after collision

    def exiting_nozzle(self, side, dimension):
        is_nozzle_wall = ((side == self.box.nozzle_side) and (dimension == self.box.nozzle_axis))
        in_nozzle_rad2 = (np.sqrt(np.linalg.norm(self.position[-2])**2 - (self.position[-2][dimension])**2) < self.box.nozzle_rad)  # Checking if the particle is inside the radius of the nozzle before exiting
        in_nozzle_rad1 = (np.sqrt(np.linalg.norm(self.position[-1])**2 - (self.position[-1][dimension])**2) < self.box.nozzle_rad)  # Checking if the particle is inside the radius of the nozzle after exiting  # UNNECESSARY???????
        moving_to_nozzle = (self.velocity[-1][dimension] > 0)  # Checking if the particle is moving towards the nozzle in x
        if moving_to_nozzle and is_nozzle_wall and in_nozzle_rad1 and in_nozzle_rad2:  # Checking if the particle is moving towards, and hitting the correct wall as well as being inside the nozzle diameter before and after exiting the nozzle
            return True
        else:
            return False

