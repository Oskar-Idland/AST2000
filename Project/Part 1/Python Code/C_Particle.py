# In this file we will define and declare all functions and classes
import numpy as np
import random
from scipy.constants import Boltzmann as k

# FUNCTIONS -------------------


# CLASSES ---------------------

class Particle:
    particle_num = 0

    def __init__(self, m, T, L, seed, box, timesteps=1e-13):
        self.name = f"Particle_{Particle.particle_num}"
        Particle.particle_num += 1

        self.dt = timesteps
        self.inside_box = True
        self.v_sigma = np.sqrt(m / k * T)
        self.v_mu = 0
        random.seed(a=seed, version=2)
        self.box = box



        self.position = np.array([random.uniform(-self.box.length/2, self.box.length/2) for _ in range(3)])
        self.velocity = np.array([random.gauss(self.v_mu, self.v_sigma) for _ in range(3)])

    def advance(self):
        for dimension in range(3):
            if abs(self.position[-1][dimension]) >= self.box.length/2:
                self.wall_collision(dimension)
            else:
                np.append(self.velocity, self.velocity)
                np.append(self.position, self.position[-1] + self.velocity * self.dt)

    def wall_collision(self, dimension):
        side = np.sign(self.velocity[-1][dimension])  # Checking which side of the box we are on (positive or negative)

        np.append(self.velocity, self.velocity)  # Appending same velocity
        np.append(self.position, self.position[-1] + self.velocity * self.dt)  # Continuing movement in same direction

        if self.exiting_nozzle():
            self.inside_box = False
        else:
            self.velocity[-1][dimension] = -self.velocity[-1][dimension]  # Changing direction of velocity in correct dimension
            self.position[-1][dimension] = side * self.box.length - self.position[-2][dimension]  # Changing position to position after collision

    def exiting_nozzle(self):
        in_x_rad1 = (np.sqrt((self.position[-1][1])**2 + (self.position[-1][2])**2) < self.box.nozzle_rad)  # Checking if the particle is inside the radius of the nozzle before exiting
        in_x_rad2 = (np.sqrt((self.position[-2][1]) ** 2 + (self.position[-2][2]) ** 2) < self.box.nozzle_rad)  # Checking if the particle is inside the radius of the nozzle after exiting
        moving_to_nozzle = (self.velocity[-1][0] > 0)  # Checking if the particle is moving towards the nozzle in x
        if in_x_rad1 and in_x_rad2 and moving_to_nozzle:
            return True
        else:
            return False





