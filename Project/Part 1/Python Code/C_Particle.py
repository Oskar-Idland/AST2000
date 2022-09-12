# In this file we will define and declare all functions and classes
import numpy as np
import random
from scipy.constants import Boltzmann as k

# FUNCTIONS -------------------


# CLASSES ---------------------

class Particle:
    particle_num = 0

    def __init__(self, m, T, seed, box):
        self.name = f"Particle_{Particle.particle_num}"
        Particle.particle_num += 1

        self.m = m
        self.inside_box = True  # UNNECESSARY?
        self.v_sigma = np.sqrt(m / k * T)
        self.v_mu = 0
        self.seed = seed
        self.box = box
        self.rand_pos_vel()
        self.p_exit = np.array([])  # Array to store momentum when exiting the nozzle

    def rand_pos_vel(self):
        random.seed(a=self.seed, version=2)
        self.position = np.array([random.uniform(-self.box.length / 2, self.box.length / 2) for _ in range(3)])
        self.velocity = np.array([random.gauss(self.v_mu, self.v_sigma) for _ in range(3)])

    def advance(self, timestep=1e-13):
        np.append(self.velocity, self.velocity[-1])  # Appending same velocity
        np.append(self.position, self.position[-1] + self.velocity[-1] * timestep)  # Continuing movement in same direction

        collision_axis = np.nonzero(abs(self.position[-2]) >= self.box.length/2)  # Checking for each dimension whether the position is outside the length of the box
        if collision_axis[0].size > 0:
            self.wall_collision(collision_axis[0][0])  # Initiating a collision with a wall

    def wall_collision(self, axis):
        side = np.sign(self.position[-1][axis])  # Checking which side of the box we are on (positive or negative)

        if self.exiting_nozzle(axis):  # Checking if the particle is going through the nozzle
            self.inside_box = False  # UNNECESSARY?
            np.append(self.p_exit, self.m*self.velocity[-1])  # Storing the momentum of the particle when exiting
            self.rand_pos_vel()  # Creating new random position and velocity inside the box
            self.inside_box = True  # UNNECESSARY?
        else:
            self.velocity[-1][axis] = -self.velocity[-1][axis]  # Changing direction of velocity in correct dimension
            self.position[-1][axis] = side * self.box.length - self.position[-2][axis]  # Changing position to position after collision

    def exiting_nozzle(self, axis):
        dist_from_nozzle = self.position[-2] - self.box.nozzle_pos
        in_nozzle_rad = (np.sqrt(np.linalg.norm(self.position[-2])**2 - (self.position[-2][axis])**2) < self.box.nozzle_rad)  # Checking if the particle is inside the radius of the nozzle before exiting
        moving_to_nozzle = (np.sign(self.velocity[-1][axis]) == np.sign(self.box.nozzle_pos[axis]))  # Checking if the particle is moving towards the nozzle in x
        if moving_to_nozzle and in_nozzle_rad:  # Checking if the particle is moving towards, and hitting the correct wall as well as being inside the nozzle diameter before and after exiting the nozzle
            return True
        else:
            return False






