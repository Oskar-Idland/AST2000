# In this file we will define and declare all functions and classes
import numpy as np
import random
from scipy.constants import Boltzmann as k

# FUNCTIONS -------------------


# CLASSES ---------------------

class Particle:
    particle_num = 0

    def __init__(self, m, T, L, seed, timesteps=1e-13):
        self.name = f"Particle_{Particle.particle_num}"
        Particle.particle_num += 1

        self.dt = timesteps
        self.inside_box = True
        self.v_sigma = np.sqrt(m / k * T)
        self.v_mu = 0
        random.seed(a=seed, version=2)
        self.L = L

        self.position = np.array([random.uniform(-L/2, L/2) for _ in range(3)])
        self.velocity = np.array([random.gauss(self.v_mu, self.v_sigma) for _ in range(3)])

    def advance(self):
        if self.position[-1][0] <= -self.L/2 or self.L/2 <= self.position[-1][0]:
            self.wall_collision("x")
        elif self.position[-1][1] <= -self.L/2 or self.L/2 <= self.position[-1][1]:
            self.wall_collision("y")
        elif self.position[-1][2] <= -self.L/2 or self.L/2 <= self.position[-1][2]:
            self.wall_collision("z")
        else:
            np.append(self.position, self.position[-1] + self.velocity * self.dt)

    def wall_collision(self, direction):
        if direction == "x":
            np.append(self.velocity, [-self.velocity[0], self.velocity[1], self.velocity[2]])
            np.append(self.position, [self.L - self.position[-1], self.velocity * self.dt, self.velocity * self.dt])
        elif direction == "y":
            np.append(self.velocity, [self.velocity[0], -self.velocity[1], self.velocity[2]])
        elif direction == "z":
            np.append(self.velocity, [self.velocity[0], self.velocity[1], -self.velocity[2]])


