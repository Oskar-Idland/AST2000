import numpy as np
from scipy.stats import maxwell
from scipy.constants import Boltzmann as k

def max_boltz_dist3D(v, T, m):
    """
    Function to calculate the probability of a particle having a certain velocity
    :param v: Velocity
    :param T: Temperature in kelvin
    :param m: mass of particle
    :return: Probability between 0 and 1
    """
    return (m/(2*np.pi*k*T))**(3/2)*np.exp(-(m*v**2)/(2*k*T))*4*np.pi*v**2


def max_boltz_dist1D(v, T, m):
    """
    Function to calculate the probability of a particle having a certain velocity
    :param v: Velocity
    :param T: Temperature in kelvin
    :param m: mass of particle
    :return: Probability between 0 and 1
    """
    return np.sqrt(m/(2*np.pi*k*T))*np.exp(-(m*v**2)/(2*k*T))

