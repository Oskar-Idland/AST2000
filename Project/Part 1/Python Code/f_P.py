# Function which computes the normal probability of a particle to have a value in an interval a to b
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def P(a, b, dt = .0001):
    '''
    Function which computes the normal probability of a particle to have a value in an interval a to b, with stepsize = 0.0001 as default
    '''
    sigma; #Sigma og mu må defineres, men ellers er dette hvordan jeg tenker ting kan gjøres
    mu; 
    def f(mu, sigma,x):
        return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-1/2 * ((x-mu)/sigma)**2)

    x = np.linspace(a,b,1/dt + 1)
    return sum( f(mu, sigma, x)*dt )


def P2(a, b, mu, sigma):
    """
    Function which computes the normal gaussian probability of a particle to have a value in an interval a to b
    using the function gauss_func
    :param a: Lower limit of interval
    :param b: Upper limit of interval
    :param mu: Mean value
    :param sigma: Standard deviation
    :return: Probability of a value to be in the interval between a and b
    """

    def gauss_dist(x, mu, sigma):
        """
        Function for Normal Gaussian distribution
        :param x: Value to calculate probability for
        :param mu: Median value
        :param sigma: Standard deviation
        :return: Probability between 0 and 1
        """
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

    return quad(gauss_dist, a, b, args=(mu, sigma))[0]  # This is the Integration
    # quad() returns a tuple with (Calculated integral value, Error margin). We return only the first value.


def P3(a, b, mu, sigma):
    """
    Function which computes the normal gaussian probability of a particle to have a value in an interval a to b
    using the function gauss_func
    :param a: Lower limit of interval
    :param b: Upper limit of interval
    :param mu: Mean value
    :param sigma: Standard deviation
    :return: Probability of a value to be in the interval between a and b
    """
    # This is the Integration - scipy.stats.norm.pdf returns probability for given x with loc=mu and scale=sigma
    return quad(norm.pdf, a, b, args=(mu, sigma))[0]
    # quad() returns a tuple with (Calculated integral value, Error margin). We return only the first value.

