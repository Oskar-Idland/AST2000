# Function which computes the normal probability of a particle to have a value in an interval a to b
import numpy as np
from scipy.integrate import quad

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
    Function which computes the normal probability of a particle to have a value in an interval a to b
    :param a: Lower limit of interval
    :param b: Upper limit of interval
    :param mu: Median value
    :param sigma: Standard deviation
    :return: Probability of a value to be in the interval between a and b
    """
    def f(x, mu, sigma):
        """
        Normal probability function to integrate
        """
        return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-1/2 * ((x-mu)/sigma)**2)

    return quad(f, a, b, args=(mu, sigma))[0]  # This is the Integration
    # quad() returns a tuple with (Calculated integral value, Error margin). We return only the first value.
