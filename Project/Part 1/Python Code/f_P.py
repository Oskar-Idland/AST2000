# Function which computes the normal probability of a particle to have a value in an interval a to b
import numpy as np

def P(a, b, dt = .0001):
    '''
    Function which computes the normal probability of a particle to have a value in an interval a to b, with stepsize = 0.0001 as default
    '''
    sigma; #Sigma og mu må defineres, men ellers er dette hvordan jeg tenker ting kan gjøres
    mu; 
    def f(mu, sigma,x):
        return 1/(2*np.pi**(1/2)*sigma) * np.exp(-1/2 * ((x-mu)/sigma)**2)

    x = np.linspace(a,b,1/dt + 1)
    return sum( f(mu, sigma, x)*dt )

P
