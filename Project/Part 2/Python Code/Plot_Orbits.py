import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit
import numpy as np

#CONSTANTS
G =                            #Gravitational Constant

#CALCULATED DATA
T =     #You want to find the planetary position from t=0 to t=T. How could you make an educated guess for T?


#SIMULATION PARAMETERS
N =             #Number of time steps
dt=             #calculate time step from T and N

@jit(nopython = True)                       #Optional, but recommended for speed, check the numerical compendium
def integrate(T, dt, N, x0, y0, vx0, vy0, G, sun_mass):

    '''
    Create the following arrays:

            Variable Name    Array Value         Array Shape
    [1]           t              Time                 N
    [2]           x            Position             N x 2
    [3]           v            Velocity             N x 2

    The Time array should consists of N points from "0" to "T", and can be
    created using "np.linspace(start, stop, step)":

    [1]           t = [0, dt, 2*dt, 3*dt, ..., T - dt, T]

    The Position array should consists of N sub-arrays; each sub-array should be
    of length 2, with index 0 representing a x-coordinate and index 1
    representing a y-coordinate.  Use the "np.zeros(shape)" function to create
    an empty array, and set its initial values to those defined above:

    [2]             x = [[x0, y0], [0, 0], ..., [0, 0]]

    The Velocity array should be of the same shape as the Position array, just
    use the inital velocity values instead of the initial position values:

    [3]            v = [[vx0, vy0], [0, 0], ..., [0, 0]]
    '''

    while t <= T + dt:
        '''
        During each iteration, you should calculate the sum of the forces on the
        planet in question at the given time-step and apply it to your system
        through your favorite integration method – Euler Cromer, Runge Kutta 4,
        or Leapfrog will do the trick.  Each have their own advantages and
        disadvantages; refer to the Numerical Compendium to learn about these.
        '''



    return t, x, v


def find_analytical_orbit(N_points,...,): ### find which input parameters you need for your planet to calculate the analytical orbit, these should be imported from ast2000tools

    angles = np.linspace(0,2.*np.pi,N_points)
    r_pos =       #use analytical expression
    x_analytic = r*      #convert to x-coordinates
    y_analytic = r*      #convert to y-coordinates

    return x_analytic, y_analytic




#RUNNING THE SIMULATION

##maybe include loop over planet here when it works for one planet?:

#ORBITAL DATA
x0 =                                #x-position at t = 0
y0 =                                #y-position at t = 0
vx0 =                               #x-velocity at t = 0
vy0 =                               #y-velocity at t = 0
sun_mass =                         #Mass of your sun
#note: the above values may be imported directly from ast2000tools.SolarSystem
x_analytic, y_analytic = find_analytical_orbit(N,...,) #find analytic orbit first to check your numerical calculation
t, x, v = integrate(T, dt, N, x0, y0, vx0, vy0, G, sun_mass)  #numerical orbit

#INTERPOLATION

'''
Interpolating our orbit allows us to calculate our planet's position at any
time, even those we didn't specifically integrate over in our loop.

SciPy allows you to accomplish this easily with the "interpolate" module, which
contains the "interp1d" function:
'''

position_function = interpolate.interp1d(t, x, axis = 0, bounds_error = False,
fill_value = 'extrapolate')

velocity_function = interpolate.interp1d(t, v, axis = 0, bounds_error = False,
fill_value = 'extrapolate')

'''
Assuming that the "x" and "v" arrays each contain pairs of elements,
"position_function" and "velocity_function" will both be functions that return
a two-element array of x and y coordinates/velocities.

As they are functions, they are used via being called.  For example, if you want
to find the planet's position after 0.1 years, you simply run:

                    pos = position_function(0.1)

This will return an array with an x and a y coordinate:

                            pos = [x, y]

If you search for times outside the range of this function (in other words,
searching for times after T) you may want to consider a solution involving
the modulo operator – you may find it handy!
'''

#PLOTTING THE ORBIT
plt.plot(analytic_x,analytic_y,color = "red", linewidth=3.0) #analytic orbit first to check your numerical calculation
plt.plot(x[:,0], x[:,1])
plt.show()
