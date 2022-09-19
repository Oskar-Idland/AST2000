# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

lambda0 = 656.3  # Wavelenght for H_2 molecules, [nm]
light_speed = 2.99792458 * 10 ** 8  # The speed of light, [m/s]
solar_mass = 1.989 * 10 ** 30  # Mass of the sun/one solar mass, [kg]
G = 6.67408 * 10 ** (-11)  # Gravitational constant, [m^3/kg/s^2]


def star_values(filee):
    """
    Function for putting stored values of time, wavelenght and flux, for a star
    in arrays. This makes it easy to compute with the values.
    Takes the file with stored star values (filee) as argument.
    """
    infile = open(filee, "r")  # Making it possible to read from the file
    t = []  # prepare lists
    lam = []
    f = []
    for line in infile:  # Looping over the lines in the file
        t_in, lam_in, f_in = line.split()  # Splitting the line and putting the values in there one array
        t.append(t_in)  # add to list
        lam.append(lam_in)
        f.append(f_in)

    t = np.array(t, dtype='float64')  # convert list to numpy array
    lam = np.array(lam, dtype='float64')
    f = np.array(f, dtype='float64')

    infile.close()  # Closing the file
    return t, lam, f  # Returning the value arrays


def values(star):
    """
    Function for getting values from the stars given in the star argument, and
    computing the velocity of the star from the wavelenght.
    Takes a list of files (strings) with stored star values (star), the
    wavelenght for H_2 molecules (l0) and the speed of light (c) as arguments.
    """
    s_l = len(star)  # Finding number of data files
    time = np.zeros(s_l).tolist()  # An array to hold all the time values of the stars
    velocity = np.zeros(s_l).tolist()  # Same for velocity instead of time
    flux = np.zeros(s_l).tolist()  # Same for flux instead of time
    for i in range(s_l):  # Looping over all the stars
        t, l, f = star_values(star[i])  # Getting measured values of the star
        v =  # convert wavelength to velocity
        time[i], velocity[i], flux[i] = t, v, f  # Putting values in right array

    return time, velocity, flux


def peculiar_velocity(vel):
    """
    Function for computing the peculiar velocity.
    Takes measured velocity (vel) as arguments.
    """
    peculiar_vel = np.zeros(len(vel))  # An array for holding the pec. velocity of the stars
    for n in range(len(vel)):  # Looping over the number of stars
    ####calculate peculiar velocity here

    return peculiar_vel


def plot_v_f(tid, hastighet, peculiar, fluks):
    """
    Function for plotting the different velocities and flux of the stars.
    Takes time (tid), velocity (hastighet), peculiar_velocity  and flux
    (fluks) arrays for all the stars as arguments. That means the arrays are
    on the form [star number, measured value]
    """
    fig = plt.figure()  # Creating a figure
    fig.suptitle("Radial velocities of five given stars")
    num_stars = len(tid)  # Finding number of stars
    for i in range(num_stars):  # Looping over the different stars
        ax = fig.add_subplot(num_stars, 1, i + 1)  # Creating a subplot in the figure
        ax.plot()  # ****Insert whatever you want to plot****
    plt.xlabel("Time, [days]")
    plt.ylabel("Radial velocity, [m/s]")
    #    plt.savefig("velocities")
    plt.show()

    fig = plt.figure()
    fig.suptitle("Flux of five given stars")
    for i in range(num_stars):  # Same as over, but plotting flux instead
        ax = fig.add_subplot(num_stars, 1, i + 1)
        ax.plot()  # ****Insert whatever you want to plot****
    plt.xlabel("Time, [days]")
    plt.ylabel("Relative flux, [m/s]")
    #    plt.savefig("flux")
    plt.show()


# 4)
def vr_model(t, t0, P, vr):
    """
    A model function to fit the measured velocity (a normal sin/cos-function).
    Takes the time interval of the star that the function shall model (t) and
    test values for t0 (time the velocity is greatest), period (P) and radial
    velocity (vr) as arguments.
    """

    # calculate the model for all time indices

    return


def best_delta(t0, P, vr, vel, peculiar, times):
    """
    Method of least squares for finding the best t0, period and radial velocity
    for the vr_model to fit the measured values.
    Takes an interval of test values for t0 (time the velocity is greatest),
    period (P) and radial velocity (vr), and the measured velocity (vel),
    peculiar velocity (peculiar) and time interval of the star that the function shall
    model (times) as arguments.
    """

    # NB! The following code is the simplest, but not the most elegant way to solve this.
    # You may use meshgrid to make a faster and more elegant code,
    # consult the Numpy section of the Numerical Compendium if you wish to learn

    vr_data =  # Computing the star's radial velocity correcting for peculiar velocity

    len_t, len_p, len_vr = len(t0), len(P), len(vr)  # Finding the lenght of the intervals
    delta_values = np.zeros((len_t, len_p, len_vr))  # Defining a matrix to hold the differece the function computes
    # calculate all elements of  delta_values here

    i, j, k = np.where()  # Finding the indices for the smallest differece, insert test
    t_i = i[0];
    p_j = j[0];
    vr_k = k[0]  # Turning the list with indices to index numbers

    return t_i, p_j, vr_k, delta_values[t_i, p_j, vr_k]  # Returning the indices and minimal value


def find_initial(vel, peculiar, time, res=20):
    """
    Function for finding the interval for t0, period and radial velocity for a
    given star with a adjustable resolution.
    Takes measured velocity (vel), peculiar velocity (peculiar) and time interval of a
    star (time) as arguments.
    Takes the resolution (res) of the returned interval
    """

    # find intervals by eye and insert values here, or if you wish
    # write an algorithm for automatically finding the invervals

    return t0, period, vr


def plot_vr_model(tid, hastighet, peculiar, best_t, best_v, best_p):
    """
    Function for plotting the radial velocity data against the model for radial
    velocity for a star over time.
    Takes the time interval (tid), measured velocity (hastighet) and peculiar
    velocity for the star and the best found t0 (best_t), radial
    velocity (best_v) and period (best_p) for the vr_model as arguments.
    """

    # up to you how you want to plot it


def mass_planet(m_star, v_star, p, g):
    """
    Function for computing the lower limit mass of an orbiting planet around the
    star.
    Takes mass of the star (m_star), best fitting radial velocity (v_star) and
    best fitting period (p) for the star, and the gravitational constant as arguments.
    """
    # calculate mass

    return


if __name__ == "__main__":
    star = []  # insert your star file names
    time, velocity, flux = values(star)  # read from file
    peculiar = peculiar_velocity(velocity)  # find peculiar velocity
    plot_v_f(time, velocity, peculiar, flux)  # plot velocities and fluxes

    star_number = 0  # choose star to analyze
    Velocity, Peculiar, TIME = velocity[star_number], peculiar[star_number], time[
        star_number]  # find data for this star
    t0, Period, v_radial = find_initial(Velocity, Peculiar, TIME)  # find intervals for free parameters
    index_t, index_P, index_vr, minimi = best_delta(t0, Period, v_radial, Velocity, Peculiar,
                                                    TIME)  # find best fit parameter indices

    T0, VR, Per = t0[index_t], v_radial[index_vr], Period[index_P]  # find best fit parameter values
    plot_vr_model(TIME, Velocity, Peculiar, T0, VR, Per)  # test your model

    print("Best t0=%0.1f days, vr=%0.1f m/s, P=%0.1f days for star %d gave differece %0.1f" % (
    T0, VR, Per, star_number, minimi))
    mp = mass_planet(mass_star * solar_mass, VR, Per, G)  # insert the mass of your star
    print("The mass of a planet orbiting star %d was estimated to be %g kg" % (star_number, mp))

