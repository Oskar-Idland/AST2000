import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts

# Initializing mission, system and shortcuts
username = "janniesc"
seed = 36874
code_launch_results = 83949
code_escape_trajectory = 74482
stable_orbit = 88515
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [stable_orbit])
shortcut1 = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])
planet_idx = 1  # Planet index
G = 6.6743015e-11  # Universal gravitational constant

# Defining some important constants
m_planet = system.masses[planet_idx] * 1.98847e30  # Planet mass in kg
g = G*m_planet/(system.radii[1]*1000)  # Gravitational acceleration on the planet
rho_atm = system.atmospheric_densities[planet_idx]  # Average atmospheric density
C_D = 1  # Drag coefficient
omega_atm = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)  # Angular velocity of the atmosphere


def cart_to_spherical(x, y, z):
    """
    Conversion from cartesian coordinates to spherical coordinates
    :param x: x-coordinate
    :param y: y-coordinate
    :param z: z-coordinate
    :return: Spherical coordinates r, theta, phi (phi measured from z-axis)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z/r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))
    return r, theta, phi


def spherical_to_cart(r, theta, phi):
    """
    Conversion from spherical coordinates to cartesian coordinates
    :param r: radius from center
    :param theta: theta-angle
    :param phi: phi-angle
    :return: Cartesian coordinates x, y, z
    """
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def find_w(current_pos):
    """
    Finds velocity of atmosphere based on the position relative to the planet
    :param current_pos: Current position relative to the planet in cartesian coordinates
    :return: velocity of the atmosphere at the given position in cartesian coordinates
    """
    r, theta, phi = cart_to_spherical(current_pos[0], current_pos[1], current_pos[2])  # Converting to spherical coordinates
    omega = (2*np.pi)/(system.rotational_periods[1]*24*3600)  # Angular velocity of planet
    abs_w = omega*r  # Calculating absolute velocity of atmosphere
    dir_w = np.array([-np.sin(theta), np.cos(theta), 0])  # Calculating direction of velocity of atmosphere
    return abs_w*dir_w


def drag(pos, vel, densities, area):
    """
    Calculates drag force based on current position, velocity and parachute area
    :param pos: Current position in cartesian coordinates
    :param vel: Current velocity in cartesian coordinates
    :param area: Area of spacecraft/lander with or without parachutes in m^2
    :return: Current drag force on the lander in cartesian coordinates
    """
    C_D = 1
    radius_planet = 3775244.8601354226  # Radius of planet
    r = round(np.linalg.norm(pos) - radius_planet)  # Altitude above planet surface
    if r >= 200_000:  # If higher than border than atmosphere, the drag force is 0
        F_D = np.array([0, 0, 0])
        return F_D
    w = find_w(pos)  # Finding velocity of atmosphere at our position
    v_drag = vel-w
    v_drag_norm = v_drag/np.linalg.norm(v_drag)  # Direction of drag force
    F_D_abs = (1 / 2) * densities[r] * C_D * area * np.linalg.norm(v_drag) ** 2  # Magnitude of drag force
    F_D = - F_D_abs*v_drag_norm
    return F_D


def terminal_velocity(area_parachute):
    """
    Calculates absolute terminal velocity based on area of parachutes/lander
    :param area_parachute: Area of parachutes im m^2
    :return: Absolute terminal velocity
    """
    area_sc = mission.spacecraft_area
    m = mission.spacecraft_mass
    rho_atm = system.atmospheric_densities[planet_idx]
    C_D = 1
    v = np.sqrt(2*m*g/(rho_atm*C_D*(area_sc + area_parachute)))
    return v


def parachute_area(vel_term):
    """
    Calculates required parachute area of lander for a given terminal velocity
    :param vel_term: Terminal velocity
    :return: Total area of parachutes
    """
    m = mission.spacecraft_mass
    rho_atm = system.atmospheric_densities[planet_idx]
    C_D = 1
    area_tot = 2*m*g/(rho_atm*C_D*(vel_term**2))
    area_parachutes = area_tot-mission.spacecraft_area
    return area_parachutes


def plot_planet(start_angle=0, stop_angle=2*np.pi, altitude=0, label="Planet"):
    """
    Function to plot planet or other circle around the position (0, 0)
    :param start_angle: Angle where the plot starts [rad]
    :param stop_angle: Angle where the plot stops [rad]
    :param altitude: Altitude above the surface for which the plot should be made
    :param label: Label for the plot. Set to "Planet" as default
    :return: None (only a nice plot)
    """
    planet_radius = system.radii[1]*1000
    r = planet_radius + altitude
    angles = np.linspace(start_angle, stop_angle, 2000)
    x = r*np.cos(angles)
    y = r*np.sin(angles)
    plt.plot(x, y, label=label)


def simulate_landing(pos0, vel0, mass_sc, mass_lander, area_sc, area_lander, area_parachute, densities, landing_booster_force, dep_height_lander, dep_height_parachute, dep_height_booster):
    """
    Function which simulates landing of the lander on the planet using initial position and velocity and other variables
    :param pos0: Initial position array [x, y, z] in meters
    :param vel0: Initial velocity array [x, y, z] in meters
    :param mass_sc: Mass of the spacecraft
    :param mass_lander: Mass of the lander
    :param area_sc: Surface area of the spacecraft
    :param area_lander: Surface area of the lander
    :param area_parachute: Surface area of the parachute
    :param densities: Array with densities of the atmosphere
    :param landing_booster_force: Force of the landing thrusters
    :param dep_height_lander: Deployment height of the lander
    :param dep_height_parachute: Deployment height of the parachute
    :param dep_height_booster: Activation height of the landing thrusters
    :return: Time array; Position array; Velocity array; Array with absolute values of drag force; Array with indexes of events during landing
    """
    # Creating basic necessary parameters for simulation
    N = 400_000
    dt = 0.01
    m_planet = 7.277787918769816e+23
    radius_planet = 3775244.8601354226  # Radius of planet
    indexes = np.zeros(4)  # Array to store indexes when Entered atmosphere, Lander deployed, Parachute deployed, Boosters activated

    # Creating variables for some values, which can change during the simulation
    area_curr = area_sc  # Current total area of system
    mass_curr = mass_sc  # Current mass of system
    booster_force = 0  # Current landing booster force
    entered_atmosphere = False
    lander_deployed = False
    parachute_deployed = False
    boosters_activated = False

    # Creating and initialising arrays for the simulation
    time = np.zeros(N)
    pos = np.zeros([N, 3])  # Position
    vel = np.zeros([N, 3])  # Velocity
    fd_abs = np.zeros(N)
    pos[0] = pos0
    vel[0] = vel0
    fd_abs[0] = 0

    for i in range(N-1):
        # Using Leapfrog method
        f_grav = -G*((m_planet*mass_curr)/(np.linalg.norm(pos[i])**3))*pos[i]  # Calculating gravitational force
        f_drag = drag(pos[i], vel[i], densities, area_curr)  # Calculating drag force
        a = (f_grav + f_drag + (booster_force*(pos[i]/np.linalg.norm(pos[i]))))/mass_curr  # Finding current acceleration using newtons second law

        vh = vel[i] + a*dt/2  # Intermediate velocity
        pos[i+1] = pos[i] + vh*dt  # Next position

        f_grav = -G * ((m_planet * mass_curr) / (np.linalg.norm(pos[i+1]) ** 3)) * pos[i+1]  # Calculating gravitational force
        f_drag = drag(pos[i+1], vh, densities, area_curr)  # Calculating drag force
        fd_abs[i+1] = np.linalg.norm(f_drag)
        a = (f_grav + f_drag + (booster_force*(pos[i+1]/np.linalg.norm(pos[i+1]))))/mass_curr  # Finding current acceleration using newtons second law

        vel[i+1] = vh + a*dt/2  # Next velocity
        time[i] = dt*i  # Saving point in time

        # Executing braking boost if the atmospheric pressure is getting too high (Above 10^7 Pa)
        if lander_deployed and (np.linalg.norm(f_drag)/area_curr) > 10**7:
            v_previous = vel[i]
            vel[i+1] = 100*v_previous/np.linalg.norm(v_previous)  # Boosting/breaking to reduce velocity to 20 m/s
            print("Drag Pressure critical! Executed braking boost")

        # Raising error if the force on the parachute is too high (over 250'000 N)
        if parachute_deployed and np.linalg.norm(f_drag) > 250_000:
            print(f"Drag Force: {np.linalg.norm(f_drag)}")
            print(time[i], pos[i], vel[i])  # Printing time, position and velocity for problem-solving
            raise RuntimeError("Parachute failed due to excessive drag force!")

        # Deploying lander with same velocity if below the deployment height
        if not lander_deployed and ((np.linalg.norm(pos[i])-radius_planet) < dep_height_lander):
            lander_deployed = True
            indexes[1] = i  # Saving time index of event
            area_curr = area_lander  # Changing current area to lander area
            mass_curr = mass_lander  # Changing current mass to lander mass
            print(f"Lander Deployed after {time[i]:.1f} seconds")

        # Deploying main parachute if below the deployment height
        if (parachute_deployed is False) and ((np.linalg.norm(pos[i])-radius_planet) < dep_height_parachute):
            parachute_deployed = True
            indexes[2] = i-1  # Saving time index of event
            area_curr = area_lander + area_parachute  # Changing current area to lander area + main parachute area
            print(f"Main Parachute Deployed after {time[i]:.1f} seconds")


        # Activating landing boosters if below the activation height
        if not boosters_activated and ((np.linalg.norm(pos[i])-radius_planet) < dep_height_booster):
            boosters_activated = True
            indexes[3] = i-1  # Saving time index of event
            booster_force = landing_booster_force  # Changing current booster force to the actual booster force and therefore "turning them on"
            print(f"Landing boosters activated after {time[i]:.1f} seconds")

        # Checking if we have entered the atmosphere
        if (entered_atmosphere is False) and (np.linalg.norm(pos[i])-radius_planet) <= 200_000:
            print(f"\n\nEntered Atmosphere after {time[i]:.1f} seconds")
            indexes[0] = i-1  # Saving time index of event
            entered_atmosphere = True

        # Ending simulation if we are on the surface
        if np.linalg.norm(pos[i]) < radius_planet:
            time = time[:i+1]  # Slicing time array
            pos = pos[:i+1]  # Slicing position array
            vel = vel[:i+1]  # Slicing velocity array
            fd_abs = fd_abs[:i+1]  # Slicing drag force array
            break

    # Printing some info about the landing simulation after touchdown
    print(f"\nLander Deployed: {lander_deployed}")
    print(f"Parachute Deployed: {parachute_deployed}")
    print(f"Boosters activated: {boosters_activated}")
    print(f"\nLanding velocity: {vel[-1]-find_w(pos[-1])}")
    print(f"Landing speed: {-np.linalg.norm(vel[-1] - find_w(pos[-1]))} m/s")
    print(f"Landing position: {pos[-1]/np.linalg.norm(pos[-1])}")
    elapsed_time = round(time[-1])  # Calculating duration of landing in hours, minutes and seconds
    hours = elapsed_time // 3600
    minutes = (elapsed_time - (hours*3600)) // 60
    seconds = elapsed_time - (hours*3600) - (minutes*60)
    print(f"Landing Duration: {hours} hours, {minutes} minutes and {seconds} seconds")

    return time, pos, vel, fd_abs, indexes


if __name__ == "__main__":
    # FOR SIMPLICITY WE WILL ONLY MOVE ON THE XY-PLANE

    # Defining some variables from earlier parts
    dt = 8.666669555556518e-05
    time0 = 2762487 * dt
    orbital_height0 = 1_000_000
    orbital_angle0 = -0.31
    radius_planet = 3775244.8601354226  # Radius of planet

    # Putting Spacecraft into low stable orbit (requires verification of launch and orientation first)
    launch_angle = 260.483012
    t_launch = 2752487 * dt
    shortcut1.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t_launch, 3_000_000, launch_angle, 392_000)
    fuel_consumed, t_after_launch, r_after_launch, v_after_launch = shortcut1.get_launch_results()
    mission.verify_manual_orientation(r_after_launch, v_after_launch, 37.01285168461271)
    shortcut.place_spacecraft_in_stable_orbit(time0, orbital_height0, orbital_angle0, planet_idx)

    # Defining some constant variables for the landing such as masses, areas and altitudes
    densities = np.load("../../Part 6/Densities.npy")  # Array with densities at different heights above the surface
    sc_mass = mission.spacecraft_mass
    lander_mass = mission.lander_mass
    sc_area = mission.spacecraft_area
    lander_area = mission.lander_area
    main_chute_area = 7
    booster_force = 250
    dh_lander = 140_000  # dh = deployment height
    dh_parachute = 1_000
    dh_booster = 200

    landing_seq = mission.begin_landing_sequence()  # Creating landing sequence instance
    vel0 = landing_seq.orient()[2]  # Finding velocity to execute boost

    boost0_strength = 1000  # Strength of boost to fall out of orbit and initiate landing
    vel0_norm = vel0 / np.linalg.norm(vel0)
    landing_seq.boost(-boost0_strength * vel0_norm)  # Decreasing tangential velocity to initiate fall into atmosphere
    t0, pos0, vel0 = landing_seq.orient()  # Orienting ourselves after the boost

    # Calling function to simulate landing
    t, pos, vel, fd_abs, indexes = simulate_landing(pos0, vel0, sc_mass, lander_mass, sc_area, lander_area, main_chute_area, densities, booster_force, dh_lander, dh_parachute, dh_booster)

    labels = ["Entered Atmosphere", "Lander Deployed", "Parachute Deployed", "Boosters Activated"]  # List with labels for plotting

    # Plotting landing of spacecraft from a closer perspective
    plot_planet(10.5*np.pi/128, 14*np.pi/128, 200_000, label="Atmosphere border")  # Plotting border of atmosphere
    plot_planet(11 * np.pi / 128, 15 * np.pi / 128, label="Planet surface")  # Plotting planet
    [plt.scatter(pos[int(indexes[i]), 0], pos[int(indexes[i]), 1], label=labels[i]) for i in range(len(indexes)-1)]
    plt.plot(pos[int(len(pos)*43/64):, 0], pos[int(len(pos)*43/64):, 1], label="Spacecraft trajectory", c="k")  # Plotting last section of spacecraft trajectory
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("../Figures/sim_landing_close.png")  # Saving plot
    plt.show()

    # Plotting landing of spacecraft from a further perspective
    plt.plot(pos[:, 0], pos[:, 1], label="Spacecraft trajectory")  # Plotting spacecraft trajectory
    plot_planet(-np.pi/2, np.pi/2, label="Planet surface")  # Plotting planet
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("../Figures/sim_landing_far.png")  # Saving plot
    plt.show()

    # Calculating velocity relative to the planets surface for later plotting
    v_rel = np.array([find_w(pos[i]) for i in range(len(pos))])
    v_abs = np.array([np.linalg.norm(vel[i] - v_rel[i]) for i in range(len(vel))])

    # Plotting velocity during the landing
    [plt.scatter(t[int(indexes[i])], v_abs[int(indexes[i])], label=labels[i]) for i in range(len(indexes))]  # Plotting points where different events during the landing happened
    plt.plot(t, v_abs, label="Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute velocity [m/s]")
    plt.legend()
    plt.savefig("../Figures/sim_landing_velocity.png")  # Saving plot
    plt.show()

    # Plotting drag force during the landing
    [plt.scatter(t[int(indexes[i])], fd_abs[int(indexes[i])], label=labels[i]) for i in range(len(indexes))]  # Plotting points where different events during the landing happened
    plt.plot(t, fd_abs)
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute drag force [N]")
    plt.legend()
    plt.savefig("../Figures/sim_landing_f_drag.png")  # Saving plot
    plt.show()

