import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from Part_7_B import drag, find_w, plot_planet

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
g = G * m_planet / (system.radii[1] * 1000)  # Gravitational acceleration on the planet
rho_atm = system.atmospheric_densities[planet_idx]  # Average atmospheric density
C_D = 1  # Drag coefficient
omega_atm = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)  # Angular velocity of the atmosphere


def landing(land_seq, pos0, vel0, dh_lander, dh_parachute, dh_booster, area_sc, area_lander, area_parachute, densities, landing_booster_force):
    """
    Function to land the rover on the surface of the planet using initial position and velocity and other variables
    :param land_seq: Landing Sequence class instance
    :param pos0: Initial position array [x, y, z] in meters
    :param vel0: Initial velocity array [x, y, z] in meters
    :param dh_lander: Deployment height of the lander
    :param dh_parachute: Deployment height of the parachute
    :param dh_booster: Activation height of the landing thrusters
    :param area_sc: Surface area of the spacecraft
    :param area_lander: Surface area of the lander
    :param area_parachute: Surface area of the parachute
    :param densities: Array with densities of the atmosphere
    :param landing_booster_force: Force of the landing thrusters
    :return: Time array; Position array; Velocity array; Array with absolute values of drag force; Array with indexes of events during landing
    """
    # Setting some values for the landing
    N = 10_000  # Number of time steps
    fall_step = 1  # Length of time step
    radius_planet = 3775244.8601354226  # Radius of planet
    land_seq.adjust_landing_thruster(landing_booster_force, dh_booster)  # Adjusting landing thruster strength and activation height
    land_seq.adjust_parachute_area(area_parachute)  # Adjusting parachute area
    area_curr = area_sc  # Creating a variable for the current area
    entered_atmosphere = False  # Status variable
    boosters_active = False  # Status variable

    time = np.zeros(N)  # Array for time points
    pos = np.zeros([N, 3])  # Position
    vel = np.zeros([N, 3])  # Velocity
    fd_abs = np.zeros(N)  # Absolute drag force
    indexes = np.zeros(4)  # Array to store time indexes of events

    # Initializing arrays
    pos[0] = pos0
    vel[0] = vel0
    fd_abs[0] = 0

    # Integration loop
    for i in range(1, N):
        land_seq.fall(fall_step)  # Integrating for fall_step seconds
        time[i], pos[i], vel[i] = land_seq.orient()  # Orienting ourselves and saving current position, velocity and time
        fd_abs[i] = np.linalg.norm(drag(pos[i], vel[i], densities, area_curr))  # Saving absolute drag force

        # Executing braking boost if atmospheric pressure is getting too high
        if entered_atmosphere and (land_seq.lander_launched is False) and fd_abs[i] > 20_000:
            land_seq.boost(-vel[i] / 5)  # Braking with 1/5 of our current velocity

        # Deploying lander if we are below the deployment height
        if (land_seq.lander_launched is False) and ((np.linalg.norm(pos[i]) - radius_planet) <= dh_lander):
            w = find_w(pos[i])
            delta_v = w-vel[i]  # Calculating a boost so that the lander is launched with as little velocity as possible relative to the atmosphere
            land_seq.launch_lander(delta_v)  # Launching lander
            area_curr = area_lander  # Changing current area to lander area
            indexes[1] = i + 1  # Saving index of event

        # Deploying parachute if we are below deployment height
        if land_seq.lander_launched and (land_seq.parachute_deployed is False) and ((np.linalg.norm(pos[i]) - radius_planet) < dh_parachute):
            land_seq.deploy_parachute()  # Deploying parachute
            area_curr = area_lander + area_parachute  # Changing current area to lander area + parachute area
            indexes[2] = i

        # Checking whether we have entered the atmosphere
        if (entered_atmosphere is False) and ((np.linalg.norm(pos[i]) - radius_planet) <= 200_000):
            entered_atmosphere = True
            indexes[0] = i + 1  # Saving index of event

        # Checking whether landing boosters were activated
        if (boosters_active is False) and land_seq.landing_thruster_activated:
            boosters_active = True
            indexes[3] = i - 1  # Saving index of event

        # Checking whether the lander has reached the surface
        if land_seq.reached_surface:
            # Slicing arrays
            pos = pos[:i]
            vel = vel[:i]
            time = time[:i]
            fd_abs = fd_abs[:i]
            break  # Ending integration loop

    return time, pos, vel, fd_abs, indexes  # Returning arrays

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
    dh_main_chute = 1_000
    dh_booster = 200

    landing_seq = mission.begin_landing_sequence()  # Creating landing sequence instance
    vel0 = landing_seq.orient()[2]  # Finding velocity to execute boost

    boost0_strength = 1000  # Strength of boost to fall out of orbit and initiate landing
    vel0_norm = vel0 / np.linalg.norm(vel0)
    landing_seq.boost(-boost0_strength * vel0_norm)  # Decreasing tangential velocity to initiate fall into atmosphere
    t0, pos0, vel0 = landing_seq.orient()  # Orienting ourselves after the boost

    landing_seq.look_in_direction_of_planet(1)  # Setting up camera
    landing_seq.start_video()  # Starting video recording

    # Calling function to land the lander
    t, pos, vel, fd_abs, indexes = landing(landing_seq, pos0, vel0, dh_lander, dh_main_chute, dh_booster, sc_area, lander_area, main_chute_area, densities, booster_force)

    landing_seq.finish_video()  # Stopping video recording

    labels = ["Entered Atmosphere", "Lander Deployed", "Parachute Deployed", "Boosters Activated"]  # List with labels for plotting

    # Plotting landing of spacecraft from a closer perspective
    plot_planet(9.5 * np.pi / 128, 16.5 * np.pi / 128, 200_000, label="Atmosphere border")  # Plotting border of atmosphere
    plot_planet(10 * np.pi / 128, 17.5 * np.pi / 128, label="Planet surface")  # Plotting planet
    [plt.scatter(pos[int(indexes[i]), 0], pos[int(indexes[i]), 1], label=labels[i]) for i in range(len(indexes) - 1)]
    plt.plot(pos[int(len(pos) * 24.5 / 64):, 0], pos[int(len(pos) * 24.5 / 64):, 1], label="Spacecraft trajectory", c="k")  # Plotting last section of spacecraft trajectory
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("../Figures/landing_close.png")  # Saving plot
    plt.show()

    # Plotting landing of spacecraft from a further perspective
    plt.plot(pos[:, 0], pos[:, 1], label="Spacecraft trajectory")  # Plotting spacecraft trajectory
    plot_planet(-np.pi / 2, np.pi / 2, label="Planet surface")  # Plotting planet
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("../Figures/landing_far.png")  # Saving plot
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
    plt.savefig("../Figures/landing_velocity.png")  # Saving plot
    plt.show()

    # Plotting drag force during the landing
    [plt.scatter(t[int(indexes[i])], 0.001*fd_abs[int(indexes[i])], label=labels[i]) for i in range(len(indexes))]  # Plotting points where different events during the landing happened
    plt.plot(t, 0.001*fd_abs)  # Scaling the drag force array by 0.001 to express it in Kilonewtons
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute drag force [kN]")
    plt.legend()
    plt.savefig("../Figures/landing_f_drag.png")  # Saving plot
    plt.show()

