import numpy as np
import ast2000tools.utils as utils
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


def landing(land_seq, pos0, vel0, dh_lander, dh_parachute, dh_booster, area_sc, area_lander, area_parachute, densities,
            landing_booster_force):
    # Setting some values for the landing
    N = 1_000_000
    fall_step = 1
    radius_planet = 3775244.8601354226  # Radius of planet
    land_seq.adjust_landing_thruster(landing_booster_force, dh_booster)
    land_seq.adjust_parachute_area(area_parachute)
    area_curr = area_sc
    entered_atmosphere = False

    time = np.zeros(N)
    pos = np.zeros([N, 3])
    vel = np.zeros([N, 3])
    fd_abs = np.zeros(N)
    indexes = np.zeros(5)

    pos[0] = pos0
    vel[0] = vel0
    fd_abs[0] = 0

    for i in range(1, N):
        try:
            land_seq.fall(fall_step)
        except:
            pass
        time[i], pos[i], vel[i] = land_seq.orient()
        fd_abs[i] = np.linalg.norm(drag(pos[i], vel[i], densities, area_curr))
        print(fd_abs[i] / area_curr)

        # Executing braking boost if atmospheric pressure is getting too high
        if land_seq.lander_launched and fd_abs[i] / area_curr > 2200:
            land_seq.boost(-vel[i] / 4)
            pass

        if (land_seq.lander_launched is False) and (np.linalg.norm(pos[i]) <= dh_lander):
            land_seq.launch_lander()
            area_curr = area_lander
            indexes[1] = i - 1

        if land_seq.lander_launched and (land_seq.parachute_deployed is False) and (
                np.linalg.norm(pos[i]) <= dh_parachute):
            land_seq.deploy_parachute()
            area_curr = area_lander + area_parachute
            indexes[3] = i - 1

        if (entered_atmosphere is False) and ((np.linalg.norm(pos[i]) - radius_planet) <= 200_000):
            indexes[0] = i - 1

        if land_seq.reached_surface:
            pos = pos[:i]
            vel = vel[:i]
            time = time[:i]
            fd_abs = fd_abs[:i]
            print(f"Landing speed: {-np.linalg.norm(vel[-1] - find_w(pos[-1]))} m/s")
            break

    return time, pos, vel, fd_abs, indexes


if __name__ == "__main__":
    # FOR SIMPLICITY WE WILL ONLY MOVE ON THE XY-PLANE

    # Defining some variables from earlier parts
    dt = 8.666669555556518e-05
    time0 = 2762487 * dt
    orbital_height0 = 1_000_000
    orbital_angle0 = 0
    radius_planet = 3775244.8601354226  # Radius of planet

    # Putting Spacecraft into low stable orbit (requires verification of launch and orientation first)
    launch_angle = 260.483012
    t_launch = 2752487 * dt
    shortcut1.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t_launch, 3_000_000, launch_angle,
                                                    392_000)
    fuel_consumed, t_after_launch, r_after_launch, v_after_launch = shortcut1.get_launch_results()
    mission.verify_manual_orientation(r_after_launch, v_after_launch, 37.01285168461271)
    shortcut.place_spacecraft_in_stable_orbit(time0, orbital_height0, orbital_angle0, planet_idx)

    # Defining some constant variables for the landing such as masses, areas and altitudes
    densities = np.load("../../Part 6/Densities.npy")  # Array with densities at different heights above the surface
    sc_mass = mission.spacecraft_mass
    lander_mass = mission.lander_mass
    sc_area = mission.spacecraft_area
    lander_area = mission.lander_area
    main_chute_area = 100
    drogue_chute_area = 0  # 20
    booster_force = 0
    dh_lander = 100_000  # dh = deployment height
    dh_main_chute = 5_000
    dh_drogue_chute = 20_000
    dh_booster = 200

    landing_seq = mission.begin_landing_sequence()  # Creating landing sequence instance
    vel0 = landing_seq.orient()[2]  # Finding velocity to execute boost

    boost0_strength = 1000  # Strength of boost to initiate landing
    vel0_norm = vel0 / np.linalg.norm(vel0)
    landing_seq.boost(-boost0_strength * vel0_norm)  # Decreasing tangential velocity to initiate fall into atmosphere
    t0, pos0, vel0 = landing_seq.orient()  # Orienting ourselves after the boost

    t, pos, vel, fd_abs, indexes = landing(landing_seq, pos0, vel0, dh_lander, dh_main_chute, dh_booster, sc_area,
                                           lander_area, main_chute_area, densities, booster_force)

    """
    # Calling function to simulate landing
    t, pos, vel, fd_abs, indexes = simulate_landing(pos0, vel0, sc_mass, lander_mass, sc_area, lander_area, main_chute_area, drogue_chute_area, densities, booster_force, dh_lander, dh_main_chute, dh_drogue_chute, dh_booster)
    """
    labels = ["Entered Atmosphere", "Lander Deployed", "Main Deployed", "Drogue Deployed",
              "Boosters Activated"]  # List with labels for plotting

    # Plotting landing of spacecraft from a closer perspective
    plt.plot(pos[int(len(pos) * 41 / 64):, 0], pos[int(len(pos) * 41 / 64):, 1], label="Spacecraft trajectory")
    plot_planet(24 * np.pi / 128, 28 * np.pi / 128, label="Planet")  # Plotting planet
    plot_planet(23 * np.pi / 128, 26.5 * np.pi / 128, 200_000, label="Atmosphere")  # Plotting border of atmosphere
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("../Figures/landing_close.png")  # Saving plot
    plt.show()

    # Plotting landing of spacecraft from a further perspective
    plt.plot(pos[:, 0], pos[:, 1], label="Spacecraft trajectory")
    plot_planet(0, np.pi / 2, label="Planet")  # Plotting planet
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
    [plt.scatter(t[int(indexes[i])], v_abs[int(indexes[i])], label=labels[i]) for i in range(len(indexes))]
    plt.plot(t, v_abs, label="Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute velocity [m/s]")
    plt.legend()
    plt.savefig("../Figures/landing_velocity.png")
    plt.show()

    # Plotting drag force during the landing
    [plt.scatter(t[int(indexes[i])], fd_abs[int(indexes[i])], label=labels[i]) for i in range(len(indexes))]
    plt.plot(t, fd_abs)
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute drag force [N]")
    plt.legend()
    plt.savefig("../Figures/landing_f_drag.png")
    plt.show()

