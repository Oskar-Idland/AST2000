import numpy as np
import sys
import ast2000tools.utils as utils
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
# sys.path.append("../../Part 1/Python Code")
# from Rocket_launch import launch_rocket  # It works when running, even if it shows an error in the editor!
from numba import njit

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
planet_idx = 1
G = 6.6743015e-11


m_planet = system.masses[planet_idx] * 1.98847e30
g = G*m_planet/(system.radii[1]*1000)
rho_atm = system.atmospheric_densities[planet_idx]
C_D = 1
omega_atm = (2 * np.pi) / (system.rotational_periods[planet_idx] * 24 * 3600)


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


def find_w(current_pos, planet_idx=1):
    """
    Finds velocity of atmosphere based on the position relative to the planet
    :param current_pos: Current position relative to the planet in cartesian coordinates
    :return: velocity of the atmosphere at the given position in cartesian coordinates
    """
    r, theta, phi = cart_to_spherical(current_pos[0], current_pos[1], current_pos[2])  # Converting to spherical coordinates
    omega = (2*np.pi)/(system.rotational_periods[planet_idx]*24*3600)  # Angular velocity of planet
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
    r = round(np.linalg.norm(pos) - radius_planet)
    if r >= 200_000:
        F_D = np.array([0, 0, 0])
        return F_D
    w = find_w(pos)
    # print(w)
    v_drag = vel-w
    print(np.linalg.norm(v_drag))
    v_drag_norm = v_drag/np.linalg.norm(v_drag)
    # print(v_drag_norm)
    F_D_abs = (1 / 2) * densities[r] * C_D * area * np.linalg.norm(v_drag) ** 2
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


def plot_planet():
    planet_radius = system.radii[1]*1000
    angles = np.linspace(0, np.pi/2, 2000)
    x = planet_radius*np.cos(angles)
    y = planet_radius*np.sin(angles)
    plt.plot(x, y)


def simulate_landing(pos0, vel0, mass_sc, mass_lander, area_sc, area_lander, area_main_parachute, area_drogue_parachute, densities, landing_booster_force, dep_height_lander, dep_height_main_chute, dep_height_drogue_chute, dep_height_booster):
    N = 200_000
    dt = 0.04
    m_planet = 7.277787918769816e+23
    radius_planet = 3775244.8601354226  # Radius of planet

    area_curr = area_sc  # Creating variables for some values, which can change during the simulation
    mass_curr = mass_sc
    booster_force = 0
    lander_deployed = False
    main_parachute_deployed = False
    drogue_parachute_deployed = False
    boosters_activated = False

    time = np.zeros(N)
    pos = np.zeros([N, 3])
    vel = np.zeros([N, 3])
    pos[0] = pos0
    vel[0] = vel0

    for i in range(N-1):
        # Using Leapfrog method
        f_grav = -G*((m_planet*mass_curr)/(np.linalg.norm(pos[i])**3))*pos[i]
        # print(pos[i]/np.linalg.norm(pos[i]))
        # print(area_curr)
        f_drag = drag(pos[i], vel[i], densities, area_curr)
        a = (f_grav + f_drag + (booster_force*(pos[i]/np.linalg.norm(pos[i]))))/mass_curr  # Finding current acceleration using newtons second law
        # print(np.linalg.norm(pos[i])-radius_planet)
        #print(f"f_drag: {f_drag}")
        # print(f"Height: {np.linalg.norm(pos[i])}")
        # print(f"f_grav: {np.linalg.norm(f_grav)}")
        # print(f"a: {np.linalg.norm(a)}")
        # print(f"vel[i]: {np.linalg.norm(vel[i])}")

        vh = vel[i] + a*dt/2
        pos[i+1] = pos[i] + vh*dt

        f_grav = -G * ((m_planet * mass_curr) / (np.linalg.norm(pos[i+1]) ** 3)) * pos[i+1]
        f_drag = drag(pos[i+1], vh, densities, area_curr)
        a = (f_grav + f_drag + (booster_force*(pos[i+1]/np.linalg.norm(pos[i+1]))))/mass_curr  # Finding current acceleration using newtons second law

        vel[i+1] = vh + a*dt/2
        time[i] = dt*i

        if lander_deployed and (np.linalg.norm(f_drag)/area_curr) > 10**7:
            v_previous = vel[i]
            vel[i+1] = 100*v_previous/np.linalg.norm(v_previous)  # Boosting/breaking to reduce velocity to 20 m/s
            print("Drag Pressure critical! Executed braking boost")

        if main_parachute_deployed and np.linalg.norm(f_drag) > 250_000:
            print(f"Drag Force: {np.linalg.norm(f_drag)}")
            print(time[i], pos[i], vel[i])  # Printing time, position and velocity for problem-solving
            raise RuntimeError("Parachute failed due to excessive drag force!")

        if not lander_deployed and (np.linalg.norm(pos[i]) < dep_height_lander):
            lander_deployed = True
            area_curr = area_lander
            mass_curr = mass_lander
            print("Lander Deployed")

        if drogue_parachute_deployed and (main_parachute_deployed == False) and (np.linalg.norm(pos[i]) < dep_height_main_chute):
            main_parachute_deployed = True
            area_curr = area_lander + area_main_parachute
            # v_previous = vel[i+1]
            # vel[i + 1] = 10 * v_previous / np.linalg.norm(v_previous)  # Boosting/breaking to reduce velocity to 20 m/s
            print("Main Parachute Deployed")

        if (drogue_parachute_deployed == False) and (np.linalg.norm(pos[i]) < dep_height_drogue_chute):
            drogue_parachute_deployed = True
            area_curr = area_lander + area_drogue_parachute
            print("Drogue Parachute Deployed")

        if not boosters_activated and np.linalg.norm(pos[i]) < dep_height_booster:
            boosters_activated = True
            booster_force = landing_booster_force
            print("Landing boosters activated")

        if np.linalg.norm(pos[i]) < radius_planet:
            time = time[:i+1]
            pos = pos[:i+1]
            vel = vel[:i+1]
            break

    print(f"Lander Deployed: {lander_deployed}")
    print(f"Main Parachute Deployed: {main_parachute_deployed}")
    print(f"Drogue Parachute Deployed: {drogue_parachute_deployed}")
    print(f"Boosters activated: {boosters_activated}")
    print(f"Landing velocity: {vel[-1]-find_w(pos[-1])}")
    print(f"Landing position: {pos[-1]/np.linalg.norm(pos[-1])}")
    return time, pos, vel



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
    shortcut1.place_spacecraft_on_escape_trajectory(6_000_000, 273.73826154189527, t_launch, 3_000_000, launch_angle, 392_000)
    fuel_consumed, t_after_launch, r_after_launch, v_after_launch = shortcut1.get_launch_results()
    mission.verify_manual_orientation(r_after_launch, v_after_launch, 37.01285168461271)
    shortcut.place_spacecraft_in_stable_orbit(time0, orbital_height0, orbital_angle0, planet_idx)


    densities = np.load("../../Part 6/Densities.npy")
    sc_mass = mission.spacecraft_mass
    lander_mass = mission.lander_mass
    sc_area = mission.spacecraft_area
    lander_area = mission.lander_area
    main_chute_area = 1000
    drogue_chute_area = 250
    booster_force = 0
    dh_lander = 60_000 + radius_planet
    dh_main_chute = 30_000 + radius_planet
    dh_drogue_chute = 50_000 + radius_planet
    dh_booster = 300 + radius_planet

    landing_seq = mission.begin_landing_sequence()  # Creating landing sequence instance
    t0, pos0, vel0 = landing_seq.orient()

    boost0_strength = 500
    vel0_norm = vel0 / np.linalg.norm(vel0)
    landing_seq.boost(-boost0_strength * vel0_norm)  # Decreasing tangential velocity to initiate fall into atmosphere

    t0, pos0, vel0 = landing_seq.orient()

    t, pos, vel = simulate_landing(pos0, vel0, sc_mass, lander_mass, sc_area, lander_area, main_chute_area, drogue_chute_area, densities, booster_force, dh_lander, dh_main_chute, dh_drogue_chute, dh_booster)

    plt.plot(pos[:, 0], pos[:, 1])  # Plotting trajectory of spacecraft
    plot_planet()
    plt.axis("equal")
    plt.show()

    # N = 1000
    # pos_list = np.zeros([N, 3])
    # for i in range(N):
    #     ti, posi, veli = landing_seq.orient()
    #     pos_list[i] = posi
    #     print(f"Velocity: {np.linalg.norm(veli)}")
    #     print(f"Position: {np.linalg.norm(posi)/1000}")
    #     landing_seq.fall(2)
    #
    # t0, pos0, vel0 = landing_seq.orient()
    # boost0_strength = 100
    # vel0_norm = pos0 / np.linalg.norm(pos0)
    # landing_seq.boost(boost0_strength * vel0_norm)
    #
    # pos_list1 = np.zeros([N, 3])
    # for i in range(N):
    #     ti, posi, veli = landing_seq.orient()
    #     pos_list1[i] = posi
    #     print(f"Velocity: {np.linalg.norm(veli)}")
    #     print(f"Position: {np.linalg.norm(posi) / 1000}")
    #     landing_seq.fall(1)
    #
    # t0, pos0, vel0 = landing_seq.orient()
    # boost0_strength = -2000
    # vel0_norm = vel0 / np.linalg.norm(vel0)
    # landing_seq.boost(boost0_strength * vel0_norm)
    #
    # pos_list2 = np.zeros([N, 3])
    # for i in range(N):
    #     ti, posi, veli = landing_seq.orient()
    #     pos_list2[i] = posi
    #     print(f"Velocity: {np.linalg.norm(veli)}")
    #     print(f"Position: {np.linalg.norm(posi) / 1000}")
    #     landing_seq.fall(0.5)
    #
    # plot_planet()
    # plt.plot(pos_list[:, 0], pos_list[:, 1])
    # plt.plot(pos_list1[:, 0], pos_list1[:, 1])
    # plt.plot(pos_list2[:, 0], pos_list2[:, 1])
    # plt.axis("equal")
    # plt.show()


