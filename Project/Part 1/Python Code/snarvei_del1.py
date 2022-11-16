import ast2000tools.constants as con
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
import numpy as np

seed = 36874

code_launch_results = 83949
code_escape_trajectory = 74482

mission = SpaceMission(seed)
system = SolarSystem(seed)
shortcut = SpaceMissionShortcuts(mission, [code_launch_results, code_escape_trajectory])

"""
Documentation
place_spacecraft_on_escape_trajectory(
    rocket_thrust, rocket_mass_loss_rate, time height_above_surface,
    direction_angle, remaining_fuel_mass):

------------------------------------------------------------------------
place_spacecraft_on_escape_trajectory() places the spacecraft on an
escape trajectory pointing directly away from the home planet.

Parameters
----------
rocket_thrust  :  float
    The total thrust of the rocket, in NEWTONS.

rocket_mass_loss_rate  :  float
    The total mass loss rate of the rocket, in KILOGRAMS PER SECOND.

time  :  float
    The time at which the spacecraft should be placed on the escape
    trajectory, in YEARS from the initial system time.

height_above_surface  :  float
    The heigh above the home planet surface to place the spacecraft, in
    METERS (after launch).

direction_angle  :  float
    The angle of the direction of motion of the spacecraft with respect
    to the x-axis, in DEGREES.

remaining_fuel_mass  :  float
    The mass of fuel carried by the spacecraft after placing it on the
    escape trajectory, in KILOGRAMS.

Raises
------
RuntimeError
    When none of the provided codes are valid for unlocking this method.
------------------------------------------------------------------------

"""

def launch_rocket_shortcut(thrust, mass_loss_rate, time, height_above_suface, direction_angle, fuel_left):
    # thrust = # insert the thrust force of your spacecraft here
    # mass_loss_rate = # insert the mass loss rate of your spacecraft here
    #
    # # choose these values freely, but they should be relevant to where you
    # # want to go, e.g., if you want to travel outwards of your solar system,
    # # don't let the direction angle be 0 if you are launching from
    # # coordinates close to (-x, 0), as this will send you in the opposite
    # # direction), and vice versa if your destination is a planet closer to
    # # your sun
    # time = # insert the time for the spacecraft to be put on escape trajectory
    # height_above_suface = # insert height above surface you want the rocket placed
    # direction_angle = # insert the angle between the x-axis and the rocket's motion
    # fuel_left = # insert how much fuel you want for your trip

    shortcut.place_spacecraft_on_escape_trajectory(thrust, mass_loss_rate, time, height_above_suface, direction_angle, fuel_left)

    fuel_consumed, time_after_launch, pos_after_launch, vel_after_launch = shortcut.get_launch_results()
    # fuel_consumed is None because place_spacecraft_on_escape_trajectory()
    # don't actually launch the rocket, but just place the rocket in correct
    # position and with the correct velocity.

    return pos_after_launch, vel_after_launch, time, fuel_consumed
