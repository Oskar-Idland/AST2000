'''
File for global variables
'''

import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

# Initializing system
seed = 36874
system = SolarSystem(seed)
mission = SpaceMission(seed)