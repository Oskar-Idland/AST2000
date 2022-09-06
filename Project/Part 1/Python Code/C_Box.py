import numpy as np
import random
from scipy.constants import Boltzmann as k


class Box:
    def __init__(self, L, nozzle_area, nozzle_pos="x+"):
        self.length = L  # Length of one side of the box
        self.nozzle_area = nozzle_area  # Nozzle Area
        self.nozzle_rad = np.sqrt(nozzle_area/np.pi)  # Nozzle Radius
        self.nozzle_pos = nozzle_pos  # Position where the nozzle is located on
        pos_dict = {"x+": [1, 1], "x-": [1, -1], "y+": [2, 1], "y-": [2, -1], "z+": [3, 1], "z-": [3, -1]}  # Just a dictionary to define the nozzle_axis and nozzle_side parameters from the nozzle_pos input
        self.nozzle_axis = pos_dict[nozzle_pos][0]  # Axis which the nozzle is positioned on. x = 0, y = 1, z = 3
        self.nozzle_side = pos_dict[nozzle_pos][1]  # Side which the nozzle is on (-1 for negative, 1 for positive)