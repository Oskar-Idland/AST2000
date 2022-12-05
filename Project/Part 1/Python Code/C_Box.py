import numpy as np

class Box:
    """
    Class for a bos, which holds attributes such as length, nozzle area, position and radius
    """
    def __init__(self, L, nozzle_pos):
        self.length = L  # Length of one side of the box
        self.nozzle_area = 0.25*L**2  # Nozzle Area
        self.nozzle_rad = np.sqrt(self.nozzle_area/np.pi)  # Nozzle Radius
        self.nozzle_pos = nozzle_pos  # Position [x, y, z] where the nozzle is located on
        # nozzle_pos_dict = {"x+": [0, 1], "x-": [0, -1], "y+": [1, 1], "y-": [1, -1], "z+": [2, 1], "z-": [2, -1]}  # Just a dictionary to define the nozzle_axis and nozzle_side parameters from the nozzle_pos input
        # self.nozzle_axis = nozzle_pos_dict[nozzle_pos][0]  # Axis which the nozzle is positioned on. x = 0, y = 1, z = 3
        # self.nozzle_side = nozzle_pos_dict[nozzle_pos][1]  # Side which the nozzle is on (-1 for negative, 1 for positive)


