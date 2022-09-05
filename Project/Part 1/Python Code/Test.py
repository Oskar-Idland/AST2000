import numpy as np

class Test:
    def __init__(self, identification):
        self.id = identification
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([1, 2, 3])

    def advance(self, dt):
        np.append(self.pos, self.vel*dt)


test_array = np.array([Test(1), Test(2), Test(3)])

print(test_array)
