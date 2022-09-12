import numpy as np
L = 1
x = np.array([[0, 1, 2], [2, 1, 0]])
print(x[np.nonzero(x >= 1)[0][0]][np.nonzero(x >= 1)[1]])
