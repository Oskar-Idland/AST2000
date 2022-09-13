import numpy as np
from C_Particle import Particle
from C_Box import Box
import matplotlib.pyplot as plt
from scipy.constants import Avogadro as A
from molmass import Formula

# x = np.array([Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2]))),Particle(1,3000,4,Box(1e-6,np.array([0,0,-1e-6/2])))])

# x = np.array([[1,2,3]])
# print(x)
# x = np.append(x,np.array([[4,5,6]]), axis=0)
# print(x[-1])

seed = 8
m = Formula('H2').mass/(A*1000) # Mass of one H2 molecule In Kg
L = 1E-6 
T = 3*1E3
box = Box(L, np.array([0,-L/2,0]))
p = Particle(m, T, seed, box)
[p.advance() for _ in range(8000)]

box_cordx = [-L/2, L/2, L/2, -L/2, -L/2]
box_cordy = [-L/2, -L/2, L/2, L/2, -L/2]
nozzle_cordx = [-box.nozzle_rad, box.nozzle_rad]
nozzle_cordy = [-L/2, -L/2]
plt.plot(box_cordx,box_cordy, c = 'blue')
plt.hlines([-L/2, -L/2], -box.nozzle_rad, box.nozzle_rad, color = 'orange', label = 'nozzle')
# print(p.position)
px = []
for arr in p.position:
    px.append(arr[0])
py = []
for arr in p.position:
    py.append(arr[1])
print(p.position[-1], px[-1])
# py = p.position[::3][1]
print('--------------------------------')
print(p.p_exit)
# print(px)
# print(py)
plt.plot(px,py, c = 'red', label = 'path')
plt.scatter(px[0],py[0], label = 'start')
plt.scatter(px[10],py[10], label = 'first 10 steps')
plt.scatter(px[-1],py[-1], label = 'end')
plt.legend()
print(px[0] - px[-1])
print(np.linalg.norm(p.position[0]) - np.linalg.norm(p.position[-1]))
plt.show()