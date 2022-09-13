import numpy as np
from C_Particle import Particle
from C_Box import Box
import matplotlib.pyplot as plt
from scipy.constants import Avogadro as A
from molmass import Formula
from itertools import product, combinations

# x = np.array([Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2]))),Particle(1,3000,4,Box(1e-6,np.array([0,0,-1e-6/2])))])

# x = np.array([[1,2,3]])
# print(x)
# x = np.append(x,np.array([[4,5,6]]), axis=0)
# print(x[-1])

seed = 0
m = Formula('H2').mass/(A*1000) # Mass of one H2 molecule In Kg
L = 1E-6 
T = 3*1E3
box = Box(L, np.array([0,0,-L/2]))
p = Particle(m, T, seed, box)

[p.advance() for _ in range(1000)]



# box_cordx = [-L/2, L/2, L/2, -L/2, -L/2]
# box_cordy = [-L/2, -L/2, L/2, L/2, -L/2]
# nozzle_cordx = [-box.nozzle_rad, box.nozzle_rad]
# nozzle_cordy = [-L/2, -L/2]
# plt.plot(box_cordx,box_cordy, c = 'blue')
# plt.hlines([-L/2, -L/2], -box.nozzle_rad, box.nozzle_rad, color = 'orange', label = 'nozzle')
# print(p.position)
px = []
for arr in p.position:
    px.append(arr[0])

py = []
for arr in p.position:
    py.append(arr[1])

pz = []
for arr in p.position:
    pz.append(arr[2])


print('--------------------------------')
print('Momentum: ' + str(m*p.v_exit))


fig = plt.figure()
ax = plt.axes(projection = '3d')

# Draw the cube
r = [-L/2, L/2]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="blue", alpha=0.4)

ax.plot3D(px,py,pz, c = 'red', label = 'path')
ax.scatter(px[0],py[0], pz[0], label = 'start')
ax.scatter(px[10],py[10], pz[10], label = 'first 10 steps')
ax.scatter(px[-1],py[-1], pz[-1], label = 'end')
# Nozzle
x = np.linspace(0,2*np.pi, 100)
ax.plot(box.nozzle_rad*np.cos(x), box.nozzle_rad*np.sin(x), -L/2, label = 'Nozzle', c = 'black')
ax.legend()
plt.show()