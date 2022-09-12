import numpy as np
from C_Particle import Particle
from C_Box import Box
def pname(p):
    return p.velocity
x = np.array([Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2]))),Particle(1,3000,4,Box(1e-6,np.array([0,0,-1e-6/2])))])

print(list(map(lambda p: p.name, x)))

p = Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2])))
print(p.name)
x = [Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2]))),Particle(1,3000,4,Box(1e-6,np.array([0,0,-1e-6/2])))]

