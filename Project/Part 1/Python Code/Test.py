import numpy as np
from C_Particle import Particle
from C_Box import Box

x = np.array([Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2]))),Particle(1,3000,4,Box(1e-6,np.array([0,0,-1e-6/2])))])

p = Particle(1,3000,2,Box(1e-6,np.array([0,0,-1e-6/2])))
[p.advance() for _ in range(3)]
print(p.position)
print(np.nonzero(abs(p.position) >= 4.5*1e-7)[0])

