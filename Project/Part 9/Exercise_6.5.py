import numpy as np
import matplotlib.pyplot as plt

N = 10000
m = 1
M = 1
R = 20*M
phi_0 = 167/180*np.pi
v_shell = 0.993
y_shell = 1/np.sqrt(1-(v_shell**2))


r = np.linspace(0, R, N)
L_m = R*y_shell*v_shell*np.sin(phi_0)
E_m = np.sqrt(1-(2*M/R))*y_shell

V = np.sqrt((1-((2*M)/r)) * (1 + (L_m / r)**2))
i_max = np.where(V[1000:] == np.amax(V[1000:]))[0] + 1000


plt.plot(r, V, c="k", label="V(t)")
plt.scatter(r[i_max], V[i_max], c="r", label=r"$\frac{E_{crit}}{m}$")
plt.axhline(y=E_m, c="r", linestyle="--", label=r"$\frac{E}{m}$")
plt.xlabel("Radius [M]")
plt.ylabel("Effective Potential [V(r)]")
plt.legend()
plt.grid()
plt.savefig("ex6_effpotential")
plt.show()

