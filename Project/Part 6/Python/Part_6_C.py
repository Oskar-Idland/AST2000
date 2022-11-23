import os
import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as constants
G = constants.G
k = constants.k_B


# Initializing system
seed = 36874
system = SolarSystem(seed)
mission = SpaceMission(seed)
planet_index = 1
planet_radius = system.radii[planet_index]*1e3              # m
planet_mass = system.masses[planet_index]*constants.m_sun   # Kg
ρ_surface = system.atmospheric_densities[planet_index]
T_surface = 420                                             # Kelvin
γ = 1.4
m_hydrogen = 1.00784 * 1.66 * 1e-27                         # Mass of hydrogen atom
μ = 31.75                                                   # Kg - mean molecular weight of our atmosphere of pure oxygen. 

def save_array(Temperature: np.ndarray, Densities: np.ndarray) -> None:
    '''
    Saves model for later use
    '''
    np.save(os.path.join('Part 6/Temperature.npy'), Temperature)
    np.save(os.path.join('Part 6/Densities.npy'), Densities)


def g(h: np.ndarray) -> np.ndarray:
    '''
    h - height in meters\n
    Calculates the gravitational acceleration as a function of height from the planet surface
    '''
    r = h + planet_radius
    a = (G * planet_mass) / r**2
    return a


def profile_atmosphere(h: np.ndarray) -> np.ndarray:
    '''
    h - height in meters \n
    Calculates Temperature T and Density Ρ as a function of height\n
    Assumes isothermal atmosphere for T >= T_surface
    '''
    Temp = np.zeros(len(h))
    Temp[0] = T_surface
    Ρ = np.zeros(len(h)) 
    Ρ[0] = ρ_surface
    dh = abs(h[1] - h[0])
    i = 0
    for i in range(len(h) - 1):
        if Temp[i] >= T_surface/2:  # Atmosphere is assumed to be adiabatic and isothermal
            dT = -(γ - 1)/γ * (μ*m_hydrogen*g(h[i]))/k
            Temp[i+1] = Temp[i] + dT*dh
            dρ = - (Ρ[i]/Temp[i]) * (dT/dh) - (Ρ[i]/Temp[i])*(g(h[i])*μ*m_hydrogen)/k
            Ρ[i+1] = Ρ[i] + dρ*dh
        else:                       # Atmosphere is assumed to be isothermal
            Temp[i+1] = T_surface/2
            dρ = - (Ρ[i]/(T_surface/2)) * (g(h[i])*μ*m_hydrogen)/k
            Ρ[i+1] = Ρ[i] + dρ*dh
    
    save_array(Temp, Ρ)    
    return Temp, Ρ



def plot_atmosphere(show: bool = False) -> None:
    '''
    Plots the temperature as a function of height from 0 to 10 km at 1 m intervals 
    '''
    h = np.arange(0, 200_000, 1)     
    Temperatures, Densities = profile_atmosphere(h)

    plt.plot(h,Temperatures, label="Temperature as a function of height from 0 to 10 km")
    plt.xlabel("Height above surface [m]")
    plt.ylabel("Temperature [K]")
    
    if show:
        plt.show()
    else: 
        plt.savefig(os.path.join('Part 6/figures/Temperature plot.pdf'), format = 'pdf')
        plt.close()
    
    plt.plot(h, Densities, label = "Densities as a function of height from 0 to 10 km")
    plt.xlabel('Height above surface [m]')
    plt.ylabel(r'Atmospheric Density $\left[ \frac{kg}{m^3} \right]$ ')
    
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join('Part 6/figures/Density plot.pdf'), format = 'pdf')
    

if __name__ == '__main__':
    plot_atmosphere()