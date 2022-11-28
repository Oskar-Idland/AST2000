
import numpy as np
import matplotlib.pyplot as plt
import os
from ast2000tools.star_population import StarPopulation
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as constants
π = np.pi
seed = 36874
system = SolarSystem(seed)


'''Constants for the sun'''
T_sun   = 5800              # [K] Temperature of the sun
L_sun   = constants.L_sun   # Luminosity of the sun 
R_sun   = constants.R_sun   # [m] Radius of the sun 
M_sun   = constants.m_sun   # [kg] mass of the sun


'''Other constants'''
c       = constants.c       # [m/s]Speed of light
σ       = constants.sigma   # Stefan-Boltzmann constant 
k       = constants.k_B     # Boltzmann's constant
G       = constants.G       # Gravitational constant
m_h     = constants.m_p     # [kg] mass of hydrogen atom
m_e     = 9.11*1E-31        # [kg] mass of electron 
μ       = 1.74              # Mean molecular weight of star
μ_core  = 1                 # Mean molecular weight of star core
ħ       = 6.62*1E-34        # Planck's constant
ε_0pp   = 1.08*1E-12   
ε_0CNO  = 8.24*1E-31



def plot_HR_diagram(ticks: list, offset: float = 0, savefig: bool = False):
    '''
    Creates the HR diagram\n
    ticks   - List of markings on the x-axis\n
    offset  - How much to move the lower limit of the x-axis
    '''
    stars = StarPopulation(seed = 36874)
    T = stars.temperatures # [K]
    L = stars.luminosities 
    r = stars.radii        

    c = stars.colors
    s = np.maximum(1e3*(r - r.min())/(r.max() - r.min()), 1.0) # Make point areas proportional to star radii
    fig, ax = plt.subplots()
    ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001)

    ax.set_xlabel('Temperature [K]')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels(list(map(str, ax.get_xticks())))
    ax.set_xlim(40000, 2000 + offset)
    ax.minorticks_off()

    ax.set_ylabel(r'Luminosity [$L/L_\odot$]')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1e6)
    if savefig:
        plt.savefig(f'Part 10/figures/HR_diagram.pdf')
    
    return fig, ax
     

def plot_star(ax, star: list, lbl: str, figname: str, x1: int, x2: int, star_radius: float, savefig: bool = True) -> None:
    '''
    Plots star in HR diagram\n
    star            - The coordinates of the star [Temperature [K], Luminosity [L_sun]]\n
    lbl             - Text of arrow pointing to star\n
    figname         - Name of the figure\n
    x1, x2          - Coordinates where the arrows will point from\n
    star_radius     - [m] Radius of the star for true size plotting
    
    '''
    
    stars = StarPopulation(seed = 36874)
    r = stars.radii
    s_star = (star_radius/R_sun - r.min())/(r.max() - r.min())*1e3

    plt.scatter(star[0], star[1], label = f'{lbl}', color = 'blue', s = s_star) # Plots our star in the diagram
    
    plot_arrow(ax, lbl, star[0], star[1], x1, x2) # Draw the arrow in the diagram
    
    plt.legend(fontsize = 14)
    
    if savefig:
        plt.savefig(os.path.join(f'Part 10/figures/HR_diagram_{figname}.pdf'))
        

def plot_arrow(ax, lbl: str, c1: float, c2: float, x1: float, x2: float) -> None: 
    ax.annotate(lbl,
        xy=(c1, c2), xycoords='data', fontsize = 12,
        xytext=(x1,x2), textcoords='offset points',
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3"),
        )    
        

def Star_L(radius: float, temp: float) -> float: 
    '''
    Calculates luminosity in relation to the sun\n
    radius  - [m]\n
    temp    - [K]
    '''
    return (4 * π*radius**2 * σ * temp**4)/L_sun 

def Star_life(m: float, L: float) -> float:
    '''
    Calculates star life in years \n 
    m - [kg] mass\n
    L - [L_⊙] Luminosity
    '''
    return (.1*.007*m*c**2)/(L*L_sun) * 1/(3600*24*365)
        

def T_M_ratio(T: float, M: float) -> float:
    '''
    Calculates a proportionality factor between star surface temperature and mass\n
    M - [kg]\n
    T - [K]
    '''
    factor = M/T**2
    return factor

def M_L_ratio(L: float, M: float) -> float:
    '''
    Calculate a proportionality factor between star mass and luminosity\n
    M - [kg]\n
    L - [W]
    '''
    factor = M**(4)/L
    return factor

def R_j(Mass: float, T: float = 10) -> float:
    '''
    Calculates the Jeans length of a gas cloud\n
    Mass -  [kg]
    T    -  [K] 
    '''
    return (3*G*μ*m_h*Mass)/(15*k*T)

def T_c(T_surface: float, ρ_star: float, star_radius: float, μ: float)-> float: 
    '''
    Calculates core temperature of star\n
    
    T_surface       - [K] Surface temperature of star\n
    ρ_star          - [kg/m^3] Density of star assumed to be uniform\n
    star_radius     - [m]\n
    μ               - Mean molecular weight of star
    '''
    return T_surface + (2*π*G*ρ_star*μ*m_h*star_radius**2)/(3*k)

def ε_pp(T: float, ρ: float)-> float: 
    '''
    Calculates energy released from pp chain\n
    T - [K] Temperature in million
    ρ - [kg/m^2] Density
    '''
    return ε_0pp * 0.745**2 * ρ * T**4

def ε_CNO(T: float, ρ: float)-> float: 
    '''
    Calculates energy released from CNO chain\n
    T - [K] Temperature in million\n
    ρ - [kg/m^3] Density of star
    '''
    return ε_0CNO * 0.745*0.002*ρ*T**20


def Dwarf_radius(m: float)-> float: 
    '''
    Calculates the radius of a star in its dwarf stage\n
    m - [kg] Mass
    '''
    return (3/(2*π))**(4/3) * ħ**2/(20*m_e*G) * (1/(2*m_h))**(5/3) * m**(-1/3)

def Dwarf_mass(m: float)-> float: 
    '''
    Calculates the mass of a star in its dwarf stage\n
    m - [kg] Mass
    '''
    return m/8 * 1.4

def Star_ρ(m: float, r: float)-> float: 
    '''
    Calculates star density int its dwarf stage\n
    m - [kg] mass\n
    r - [m] radius
    '''
    return m/(4/3 * π * r**3)

def G_acc(M: float, radius: float)-> float: 
    '''
    Calculates gravitational pull of star\n 
    M       - [kg] mass\n 
    radius  - [m]
    '''
    return G*M/radius**2


if __name__ == "__main__":
    
    '''Star constants'''
    star_radius = system.star_radius*1E3                    # [m] 
    star_temp   = system.star_temperature                   # [K] 
    star_mass   = system.star_mass                          # [M_⊙] Star mass
    star_ρ      = Star_ρ(star_mass*M_sun, star_radius)      # [kg] Star density
    star_L      = Star_L(star_radius, star_temp)            # [L_̇⊙] The luminosity of our star in relation to the sun 
    t_life      = Star_life(star_mass*M_sun, star_L)        # [yr] Life time of our star
    star_temp_c = T_c(star_temp, star_ρ, star_radius, 1)    # [K] Core temp of star
    
    '''
    Printing facts about our star and its proportionality in relation to the sun
    '''
    print(f'Star mass:                  {star_mass: .1f} [Solar masses]')
    print(f'Star radius:                {star_radius: .2e} [m]')
    print(f'Star surface temperature:   {star_temp: .0f} [K]')
    print(f'Star luminosity:            {star_L: .2e} [Solar Luminosity]')
    print(f'Star luminosity:            {star_L*L_sun: .2e} W')
    print(f'Star lifetime is:           {t_life: .2e} [yr]')
    print()
    
    '''
    Printing facts about our star as a giant molecular cloud
    '''
    R_cloud = R_j(star_mass*M_sun)
    T_cloud = 10
    L_cloud = Star_L(R_cloud, T_cloud) 
    
    print(f'Min radius of GMC:  {R_cloud: .1e} [m]')
    print(f'Luminosity of GMC:  {L_cloud: .1e} [Solar Luminosities]')
    print()
    
    '''Printing facts about our star's nuclear reactions '''
    E_pp    = ε_pp(star_temp_c*1E-6, star_ρ)
    E_CNO   = ε_CNO(star_temp_c*1E-6, star_ρ)
    
    print(f'Star core temperature:                      {star_temp_c: .2e} [K]')
    print(f'Energy released from pp chain:              {E_pp: .2e} [W/kg]')
    print(f'Energy released form CNO chain:             {E_CNO: .2e} [W/kg]')
    print(f'Calculated luminosity form core reactions   {star_ρ*4/3*π *(E_pp + E_CNO)*0.2*star_radius**3: .1e} W')
    print()
    
    '''Printing factors derived from constants from the star'''
    print('ML factor')
    print(f'\tStar: {M_L_ratio(star_L*L_sun, star_mass*M_sun): .1e}')
    print(f'\tSun:  {M_L_ratio(L_sun, M_sun): .1e}')
    print('TM factor')
    print(f'\tStar: {T_M_ratio(star_temp, star_mass*M_sun): .1e}')
    print(f'\tSun:  {T_M_ratio(T_sun, M_sun): .1e}')
    print()
    
    
    '''Printing facts about our star as a white dwarf'''
    D_mass      = Dwarf_mass(star_mass*M_sun)
    D_radius    = Dwarf_radius(D_mass)
    D_ρ         = Star_ρ(D_mass,D_radius)
    D_g         = G_acc(D_mass, D_radius)
    
    print(f'Star mass when dwarf:               {D_mass: .2e} [kg] / {D_mass/M_sun: .2e} [Solar masses]') 
    print(f'Star radius when dwarf:             {D_radius: .2e} [m]')
    print(f'Star density when dwarf:            {D_ρ: .2e} [kg/m^3]')
    print(f'Weight of 1L dwarf star material    {D_ρ/1000: .2e} [kg]')
    print(f'Star gravitational pull when dwarf: {D_g: .2e} [m/s^2]')
    
    
    '''Plotting the sun in HR diagram'''
    # TODO: Slett kommentar før innlevering
    sun_HR_coords = [T_sun, 1]
    lbl = 'The Sun'
    figname = 'The Sun'
    fig, ax = plot_HR_diagram([35000, 18000, 10000, 6000, 4000, 3000])
    plot_star(ax, sun_HR_coords, lbl, figname, -100, -50, R_sun, savefig = True)
    
    '''Plotting our star together with the sun in its current state in the HR-diagram'''
    # TODO: Slett kommentar før innlevering
    sun_HR_coords = [T_sun, 1]
    lbl = 'The Sun'
    figname = 'Sun and Star'
    fig, ax = plot_HR_diagram([35000, 18000, 10000, 6000, 4000, 3000])
    plot_star(ax, sun_HR_coords, lbl, figname, -100, -50, R_sun, savefig = False)
    
    star_HR_coords = [star_temp, star_L]
    lbl = 'Our Star Today'
    figname = 'Sun and Star'
    plot_star(ax, star_HR_coords, lbl, figname, -100, -50, star_radius, savefig = True)
    

    
    '''Plotting our star when it was a gas cloud in the HR-diagram'''
    # TODO: Slett kommentar før innlevering
    lbl = 'Our Star Pre-Collapse'
    figname = 'Pre-Collapse'
    cloud_HR_coords = [T_cloud, L_cloud]
    fig, ax = plot_HR_diagram([35000, 3000, 10], offset = -1999)
    plot_star(ax, cloud_HR_coords, lbl, figname, -10, -50, R_cloud/1E4, savefig = True)
    
    '''Plotting arrows on the HR-diagram'''
    # TODO: Slett kommentar før innlevering
    fig, ax = plot_HR_diagram([35000, 18000, 10000, 6000, 4000, 3000])
    dwarf_lbl   = 'White Dwarfs'
    main_lbl    = 'Main Sequence'
    giant_lbl   = 'Giants'
    Sgiant_lbl  = 'Super Giants'
    S_groups    = [dwarf_lbl, main_lbl, giant_lbl, Sgiant_lbl]    
    
    dwarf_coords    = [18000, 1E-2]
    main_coords     = [9000, 1E1]
    giant_coords    = [3500, 1E2]
    Sgiant_coords   = [5000, 1E4]
    S_coords        = [dwarf_coords, main_coords, giant_coords, Sgiant_coords]
    
    dwarf_x     = [-75, -25]
    main_x      = [-75, -50]
    giant_x     = [25, -25]
    Sgiant_x    = [-50, -25]
    S_x         = [dwarf_x, main_x, giant_x, Sgiant_x]
    
    for i in range(len(S_groups)):
        c1, c2 = S_coords[i][0], S_coords[i][1]
        x1, x2 = S_x[i][0], S_x[i][1]
        lbl = S_groups[i]
        plot_arrow(ax, lbl, c1, c2, x1, x2)


    plt.savefig(os.path.join('Part 10/figures/HR-Diagram_Classification.pdf'))
    
    '''
    Output
    ------------------------------------------------------------------------------
    Star mass:                   4.4 [Solar masses]
    Star radius:                 2.07e+09 [m]
    Star surface temperature:    11733 [K]
    Star luminosity:             1.51e+02 [Solar Luminosity]
    Star luminosity:             5.79e+28 W
    Star lifetime is:            3.04e+08 [yr]

    Min radius of GMC:   2.5e+15 [m]
    Luminosity of GMC:   1.1e+02 [Solar Luminosities]

    Star core temperature:                       1.72e+07 [K]
    Energy released from pp chain:               1.26e-05 [W/kg]
    Energy released form CNO chain:              1.58e-06 [W/kg]
    Calculated luminosity form core reactions    2.5e+25 W

    ML factor
        Star:  1.0e+95
        Sun:   4.1e+94
    TM factor
        Star:  6.4e+22
        Sun:   5.9e+22

    Star mass when dwarf:                1.55e+30 [kg] /  7.77e-01 [Solar masses]
    Star radius when dwarf:              1.55e+06 [m]
    Star density when dwarf:             9.82e+10 [kg/m^3]
    Weight of 1L dwarf star material     9.82e+07 [kg]
    Star gravitational pull when dwarf:  4.27e+07 [m/s^2]
    ------------------------------------------------------------------------------
    '''