import os
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as constants
from numba import njit as TURBOOOO
from multiprocessing import Pool
c = constants.c
k = constants.k_B
u = 1.6605402E-27




def convert_data():
    '''
    !!! Has to be run the first time program is used to convert text files to npy files !!!
    '''
    with open('Part 6\Flux Data\sigma_noise.txt', 'r') as f:
        data = []
        for line in f.readlines():
            data.append([float(line.split()[0]), float(line.split()[1])])

    np.save('Part 6\Flux Data\sigma_noise', data)

    with open('Part 6\Flux Data\Flux.txt', 'r') as f:
        data = []
        for line in f.readlines():
            data.append([float(line.split()[0]), float(line.split()[1])])

    np.save('Part 6\Flux Data\Flux', data)
    

# convert_data()
flux_data = np.load('Part 6\Flux Data\Flux.npy')
wavelengths = flux_data[:, 0]
Φ = flux_data[:, 1]
noise = np.load('Part 6\Flux Data\sigma_noise.npy')[:, 1]



@TURBOOOO
def σ(λ_0: float, m: float, T: float) -> np.ndarray:
    '''
    Standard deviation of a spectral line's Gaussian line profile
    '''
    return λ_0 / c * np.sqrt((k*T)/m)


@TURBOOOO
def GLD(f_min: float, λ_0: float, λ: float, σ) -> np.ndarray:
    '''
    Gaussian Line Distribution
    '''
    return 1 + (f_min - 1)*np.exp(-.5*((λ - λ_0)/σ)**2)


@TURBOOOO(fastmath=True)
def χ_Squared(flux: np.ndarray, data: np.ndarray, GLD, σ, λ_0: float, m: float, N = 50, tol=1e-11):
    '''
    Finds the best parameters of the gas in the atmosphere based on known values of spectral lines\n

    data    - Array of data point\n
    f       - Function to evaluate the data\n
    σ       - Function to evaluate the standard deviation\n
    λ_0     - Central spectral line of gas in question\n
    m       - Mass of a molecule of the gas in question\n

    Returns: Flux and temperature, corresponding to the lowest χ squared and the slice of data used together with the lowest χ squared.  
    '''

    # Doppler shift with max velocity of 10 000 m/s and max temp of 450 k
    Δλ = λ_0*(10_000 + ((2*k*450)/m)**.5)/c
    # Range of possible temperatures on planet
    temps = np.linspace(150, 450, N)
    # Range of minimum flux at center of spectral line
    f_mins = np.linspace(0.7, 1, N)
    # Range of wavelengths adjusted for doppler shift
    wavelengths = np.linspace(λ_0 - Δλ, λ_0 + Δλ, N)
    lowest_value = 1e20
    index1 = np.where(np.abs(data - (λ_0 - Δλ)) < tol)[0][0]
    index2 = np.where(np.abs(data - (λ_0 + Δλ)) < tol)[0][-1]
    data_slice = data[index1:index2]
    flux_slice = flux[index1:index2]
    noise_slice = noise[index1:index2]
    for T in temps:
        σ_t = σ(λ_0, m, T)
        for f_min in f_mins:
            for λ in wavelengths:
                computed = GLD(f_min, data_slice, λ, σ_t)
                result = np.sum( ((flux_slice - computed) / noise_slice)**2 )
                if result < lowest_value:
                    lowest_value    = result
                    f_min_lowest    = f_min
                    temp_lowest     = T
                    computed_lowest = computed
                    λ_lowest        = λ
                    
    return f_min_lowest, temp_lowest, data_slice, flux_slice, noise_slice, computed_lowest, Δλ*1E9, λ_lowest*1E9


def plot(data_slice, flux_slice, noise_slice, computed, name, λ_lowest, λ_0):
    plt.plot(data_slice*1e9, flux_slice, label = 'Measured flux')
    plt.plot(data_slice*1e9,
             computed, label=f'λ_0 = {λ_0}\nΔλ = {abs(λ_lowest - λ_0): .3f} ')
    plt.plot(data_slice*1e9, noise_slice, label = 'Noise')
    plt.title(f'Gaussian line profile of {name}')
    plt.xlabel(r'λ (nm)')
    plt.ylabel('Flux')
    plt.legend(loc = 'lower left', fontsize = 16)


    
def plot_gas(gas, show = False):
    for λ_0 in gas['Spectral Lines']:
        data_slice  =   gas[f'result {λ_0}']['data_slice']
        flux_slice  =   gas[f'result {λ_0}']['flux_slice']
        noise_slice =   gas[f'result {λ_0}']['noise_slice']
        computed    =   gas[f'result {λ_0}']['computed']
        λ_lowest    =   gas[f'result {λ_0}']['λ_lowest']
        name        =   gas['name']
        
        plt.figure(figsize=(20,8))
        plot(data_slice, flux_slice, noise_slice+.95, computed, name, λ_lowest, λ_0)
        if show:
            plt.show()
            
        else:
            plt.savefig(os.path.join(f'Part 6\Figures\{name} {λ_0}.pdf'), format='pdf')
            plt.close()


def plot_gases():
    for gas in gases:
        plot_gas(gas)
        
        
def analyse_gas(gas, N = 50, table = True):
    if table:
        print('Gas|Wavelength (nm)|Min. Flux (W/m^3)|T (K)| Δλ')
        print('---|---------------|-----------------|-----|-------')
    m = gas['mass']
    name = gas['name']
    for λ_0 in gas['Spectral Lines']:
        flux, temp, data_slice, flux_slice, noise_slice, computed, Δλ, λ_lowest = χ_Squared(
            Φ, wavelengths/1E9, GLD, σ, λ_0/1E9, m, N)
        print(f'{name:^4}{λ_0:^16.0f}{flux:^18.2f}{temp:^6.1f}{abs(λ_lowest-λ_0):^8.2e}')
        gas[f'result {λ_0}'] = {'flux': flux, 'temp': temp, 'data_slice': data_slice,
                                'flux_slice': flux_slice, 'noise_slice': noise_slice, 'computed': computed, 'λ_lowest': λ_lowest, 'Δλ': Δλ}


def analyse_gases(N = 50):
    print('Gas|Wavelength (nm)|Min. Flux (W/m^3)|T (K)| Δλ')
    print('---|---------------|-----------------|-----|-------')
    for gas in gases:
        analyse_gas(gas, N, table = False)


# Gases and their properties stored in dictionaries
O2  = {'name': 'O2',   'Spectral Lines': [632, 690, 760], 'mass': (2*15.9994)*u}
H2O = {'name': 'H2O',  'Spectral Lines': [720, 820, 940], 'mass': (2*1.00794 + 15.9994)*u}
CO2 = {'name': 'CO2',  'Spectral Lines': [1400, 1600],    'mass': (12.0107 + 2*15.9994)*u}
CH4 = {'name': 'CH4',  'Spectral Lines': [1660, 2200],    'mass': (12.0107 + 4*1.00794)*u}
CO  = {'name': 'CO',   'Spectral Lines': [2340],          'mass': (12.0107 + 15.9994)*u}
N2O = {'name': 'N2O',  'Spectral Lines': [2870],          'mass': (2*14.0067 + 15.9994)*u}

gases = [O2, H2O, CO2, CH4, CO, N2O]


if __name__ == '__main__':
    analyse_gases(N = 200)
    plot_gases()