import numpy as np
import matplotlib.pyplot as plt
from os import system
import sys
import matplotlib
matplotlib.use('TKAgg')
from tqdm import tqdm 
from matplotlib import animation


a = 0.1
M = 0.1
k = 0.1
phi0 = float(input("Mobility (phi0): "))

dt = 2.         # if this is too small the dynamics is very slow. can go up to 2
dx = 1.


def physical_sys(N):
    # some random noise
    grid = np.random.uniform(-0.1 + phi0, 0.1 + phi0, (N, N) )
    
    return grid


def update(grid):
    # method to update the grid
    mu = -a * grid + a * np.power(grid, 3) - (k / dx**2) * ( np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + 
                                                     np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - np.multiply(grid, 4) )
    
    grid = M * (dt/dx **2) * (np.roll(mu, 1, axis=0) + np.roll(mu, -1, axis=0) + 
                               np.roll(mu, 1, axis=1) + np.roll(mu, -1, axis=1) - np.multiply(mu,4) )
    
    return grid



def free_energy_density(grid):
    """Calculates free density

    Args:
        grid (array): initial array

    Returns:
        _array_: _new system_
    """
    # free energy calculation
    return np.sum( -(a/2) * np.power(grid, 2) + (a/4) * np.power(grid, 4) + 
                  ( 0.5 * k/dx**2 ) * (np.power(np.roll(grid, 1, axis=0) - grid, 2) + #central
                                       np.power(np.roll(grid, 1, axis=1) - grid, 2)) )
                

def simulation(animate=False):

    N = int(input("System size : "))
    grid = physical_sys(N)
    nstep = int(1e+6)+1
    

    plt.figure()
    plt.imshow(grid, vmax=1, vmin=-1, animated=True, cmap='coolwarm') 
    plt.colorbar()
    free_energies = np.empty(int(nstep/100)+1)

    for i, step in tqdm(enumerate(range(nstep))):
        grid += update(grid)
        
        if i%500 == 0 and animate==True:

            plt.cla()   
            plt.title(step)
            im = plt.imshow(grid, vmax=1, vmin=-1, animated=True, interpolation='gaussian', cmap='coolwarm')
            plt.draw()
            plt.pause(0.0001)
            
        if i%100 == 0:
            free_energies[int(i/100)] = free_energy_density(grid)
            
    np.savetxt(f'free_energies_phi0={phi0}.csv', free_energies, delimiter=',')
    

def plot_energy():
    plt.clf()
    x = np.arange(0,int(1e+6)+100,100)
    energies = np.genfromtxt(f'free_energies_phi0={phi0}.csv')

    plt.xlabel('Time step')
    plt.ylabel('Free Energy Density')
    plt.title('Free energy density over time')

    # plt.xlim(0,np.max(x))
    plt.plot(x, energies)
    plt.savefig(f'phi0={phi0}.png')
    plt.show()



# call main
if __name__ == '__main__':
    simulation(animate=True)
    # plot_energy()
