import matplotlib
matplotlib.use('TKAgg')
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from numpy.random import rand
import random
from numpy import savetxt

import scipy
from numpy import array
from matplotlib import colors


from tqdm import tqdm



# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]

N = 50 #int(input('Array size: '))



def initial_state():
    i, j  = random.choices(np.arange(N), k=2)
    grid = np.zeros((N, N))
    grid[i, j] = ON
    
    return grid



def selection(grid, p):
    '''
    ON  --> OFF with probability 1 - p
    ON --> infects one of its four nearest neighbours with probability p
    
    the neighbour to be infected is chosen randomly
    if the chosen neighbour is already infected nothing happens
    
    '''
    
    r = rand()
    i, j  = random.choices(np.arange(N), k=2)          # choose a random cell

    neighbours = [[(i-1)%N, j], [(i+1)%N, j], [i, (j+1)%N], [i, (j-1)%N]]
    
    # choose random neighbour
    k, l = neighbours[random.choice(np.arange(len(neighbours)))]
    
    if grid[i ,j] == ON:
        if (1 - p) > r:
            grid[i, j] = OFF
        
        else:    
            grid[k, l] = ON
        
    return grid
    

def plot_survival(p):
    probs, time = np.genfromtxt(f'survival={p}.csv')
    plt.plot(time, probs)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    
    
def survival_prob():
    
    nstep = 300
    p = float(input("p: "))
    exps = 100
    bool_inf = np.zeros(nstep)
    
    for exp in tqdm(range(exps)):
        #initialise spins randomly
        state = initial_state()
    
        for n in tqdm(range(nstep)):     
            
            for i in range(N**2):
                state = selection(state, p)
            
            if np.any(state==ON):
                bool_inf[n] += 1
                
        #all_inf[exp] =  np.array(no_infected)
        # all_inf.append(np.array(no_infected))
        
    # mean_inf = np.mean(bool_inf/exps, axis=0)
    probs = bool_inf/exps
    data = probs, np.arange(nstep)
    np.savetxt(f"survival={p}.csv", data)
    
    plot_survival(p)



# call main
if __name__ == '__main__':
    # move(random_state(50), 50)
    # simulation()
    survival_prob()
    # plot_survival(0.65)
    # many_probs()
    
    
    
    
    
# def simulation():

#     '''
#     Animated simulation of the SIRS model
        
#     Returns
#     -------
#     animated simulation
        
#     '''
    
#     nstep = 300
#     p = float(input("p: "))

#     #initialise spins randomly
#     state = initial_state()
#     # make a color map of fixed colors
#     cmap = colors.ListedColormap(['white', 'red'])
#     bounds=[0,0.5,1]
#     norm = colors.BoundaryNorm(bounds, cmap.N)

#     plt.figure()
#     plt.imshow(state, animated=True, interpolation='nearest', origin='lower',
#                     cmap=cmap, norm=norm) 
#     plt.colorbar()
    
#     no_infected = [] 
#     times = []
#     for n in tqdm(range(nstep)):    
#         for i in range(N**2):
#             state = selection(state, p)
            
#         # occasionally plot or update measurements, eg every 10 sweeps
#         if(n%1==0):
#             plt.cla()
#             plt.title(n)
#             im = plt.imshow(state, animated=True, interpolation='nearest', origin='lower',
#                     cmap=cmap, norm=norm)
#             plt.draw()
#             plt.pause(0.0001)
        
#         if n%1==0:
#             times.append(n)
#             no_infected.append(np.count_nonzero(state==ON))
    
#     data = no_infected, times
#     np.savetxt(f"no_survival_VS_time_p={p}.csv", data)
    
#     plt.clf()
#     plt.plot(times, no_infected)
#     plt.savefig(f"p={p}.png")
#     plt.show()
    
    