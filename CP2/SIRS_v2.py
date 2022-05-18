import matplotlib
matplotlib.use('TKAgg')
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from numpy.random import rand
import random
from numpy import savetxt

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from numpy import array

from tqdm import tqdm

#####################################################################################################
#####################################################################################################
#####################################################################################################


# setting up the values for the grid
S = 1
I = 2
R = 0
vals = [S, I, R]

#(i) an absorbing state with all sites susceptible to the infection; 
#       ---> 0.5, 0.6, 0.1
#(ii) a dynamic equilibrium between S, I and R;
#       ---> 0.5, 0.5, 0.5
#(iii) a case with cyclic wave of infections through the lattice
#       ---> 0.8, 0.1, 0.01

# pseudo-random numbers to use in future functions
N = int(input('Array size: '))

#####################################################################################################
#####################################################################################################
#####################################################################################################


def random_state():
    '''
    returns a grid of NxN random values
    '''
    return np.random.choice(vals, size=(N,N))
    
    

def selection(grid, p1, p2, p3):
    '''
    S--> I with probability p1 if at least one neighbout is I
    I--> R with probability p2
    R--> S with probability p3
    '''
    
    r = rand()
    
    i, j = random.choices(np.arange(N), k=2)          # choose 1 random cell
    
    times_updated = 0
    
    if grid[i ,j] == S:
        # if life_array[i,j] >= I :
        if (grid[(i+1)%N,j] == I or grid[(i-1)%N,j] == I or grid[i,(j-1)%N] == I or grid[i,(j+1)%N] == I):
            if p1 > r:
                grid[i, j] = I
    elif grid[i ,j] == I:
        if p2 > r:
                grid[i, j] = R
    elif grid[i ,j] == R:
        if p3 > r:
                grid[i, j] = S

    return grid




    

def simulation():

    '''
    Animated simulation of the SIRS model
        
    Returns
    -------
    animated simulation
        
    '''
    
    behaviour = str(input("Choose behaviour: \n 'A' : an absorbing state with all sites susceptible to the infection; \n 'D' : a dynamic equilibrium between S, I and R; \n 'C' : a case with cyclic wave of infections through the lattice \n"))
    
    if behaviour == 'A':
        p1, p2, p3 = 0.5, 0.6, 0.1
        
    elif behaviour == 'D':
        p1, p2, p3 = 0.5, 0.5, 0.5
        
    elif behaviour == 'C':
        p1, p2, p3 = 0.8, 0.1, 0.01
    
    else:
        print("Invalid behaviour chosen. Please run again")
    
    nstep = 1000

    #initialise spins randomly
    state = random_state()

    fig = plt.figure()
    im = plt.imshow(state, animated=True)
    
    for n in tqdm(range(nstep)):
        for i in range(N**2):
            state = selection(state, p1, p2, p3)
            
        #occasionally plot or update measurements, eg every 10 sweeps
        #if(n%10==0):
        plt.cla()
        plt.title(n)
        im = plt.imshow(state, animated=True)
        plt.draw()
        plt.pause(0.0001)



def phase(start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps):
    
    nstep = sweeps + 101
    p1_array = np.linspace(start_p1, end_p1, no_points1)
    p3_array = np.linspace(start_p3, end_p3, no_points3)
    
    probs = [(p1, p3) for p1 in p1_array for p3 in p3_array]
    print(len(probs))
    #initialise spins randomly
        
    avg_infected = []
    I_var = []
    I_errs = []
    
    for comb in tqdm(range(len(probs))):
        p1, p3 = probs[comb]
        state = random_state()
        no_infected = []
        
        for n in tqdm(range(nstep)):
            
            for i in range(N**2):
                state = selection(state, p1, 0.5, p3)
                
            if n > 100:
                # count number of infected sites
                no_infected.append(np.count_nonzero(state==I))

        # estimate bootstrap errors for the case where we have one of p1 or p3 fixed
        if (len(p1_array) == 1 or len(p3_array) == 1):
            resampled_calc = []
            for i in range(200):
                resampled_I = np.random.choice(no_infected, 800)
                new_calc_I = np.var(resampled_I)
                resampled_calc.append(new_calc_I)
            
            I_errs.append(np.var(resampled_calc)**(1/2))
            
        avg_infected.append(np.mean(no_infected))
        I_var.append(np.var(no_infected))
    
    avg_infected = np.array(avg_infected)/N**2

    print(len(I_errs))
    
    np.savetxt(f'infected_{N}.csv', avg_infected, delimiter=',')
    np.savetxt(f'variance_{N}.csv', np.array(I_var), delimiter=',')
    np.savetxt(f'errors_{N}.csv', np.array(I_errs), delimiter=',')
        
    return avg_infected, np.array(I_var), p1_array, p3_array, np.array(I_errs)


def plot_inf():
    start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps = 0., 0., 1., 1., 21, 21, 1000
    
    avg_infected, I_var, p1_array, p3_array, _ = phase(start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps)
    xx, yy = np.meshgrid(p1_array, p3_array)

    fig = plt.figure()
    ax1 = plt.contourf(xx,yy,avg_infected.reshape((len(p1_array),len(p3_array))))
    
    plt.savefig(f'phase_plot_{N}.png')
    plt.show()



def plot_var():
    start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps = 0., 0., 1., 1., 21, 21, 1000
    
    avg_infected, I_var, p1_array, p3_array, _ = phase(start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps)
    xx, yy = np.meshgrid(p1_array, p3_array)

    fig = plt.figure()
    ax1 = plt.contourf(xx,yy,I_var.reshape((len(p1_array),len(p3_array))))
    
    plt.savefig(f'var_plot_{N}.png')
    plt.show()



def trimmed_plot():
    start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps = 0.2, 0.5, 0.5, 0.5, 21, 1, 10000
    
    avg_infected, I_var, p1_array, p3_array, I_errs = phase(start_p1, start_p3, end_p1, end_p3, no_points1, no_points3, sweeps)
    plt.plot(p1_array, I_var)
    plt.errorbar(p1_array, I_var, yerr=I_errs, fmt='x', ecolor = 'lightblue',color='m')
    plt.xlabel('p1')
    plt.ylabel('Variance of the number of infected sites')
    plt.savefig(f'trimmed_var_plot_{N}.png')
    plt.show()
    
    
def fIm_plot(sweeps):
    # modify our program so that some fraction of agents are permanently immune to the infection (i.e., in the R state)
    # loop over fIm array
    fIm_array = np.arange(0., 1.1, 0.1)
    
    avg_infected = []
    I_errs = []
        
    for frac in tqdm(fIm_array):
        avg_infected_single = []

        for i in tqdm(range(5)):
            # choose random state only with S and I and then turn fraction of them to unaffected 
            state = random_state()
            indices_i = np.random.randint(0, state.shape[0], int(state.size*frac))
            indices_j = np.random.randint(0, state.shape[0], int(state.size*frac))
            state[indices_i, indices_j] = 3.

            nstep = sweeps + 101
                    
            no_infected = []
            
            for n in tqdm(range(nstep)):
                
                for i in range(N**2):
                    state = selection(state, 0.5, 0.5, 0.5)
                    
                if n > 100:
                    # count number of infected sites
                    no_infected.append(np.count_nonzero(state==I))
            avg_infected_single.append(np.mean(no_infected))
        
        # estimate errors using the standard error for the mean
        I_errs.append(np.var(no_infected)/np.sqrt(5))
        avg_infected.append(np.mean(avg_infected_single))
        
    avg_infected = np.array(avg_infected)/N**2

    
    np.savetxt(f'infected_{N}_fIm.csv', avg_infected, delimiter=',')
    np.savetxt(f'fIm_array{N}_fIm.csv', fIm_array, delimiter=',')
    np.savetxt(f'errors_{N}_fIm.csv', np.array(I_errs), delimiter=',')

    plt.plot(fIm_array, avg_infected)
    plt.errorbar(fIm_array, avg_infected, yerr=I_errs, fmt='x', ecolor = 'lightblue',color='m')
    plt.xlabel('fIm')
    plt.ylabel('Average number of infected sites')
    plt.show()
    
    


# call main
if __name__ == '__main__':
    # simulation()
    # phase()
    # plot_inf()
    # plot_var()
    # trimmed_plot()
    fIm_plot(10000)
    
