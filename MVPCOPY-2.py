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


# pseudo-random numbers to use in future functions
N = int(input('Array size: '))

# make results reproducible
np.random.seed(10)

# initialise random state
init_state = np.random.choice([-1, 1],size=(N, N))

# create array of pseudo-random indices to use in the simulation. This is a faster solution than using randint in the loop
random_indx = np.random.randint(0, N, size=(255000*(N**2), 4))


def total_energy(state):
    '''
    Calculates total energy of a state by summing product of all spin pairs
    
    Parameters
    ----------
    state : array
        state of which the total energy will be calculated
        
    Returns
    -------
    E: float
        total energy of the given state

    '''
    
    E = 0
    
    # loop through spins without repetition
    for i in range(N):
        for j in range(i+1, N):
           E += -(state[(i+1)%N, j] + state[i, (j+1)%N] + state[(i-1)%N, j] + state[i, (j-1)%N]) * state[i,j]
    return E



def metropolis(indices,beta=1):
    '''
    Monte Carlo move using Metropolis algorithm and Glauber dynamics

    
    Parameters
    ----------
    indices : array
        index of the target spin to flip (or not)
    
    beta: float
        beta = 1/kT
        
    Returns
    -------
    init_state: array
        modified state to be fed into the simulation (either with spin flipped or not)
        
    '''
    
    i, j = indices[0:2]     # we only require one target spin
    
    delta_E = (init_state[(i+1)%N, j] + init_state[i,(j+1)%N] + init_state[(i-1)%N, j] + \
        init_state[i,(j-1)%N]) * 2 * init_state[i,j]
    
    #only 2 possible values for np.exp(sth) 8 or 4
    if delta_E <= 0:
        init_state[i,j] *= -1

    elif rand() < np.exp(-delta_E*beta):
        init_state[i, j] *= -1
       
    return init_state


def kawasaki(indices, beta=1):
    '''
    Monte Carlo move using Kawasaki algorithm


    Parameters
    ----------
    indices : array
        index of the target spin to flip (or not)
    
    beta: float
        beta = 1/kT
        
    Returns
    -------
    init_state: array
        modified state to be fed into the simulation (either with spin flipped or not)
        
    '''
    
    i, j, k, l = indices        # we require 2 target spins
    S1 = init_state[i, j] 
    S2 = init_state[k, l]
    
    
    if S1 != S2:    # if spins are equal then we do nothing but if not, continue
        # change in energy in terms of the initial state
        delta_E = 2 * S1 * (init_state[(i+1)%N, j] + init_state[i, (j+1)%N] + init_state[(i-1)%N, j] + init_state[i, (j-1)%N]) + \
            2 * S2 * (init_state[(k+1)%N, l] + init_state[k, (l+1)%N] + init_state[(k-1)%N, l] + init_state[k, (l-1)%N])
             
        if np.linalg.norm(np.array((i-k , j-l)))%N == 1:        # if randomly chosen spins are nearest neighbours
            delta_E += 4            # factor of 4 is due to double-counting and nearest neighbour effect
    
        # swap spins under conditions
        if delta_E <= 0:
            init_state[i][j] *= -1
            init_state[k][l] *= -1

        elif rand() < np.exp(-delta_E*beta):
            init_state[i][j] *= -1
            init_state[k][l] *= -1

    return init_state


def simulation(method, beta=1):
    '''
    Animated simulation of the Ising model

    
    Parameters
    ----------
    method : string
        method to use to move through spins and flip (or not)
    
    beta: float
        beta = 1/kT
        
    Returns
    -------
    animated simulation
        
    '''
    
    # set constants
    J = 1.0
    beta = float(input('Temperature: '))
    kT = 1/beta
    
    nstep = 10000

    #initialise spins randomly
    state = init_state.copy()

    fig = plt.figure()
    im = plt.imshow(state, animated=True)

    for n in tqdm(range(nstep)):
        for l in range(N**2):
            indices = random_indx[l + N**2 * n]
            state = method(indices, beta=beta)
        
        #occasionally plot or update measurements, eg every 10 sweeps
        if(n%10==0):
            
            plt.cla()
            plt.title(n)
            im = plt.imshow(state, animated=True)
            plt.draw()
            plt.pause(0.0001)


def run_sim():
    '''
    Runs simulation above according to method input
    '''
    
    method = str(input("Dynamical rule ('g' for Glauber or 'k' for Kawasaki): "))
    
    if method=='k':
        simulation(kawasaki)
    elif method =='g':
        simulation(metropolis)
    else:
        print("Invalid method. Please try again")
        

#run_sim()

def equil_metr(method):
    '''
    Returns 1000 equilibrated states at a specific temperature
    
    Parameters
    ----------
    method : string
        method to use to move through spins and flip (or not)
        
    Returns
    -------
    kT_array, M_avg, susc, E_array, H_capacity, M_errs, susc_errs, E_errs, H_errs : arrays
        desired quantities to plot
        
    '''
    
    J=1.0
    nsteps = 10201

    # initialise spins randomly
    state = init_state.copy()
    
    M_avg = []
    susc = []
    E_array = []
    E_errs = []
    susc_errs = []
    M_errs = []
    H_capacity = []
    H_errs = []
    
    no_particles = N**2
    
    for T in tqdm(np.arange(1,3.1,0.1)):    # loop through a range of temperatures
        M_calc = []
        M_sq = []
        E_vals = []
        
        for n in tqdm(range(nsteps)):
            for l in range(N**2):
                indices = random_indx[l + N**2 * n]
                state = method(indices, beta=1/T)

            if n%10==0 and n>=200:    
                sum_state = np.sum(state)
                M_calc.append(sum_state)
                M_sq.append(sum_state**2)
                E_vals.append(total_energy(state))
              
        m_avg = np.average(M_calc)
        e_avg = np.average(E_vals)
        
        M_avg.append(m_avg)
        susc.append(1/(T*no_particles) * (np.var(M_calc)))
        E_array.append(e_avg)
        H_capacity.append(1/(T * no_particles) * (np.var(E_vals)))
        
        M_errs.append(np.std(M_calc)/np.sqrt(len(M_calc)))
        E_errs.append(np.std(E_vals)/np.sqrt(len(E_vals)))
        H_errs.append(np.std(np.square(np.array(E_vals) - e_avg)/(no_particles * T**2))/len(E_vals))
        susc_errs.append(np.std(np.square(np.array(M_calc) - m_avg)/(N**2 *T))/len(M_calc))

    return M_avg, susc, E_array, H_capacity, M_errs, susc_errs, E_errs, H_errs



def plot_quant():
    '''
    Plots desired quantities according to the method used.
    '''
    
    method = str(input("Dynamical rule ('g' for Glauber or 'k' for Kawasaki): "))
    
    if method=='k':
        M_avg, susc, E_array, H_capacity, M_errs, susc_errs, E_errs, H_errs = equil_metr(kawasaki)
    elif method =='m':
        M_avg, susc, E_array, H_capacity, M_errs, susc_errs, E_errs, H_errs = equil_metr(metropolis)
    else:
        print("Invalid method. Please try again")
    
    kT_array = np.arange(1, 3.1, 0.1)
    
    # save to csv files
    np.savetxt('magnetisation.csv', M_avg, delimiter=',')
    np.savetxt('suscepibility.csv', susc, delimiter=',')
    np.savetxt('total_energy.csv', E_array, delimiter=',')
    np.savetxt('heat_capacity.csv', H_capacity, delimiter=',')
    
    plt.subplot(2, 2, 1)
    plt.plot(kT_array, M_avg)
    plt.errorbar(kT_array, M_avg, yerr= M_errs, fmt='x', ecolor = 'lightblue',color='m')
    plt.ylabel("Magnetisation")
    plt.xlabel("kT")
    
    plt.subplot(2, 2, 2)
    plt.plot(kT_array, susc)
    plt.ylabel("Susceptibility")
    plt.errorbar(kT_array, susc, yerr= susc_errs, fmt='x', ecolor = 'lightblue',color='m')
    plt.xlabel("kT")
    
    plt.subplot(2, 2, 3)
    plt.plot(kT_array, E_array)
    plt.ylabel("Total energy")
    plt.errorbar(kT_array, E_array, yerr= E_errs, fmt='x', ecolor = 'lightblue',color='m')
    plt.xlabel("kT")
    
    plt.subplot(2, 2, 4)
    plt.plot(kT_array, H_capacity)
    plt.ylabel("Heat Capacity")
    plt.errorbar(kT_array, H_capacity, yerr= H_errs, fmt='x', ecolor = 'lightblue',color='m')
    plt.xlabel("kT")
    
    plt.autoscale(enable=True, axis='both', tight=False)
    plt.show()

plot_quant()
