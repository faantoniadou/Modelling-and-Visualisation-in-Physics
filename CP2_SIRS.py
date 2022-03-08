#%%
import matplotlib
matplotlib.use('TKAgg')
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from sklearn.utils import resample
from numpy.random import rand
import random
from numpy import savetxt
import argparse

from tqdm import tqdm

#####################################################################################################
#####################################################################################################
#####################################################################################################

parser = argparse.ArgumentParser(description="Game of Life simulation.")

#state = str(input('Choose between a random (r) initial condition and one in a set of selected (s) initial conditions'))

# make results reproducible
#np.random.seed(10)


# setting up the values for the grid
S = 1
I = 2
R = 0
vals = [S, I, R]

#%%
# Command line args are in sys.argv[1], sys.argv[2] ..
# sys.argv[0] is the script name itself and can be ignored
# parse arguments
# example: python3 MVP_CP2.py --grid-size 100 --interval 10 --random

# add arguments
parser.add_argument('--sys-size', dest='N', required=False)
parser.add_argument('--mov-file', dest='movfile', required=False)
parser.add_argument('--interval', dest='interval', required=False)
parser.add_argument('--p1', action='store_true', required=False)
parser.add_argument('--p2', action='store_true', required=False)
parser.add_argument('--p3', action='store_true', required=False)

args = parser.parse_args()

#####################################################################################################
#####################################################################################################
#####################################################################################################
N = 50



def random_state(N):
    '''
    returns a grid of NxN random values
    '''
    return np.random.choice(vals, size=(N, N))
    
    

def selection(p1, p2, p3):#frameNum, img, N, p1, p2, p3):
    '''
    S--> I with probability p1 if at least one neighbout is I (i.e., if sum of neighbours is greater than 8)
    I--> R with probability p2
    R--> S with probability p3
    '''
    life = 0
    grid = random_state(N)
    new_grid = grid.copy()
    
    print(new_grid[np.where(new_grid == I)])
    
    S_points = 
    I_points = 
    R_points = 
    
    (new_grid[np.where(new_grid == I)])[np.random.rand(*new_grid.shape) < p2] = R
    (new_grid[np.where(new_grid == R)])[np.random.rand(*new_grid.shape) < p1] = S
    
    
    # img.set_data(new_grid)
    # grid[:] = new_grid[:]
    # return img,

selection(0.5, 0.4, 0.1)

# def move(grid, N):
#     '''
#     S--> I with probability p1 if at least one neighbout is I (i.e., if sum of neighbours is greater than 8)
#     I--> R with probability p2
#     R--> S with probability p3
#     '''
#     life = 0
#     new_grid = grid.copy()
#     # loop through the lattice to determine living neighbours 
#     for i in range(N):
#         for j in range(N):
#             # compute 8-neighbor sum
#             life = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
#                         grid[(i-1)%N, j] + grid[(i+1)%N, j] +
#                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
#                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/ON)
            
#             if grid[i,j] == ON:     # if cell is live
#                 if (life < 2) or (life > 3):
#                     new_grid[i,j] = OFF
                
#             else:       # if cell is dead
#                 if life == 3:
#                     new_grid[i,j] = ON

#     grid[:] = new_grid[:]
#     return grid,

    

# def simulation():

#     # set grid size
#     N = 100
#     if args.N and int(args.N) > 8:
#         N = int(args.N)
         
#     # set animation update interval
#     updateInterval = 10
#     if args.interval:
#         updateInterval = int(args.interval)
 
#     # declare grid
#     grid = np.array([])
 
#     # check if "glider" demo flag is specified
#     if args.glider:
#         grid = np.zeros(N*N).reshape(N, N)
#         glider(1, 1, grid)

#     elif args.oscillator:
#         grid = np.zeros(N*N).reshape(N, N)
#         oscillator(int(N/2), int(N/2), grid)
 
#     elif args.random:   # populate grid with random on/off -
#             # more off than on
#         grid = random_state(N)

#     # set up animation
#     fig, ax = plt.subplots()
#     img = ax.imshow(grid, interpolation='nearest')
#     ani = animation.FuncAnimation(fig, selection, fargs=(img, grid, N, ),
#                                   frames = 10,
#                                   interval=updateInterval,
#                                   save_count=50)
 
#     # # of frames?
#     # set output file
#     if args.movfile:
#         ani.save(args.movfile, fps=30, extra_args=['-vcodec', 'libx264'])
 
#     plt.show()




# def equil():
    
#     # set grid size
#     N = 100

#     # set number of simulations
#     N_sim = 100

#     # number of steps
#     nstep = 10000

#     # set number of 
    
#     t_equil = []
    
#     for i in tqdm(range(N_sim)):
#         active_sites = []
#         init_grid = random_state(N)
        
#         for n in range(nstep):
#             new_grid = move(init_grid, N)
#             active_sites.append(np.sum(new_grid))
            
#             #if n >= 1000:
#             if n>100 and np.all((active_sites[n-50:n] == active_sites[n])) == True:
#                 t_equil.append(n)
#                 break
#     np.savetxt('equiltimes.csv', t_equil, delimiter=',')
                
#     return t_equil

# def plot_equil():
#     t_equil = equil()
#     np.savetxt('equiltimes.csv', t_equil, delimiter=',')
#     plt.hist(t_equil, 40, density=True)
#     plt.xlabel('Times needed to equilibrate')
#     plt.ylabel('Normalised Frequency')
#     plt.savefig('equil.png')
#     plt.show()
    
# #%%
# def centre_of_mass():
#     # set grid size
#     N = 100

#     # number of steps
#     nstep = 100

#     grid = np.zeros((N,N))
#     init_grid = glider(1, 1, grid)
#     #print(init_grid)
    
#     nonZeroMasses = np.nonzero(init_grid)
#     com = np.array((np.average(nonZeroMasses[0]), np.average(nonZeroMasses[1])))
#     com_array = []
#     com_array.append(com.tolist())
    
#     time = 0
#     times = []
#     times.append(time)
    
#     for i in range(N*2):
#         time += 1
#         times.append(0)
#         new_grid = move(init_grid, N)
#         nonZeroMasses_new = np.nonzero(new_grid)[1:3]
        
#         #print(nonZeroMasses_new)
#         com = np.array((np.average(nonZeroMasses_new[0]), np.average(nonZeroMasses_new[1])))
#         com_array.append(com.tolist())
#         #delta_com = (com_array[i-1]%N - com_array[i]%N)
#     print(np.array(com_array))
    
    

#%%

#  # call main
# if __name__ == '__main__':
#     # simulation()
#     # equil()
#     # plot_equil()
