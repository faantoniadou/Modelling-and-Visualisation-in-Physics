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

#N = int(input('Lattice size: '))
#state = str(input('Choose between a random (r) initial condition and one in a set of selected (s) initial conditions'))

# make results reproducible
#np.random.seed(10)


# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]


# Command line args are in sys.argv[1], sys.argv[2] ..
# sys.argv[0] is the script name itself and can be ignored
# parse arguments
# example: python3 MVP_CP2.py --grid-size 100 --interval 10 --random

# add arguments
parser.add_argument('--grid-size', dest='N', required=False)
parser.add_argument('--mov-file', dest='movfile', required=False)
parser.add_argument('--interval', dest='interval', required=False)
parser.add_argument('--glider', action='store_true', required=False)
parser.add_argument('--oscillator', action='store_true', required=False)
parser.add_argument('--random', action='store_true', required=False)

args = parser.parse_args()

#####################################################################################################
#####################################################################################################
#####################################################################################################




def random_state(N):
    '''
    returns a grid of NxN random values
    '''
    return np.random.choice(vals, size=(N,N))#, p=[0.2, 0.8])
    
    

def glider(i, j, grid):
 
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[OFF, ON, OFF],
                       [OFF, OFF, ON],
                       [ON,  ON,  ON]])
    grid[i:i+3, j:j+3] = glider

    #return grid[i:i+3, j:j+3]



def oscillator(i, j, grid):
    """adds an oscillator with top left cell at (i, j)"""
    glider = np.array([[OFF, ON,  OFF],
                       [OFF,  ON, OFF],
                       [OFF,  ON, OFF]])
    grid[i:i+3, j:j+3] = glider

    #return grid[i:i+3, j:j+3]



def selection(frameNum, img, grid, N):
    '''
    Any live cell with less than 2 live neighbours dies.
    Any live cell with 2 or 3 live neighbours lives on to the next step.
    Any live cell with more than 3 live neighbours dies.
    Any dead cell with exactly 3 live neighbours becomes alive.
    '''
    life = 0
    new_grid = grid.copy()
    # loop through the lattice to determine living neighbours 
    for i in range(N):
        for j in range(N):
            # compute 8-neighbor sum
            life = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                        grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                        grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                        grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/ON)
            
            if grid[i,j] == ON:     # if cell is live
                if (life < 2) or (life > 3):
                    new_grid[i,j] = OFF
                
            else:       # if cell is dead
                if life == 3:
                    new_grid[i,j] = ON
    # # update data
    # grid[:] = new_grid[:]
    # return grid
    # update data
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,



def move(grid, N):
    '''
    Any live cell with less than 2 live neighbours dies.
    Any live cell with 2 or 3 live neighbours lives on to the next step.
    Any live cell with more than 3 live neighbours dies.
    Any dead cell with exactly 3 live neighbours becomes alive.
    '''
    life = 0
    new_grid = grid.copy()
    # loop through the lattice to determine living neighbours 
    for i in range(N):
        for j in range(N):
            # compute 8-neighbor sum
            life = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                        grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                        grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                        grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/ON)
            
            if grid[i,j] == ON:     # if cell is live
                if (life < 2) or (life > 3):
                    new_grid[i,j] = OFF
                
            else:       # if cell is dead
                if life == 3:
                    new_grid[i,j] = ON

    grid[:] = new_grid[:]
    return grid,

    

def simulation():

    # set grid size
    N = 100
    if args.N and int(args.N) > 8:
        N = int(args.N)
         
    # set animation update interval
    updateInterval = 10
    if args.interval:
        updateInterval = int(args.interval)
 
    # declare grid
    grid = np.array([])
 
    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N*N).reshape(N, N)
        glider(1, 1, grid)

    elif args.oscillator:
        grid = np.zeros(N*N).reshape(N, N)
        oscillator(int(N/2), int(N/2), grid)
 
    elif args.random:   # populate grid with random on/off -
            # more off than on
        grid = random_state(N)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, selection, fargs=(img, grid, N, ),
                                  frames = 10,
                                  interval=updateInterval,
                                  save_count=50)
 
    # # of frames?
    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=['-vcodec', 'libx264'])
 
    plt.show()




def equil():
    
    # set grid size
    N = 100

    # set number of simulations
    N_sim = 100

    # number of steps
    nstep = 10000

    # set number of 
    
    t_equil = []
    
    for i in range(N_sim):
        active_sites = []
        init_grid = random_state(N)
        
        for n in tqdm(range(nstep)):
            new_grid = move(init_grid, N)
            active_sites.append(np.sum(new_grid))
            
            #if n >= 1000:
            if n>100 and np.all((active_sites[n-100:n] == active_sites[n])) == True:
                t_equil.append(n)
                print(n)
                break
    np.savetxt('equiltimes.csv', t_equil, delimiter=',')
                
    return t_equil

def plot_equil():
    t_equil = equil()
    plt.hist(t_equil, 20)
    plt.savefig('equil.png')
    plt.show()
    


 # call main
if __name__ == '__main__':
    equil()
    plot_equil()





# def state(N):
#     state = str(input('Select between random state ("r"), glider ("g"), and oscillator ("o"): '))

#     if state == 'r':
#         state = random_state(N)
    
#     elif state == 'g':
#         state = glider()
#     elif state == 'o':
    
#     else:
#         print('Please select a valid state')
#         state == state(N)

# def create_state():
#     state = str(input('Choose state ("g" for glider, "r" for random and "o" for oscillator: '))

#     if state == "r":
#         state = random_state()                   
        

# def simulation(method):
#     '''
#     Animated simulation of the Game of Life
    
#     Parameters
#     ----------
#     method : string
#         method to use to move through spins and flip (or not)
    
#     beta: float
#         beta = 1/kT
        
#     Returns
#     -------
#     animated simulation
        
#     '''
#     # declare grid
#     grid = np.array([])
    
    
#     nstep = 10000

#     #initialise spins randomly
#     state = init_state.copy()

#     fig = plt.figure()
#     im = plt.imshow(state, animated=True)

#     for n in tqdm(range(nstep)):
#         for l in range(N**2):
#             indices = random_indx[l + N**2 * n]
#             state = method(indices, beta=beta)
        
#         #occasionally plot or update measurements, eg every 10 sweeps
#         if(n%10==0):
            
#             plt.cla()
#             plt.title(n)
#             im = plt.imshow(state, animated=True)
#             plt.draw()
#             plt.pause(0.0001)


# main() function
