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
from scipy.ndimage import convolve



from tqdm import tqdm

#####################################################################################################
#####################################################################################################
#####################################################################################################

parser = argparse.ArgumentParser(description="Game of Life simulation.")

# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]


# Command line args are in sys.argv[1], sys.argv[2] ..
# sys.argv[0] is the script name itself and can be ignored
# parse arguments
# example: python3 CP2_v2.py --grid-size 100 --interval 10 --random

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
    return np.random.choice(vals, size=(N,N))
    
    

def glider(i, j, grid):
 
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[OFF, ON, OFF],
                       [OFF, OFF, ON],
                       [ON,  ON,  ON]])
    grid[i:i+3, j:j+3] = glider

    return grid



def oscillator(i, j, grid):
    """adds an oscillator with top left cell at (i, j)"""
    glider = np.array([[OFF, ON,  OFF],
                       [OFF,  ON, OFF],
                       [OFF,  ON, OFF]])
    grid[i:i+3, j:j+3] = glider

    return grid



def selection(frameNum, img, grid, N):
    '''
    Any live cell with less than 2 live neighbours dies.
    Any live cell with 2 or 3 live neighbours lives on to the next step.
    Any live cell with more than 3 live neighbours dies.
    Any dead cell with exactly 3 live neighbours becomes alive.
    '''
    new_grid = grid.copy()
    
    padded_grid = np.pad(grid, pad_width=1, mode='wrap')
    kernel = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])                      # for convolution
    life_array = convolve(padded_grid, kernel, mode='constant') [1:-1, 1:-1]        # array containing sum of neighbours
    
    #new_grid[np.where((new_grid==1) & ((life_array==2) | (life_array == 3)))] = ON
    
    new_grid[np.where((new_grid==1) & (life_array <2))] = 0
    new_grid[np.where((new_grid==1) & (life_array >3))] = 0
    new_grid[np.where((new_grid==0) & (life_array==3))] = 1

    img.set_data(new_grid)
    grid[:] = new_grid[:]
    
    return img



def move(grid, N):
    '''
    Any live cell with less than 2 live neighbours dies.
    Any live cell with 2 or 3 live neighbours lives on to the next step.
    Any live cell with more than 3 live neighbours dies.
    Any dead cell with exactly 3 live neighbours becomes alive.
    '''
    new_grid = grid.copy()
    
    padded_grid = np.pad(grid, pad_width=1, mode='wrap')
    kernel = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])                      # for convolution
    life_array = convolve(padded_grid, kernel, mode='constant') [1:-1,1:-1]        # array containing sum of neighbours
    
    #new_grid[np.where((new_grid==1) & ((life_array==2) | (life_array == 3)))] = ON
    
    new_grid[np.where((new_grid==1) & (life_array <2))] = 0
    new_grid[np.where((new_grid==1) & (life_array >3))] = 0
    new_grid[np.where((new_grid==0) & (life_array==3))] = 1
    
    grid[:] = new_grid[:]
    
    return grid

    

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
        grid = np.zeros((N, N))
        glider(1, 1, grid)

    elif args.oscillator:
        grid = np.zeros((N, N))
        oscillator(int(N/2), int(N/2), grid)
 
    elif args.random:   # populate grid with random on/off
        grid = random_state(N)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, selection, fargs=(img, grid, N, ),
                                  frames = 10,
                                  interval=updateInterval,
                                  save_count=50, blit=True)
 
    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=10, extra_args=['-vcodec', 'libx264'])
 
    plt.show()




def equil():
    
    # set grid size
    N = 100

    # set number of simulations
    N_sim = 100

    # number of steps
    nstep = 10000
        
    t_equil = []
    
    for i in tqdm(range(N_sim)):
        active_sites = []
        init_grid = random_state(N)
        
        for n in (range(nstep)):
            new_grid = move(init_grid, N)
            active_sites.append(np.sum(new_grid))
            
            #if n >= 1000:
            if n>100 and np.all((active_sites[n-100:n] == active_sites[n])) == True:
                t_equil.append(n)
                break
            
    np.savetxt('equiltimes.csv', t_equil, delimiter=',')
                
    return t_equil



def plot_equil():
    t_equil = equil()
    plt.hist(t_equil, 20)
    plt.savefig('equil.png')
    plt.show()



def centre_of_mass():
        
    # number of steps
    if args.N and int(args.N) > 8:
        N = int(args.N)
    else: N = int(input('Lattice size: '))
        
    
    grid = np.zeros((N, N))
    init_grid = glider(1, 1, grid)
    
    nonZeroMasses = np.nonzero(init_grid)
    com = np.array((np.average(nonZeroMasses[0]), np.average(nonZeroMasses[1])))
    
    com_array = []
    com_array.append(com.tolist())

    time = 0
    times = []
    
    dists = []
    
    for i in range(N*2):
        times.append(time)
        time += 1
        new_grid = move(init_grid, N)

        nonZeroMasses_new = np.nonzero(new_grid)
        com = np.array((np.average(nonZeroMasses_new[0]), np.average(nonZeroMasses_new[1])))
        com_array.append(com.tolist())
        dists.append(np.linalg.norm(com))
    
    plt.plot(times, dists/times)
    np.savetxt('speed.csv', dists, delimiter=',')
    plt.xlabel('Time')
    plt.ylabel('CoM distance from origin')
    plt.title('Distance of the CoM from the origin against time')
    plt.show()
    
    
    
# call main
if __name__ == '__main__':
    simulation()
    equil()
    plot_equil()
    centre_of_mass()
