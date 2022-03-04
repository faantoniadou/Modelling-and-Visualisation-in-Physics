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


from tqdm import tqdm

N = int(input('Lattice size: '))
#state = str(input('Choose between a random (r) initial condition and one in a set of selected (s) initial conditions'))

# make results reproducible
np.random.seed(10)

class GameOfLife:
    def __init__ (self):
        self.input = str(input('Choose between a random (r) initial condition and one in a set of selected (s) initial conditions'))

    def glider(self):
        
    
    # def beehive(self):
        
    def blinker(self):
    
    # def empty(self):
        
    def gen_state(self):
    
    def random_state(self):
        init_state = np.random.choice([0, 1],size=(N, N))
        
    def selection(self):
        '''
        Any live cell with less than 2 live neighbours dies.
        Any live cell with 2 or 3 live neighbours lives on to the next step.
        Any live cell with more than 3 live neighbours dies.
        Any dead cell with exactly 3 live neighbours becomes alive.
        '''
        life = 0
        
        # loop through the lattice to determine living neighbours 
        for i in range(N):
            for j in range(N):
                life = state[(i+1)%N, j] + state[i, (j+1)%N] + state[(i-1)%N, j] + state[i, (j-1)%N] + \
                    state[(i+1)%N, (j+1)%N] + state[(i-1)%N, (j+1)%N] + state[(i-1)%N, (j-1)%N] + state[(i-1)%N, (j-1)%N] 
                
                if state[i,j] == 1:     # if cell is live
                    if life < 2:
                        state[i,j] *= 0
                    # elif (life == 3 or life == 2):
                    #     continue
                    elif life > 3:
                        state[i,j] *= 0
                else:       # if cell is dead
                    if life == 3:
                        state[i,j] += 1
                    
            
            
                        