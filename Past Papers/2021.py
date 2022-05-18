import numpy as np
import matplotlib.pyplot as plt
from os import system
import sys
import matplotlib
matplotlib.use('TKAgg')
from tqdm import tqdm 
from matplotlib import animation

np.random.seed(10)

a = 0.1
c = 0.1
k = 0.1
M = 0.1
D = 1.

phi0 = 0.5#float(input("phi0: "))
phi_dag = phi0
chi = 0.3#float(input("chi: "))
alpha = float(input("alpha: "))
m0 = 0.

choice = None #str(input("Do you want to animate phi or m? "))

dt = 0.2
dx = 1.


def physical_sys(N):
    # some random noise
    phi = np.random.uniform(-0.1 + phi0, 0.1 + phi0, (N, N) )
    m = np.random.uniform(-0.1 + m0, 0.1 + m0, (N, N) )
    
    return phi, m


def update(phi, m):
    # method to update the grid

    mu = - a * phi + a * np.power(phi, 3) - np.multiply(chi/2, np.power(m, 2)) - \
        (k / dx**2) * ( np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) + 
                        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - np.multiply(phi, 4) )

    m += dt* (D * (1/dx **2) * ( np.roll(m, 1, axis=0) + np.roll(m, -1, axis=0) + 
                              np.roll(m, 1, axis=1) + np.roll(m, -1, axis=1) - np.multiply(m, 4) ) - \
                                  (np.multiply((c - chi * phi), m) + np.multiply(c, np.power(m, 3))))
    
    phi += M * (dt/dx **2) * (np.roll(mu, 1, axis=0) + np.roll(mu, -1, axis=0) + \
                               np.roll(mu, 1, axis=1) + np.roll(mu, -1, axis=1) - np.multiply(mu,4) )
    
    return (phi, m)



def update2(phi, m):
    # method to update the grid
    mu = - a * phi + a * np.power(phi, 3) - np.multiply(chi/2, np.power(m, 2)) - \
        (k / dx**2) * ( np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) + 
                        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - np.multiply(phi, 4) )
        
        
    m += dt* (D * (1/dx **2) * ( np.roll(m, 1, axis=0) + np.roll(m, -1, axis=0) + 
                              np.roll(m, 1, axis=1) + np.roll(m, -1, axis=1) - np.multiply(m, 4) ) - \
                                  (np.multiply((c - chi * phi), m) + np.multiply(c, np.power(m, 3))))


    phi += M * (dt/dx **2) * (np.roll(mu, 1, axis=0) + np.roll(mu, -1, axis=0) + \
                               np.roll(mu, 1, axis=1) + np.roll(mu, -1, axis=1) - np.multiply(mu,4) )\
                                   - dt * (alpha * (phi - phi_dag))
    
    return (phi, m)


def choose_param():
    global choice
    choice = str(input("Do you want to animate phi or m? "))
    
    idx = 0
    
    if choice == 'phi':
        idx = 0
    elif choice == 'm':
        idx = 1
    else:
        print("Error! Choose a valid parameter")
        choose_param()
        
    return idx
        

def simulation(animate=False):
    idx = int(2)
    N = 50
    phi, m = physical_sys(N)
    grids = phi, m
    nstep = int(1e+5)+1
    
    avg_phi = [] 
    avg_phi.append(np.nanmean(phi))
    
    avg_m = []
    avg_m.append(np.nanmean(m))
    
    time = []
    time.append(0)
    
    if animate==True:
        idx = choose_param()
        plt.figure()
        plt.imshow(grids[idx], animated=True, interpolation='gaussian', cmap='gnuplot') 
        plt.colorbar()

    for i, step in tqdm(enumerate(range(nstep))):
        phi, m = update(phi, m)
        grids = phi, m
        
        if i%10:
            avg_phi.append(np.nanmean(phi))
            avg_m.append(np.nanmean(m))
            time.append(i)
        
        if i%1000 == 0 and animate==True:
            plt.title(step)
            im = plt.imshow(grids[idx], animated=True, interpolation='gaussian', cmap='gnuplot') 
            plt.draw()
            plt.pause(0.0001)
            plt.savefig(f"{choice}/{choice}_phi0={phi0}_chi={chi}_snap{i}.png")
    
    data = time, avg_phi, avg_m
    np.savetxt(f"avg_data_phi0={phi0}_chi={chi}.csv", np.c_[data])
    
    plt.clf()
    plt.plot(time, avg_phi, label=r"$\phi$", c='b')
    plt.plot(time, avg_m, label="m", c='r')
    plt.legend()
    plt.xlabel('Time')
    plt.savefig(f'Averages_phi0={phi0}_chi={chi}.png')
    plt.show()
    


def simulation2(animate=False):
    idx = int(2)
    N = 50
    phi, m = physical_sys(N)
    grids = phi, m
    nstep = int(1e+5)+1
    
    avg_phi = [] 
    avg_phi.append(np.nanmean(phi))
    
    avg_m = []
    avg_m.append(np.nanmean(m))
    
    time = []
    time.append(0)
    
    if animate==True:
        idx = choose_param()
        plt.figure()
        plt.imshow(grids[idx], animated=True, interpolation='gaussian', cmap='gnuplot') 
        plt.colorbar()

    for i, step in tqdm(enumerate(range(nstep))):
        phi, m = update2(phi, m)
        grids = phi, m
        
        if i%10:
            avg_phi.append(np.nanmean(phi))
            avg_m.append(np.nanmean(m))
            time.append(i)
        
        if i%1000 == 0 and animate==True:
            plt.title(step)
            im = plt.imshow(grids[idx], animated=True, interpolation='gaussian', cmap='gnuplot') 
            plt.draw()
            plt.pause(0.0001)

    

            



# call main
if __name__ == '__main__':
    simulation(animate=True)
    # simulation2(animate=True)
    # simulation(animate=False)
