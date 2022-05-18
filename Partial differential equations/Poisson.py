#%%
import matplotlib
matplotlib.use('TKAgg')
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

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



def create_checkerboard(shape):         # works
        mask =  np.bool_(np.indices(shape).sum(axis=0) % 2)
        return mask, np.invert(mask)            # return white and black spaces arrays 



def jacobi(potential, rho, active_region, npad):      # works

    init_potential = np.copy(potential)
    
    potential = 1/6 * (np.roll( init_potential, 1, axis=0 ) + np.roll( init_potential, -1, axis=0 ) + 
                       np.roll( init_potential, 1, axis=1 ) + np.roll( init_potential, -1, axis=1 ) +
                       np.roll( init_potential, 1, axis=2 ) + np.roll( init_potential, -1, axis=2 ) + rho)
    
    # enforce boundary
    final_potential = np.pad(potential[active_region], npad)
    error = np.sum(np.absolute(final_potential - init_potential))

    return final_potential, error



def gauss_seidel(potential, rho, active_region, npad, white, black):
    
    init_potential = np.copy(potential)
    
    # first update all white
    potential = 1/6 * (np.roll( potential, 1, axis=0 ) + np.roll( potential, -1, axis=0 ) + 
                       np.roll( potential, 1, axis=1 ) + np.roll( potential, -1, axis=1 ) + 
                       np.roll( potential, 1, axis=2 ) + np.roll( potential, -1, axis=2 ) + rho)
    # reverse black updates
    potential[black] = init_potential[black]
    potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
    updated_whites = potential.copy()
    
    # then update the black
    potential = 1/6 * (np.roll(potential, 1, axis=0) + np.roll(potential, -1, axis=0) + 
                       np.roll(potential, 1, axis=1) + np.roll(potential, -1, axis=1) +
                       np.roll(potential, 1, axis=2) + np.roll(potential, -1, axis=2) + rho)
    # reverse white updates
    potential[white] = updated_whites[white]
    final_potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
    
    error = np.sum(np.abs(final_potential - init_potential))
    
    return final_potential, error



def sor(potential, rho, active_region, npad, white, black, omega=1.89):
    
    init_potential = np.copy(potential)
    
    potential = 1/6 * (np.roll(potential, 1, axis=0) + np.roll(potential, -1, axis=0) + 
                       np.roll(potential, 1, axis=1) + np.roll(potential, -1, axis=1) +
                       np.roll(potential, 1, axis=2) + np.roll(potential, -1, axis=2) + rho)
    # over relax
    potential *= omega
    potential += init_potential * (1 - omega) 
    
    # reverse black updates
    potential[black] = init_potential[black]
    
    # enforce boundary
    potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
    updated_whites = np.copy(potential)
    
    potential = 1/6 * (np.roll(potential, 1, axis=0) + np.roll(potential, -1, axis=0) + 
                       np.roll(potential, 1, axis=1) + np.roll(potential, -1, axis=1) +
                       np.roll(potential, 1, axis=2) + np.roll(potential, -1, axis=2) + rho)
    # overrelax
    potential *= omega
    potential += init_potential * (1 - omega)
    
    # reverse white updates
    potential[white] = updated_whites[white]
    
    # enforce boundary
    final_potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
    error = np.sum(np.abs(final_potential - init_potential))
    
    return final_potential, error



def field_calc(N, potential, s, dx):
    # calculate the field
   
    field_x = (np.roll(potential, 1, axis=0) - np.roll(potential, -1, axis=0))[s, s, N//2].ravel()/(-2 * dx)
    field_y = (np.roll(potential, 1, axis=1) - np.roll(potential, -1, axis=1))[s, s, N//2].ravel()/(-2 * dx)
    field_z = (np.roll(potential, 1, axis=2) - np.roll(potential, -1, axis=2))[s, s, N//2].ravel()/(-2 * dx)
    
    return field_x, field_y, field_z



def save_data(N, potential, s, dx, field):
    pos_arr = np.array(np.meshgrid(np.arange(1, N+1), np.arange(1, N+1))).T.reshape(-1, 2)
    dis_arr = np.linalg.norm(pos_arr - N//2, axis=1)
    pot_arr = potential[1:N+1, 1:N+1, N//2].ravel()
    
    plt.figure()
    plt.scatter(dis_arr, pot_arr, marker='+', s=10, c='purple')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Distance")
    plt.ylabel("Potential")
    plt.savefig(f"PotentialVSDistance_N={N}_{field}.png")
    plt.yscale('linear')
    
    if field == "magnetic":
        plt.clf()
        M_y, M_x, _ = field_calc(N, potential, s, dx)
        
        # more efficient to do 2d just do padding in x and y
        mag_B = np.sqrt(M_x**2 + M_y**2)
        np.savetxt(f"{field}_{N}_output.csv", np.column_stack((pos_arr, dis_arr, pot_arr, M_x, - M_y)) )
        plt.yscale('linear')
        plt.quiver(pos_arr.T[0], pos_arr.T[1], M_x/mag_B, - M_y/mag_B, angles='xy', scale_units='xy', scale=1)
        plt.title("Magnetic Field")
        plt.savefig('mag.png')
        
        plt.clf()
        plt.scatter(dis_arr, np.sqrt(M_x**2 + M_y**2), marker='+', s=17, c='purple')
        plt.ylabel("Magnetic field strength")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Distance")
        plt.savefig("mag_arr.png")
        #plt.show()
        
    elif field == "electric":
        plt.clf()
        E_x, E_y, E_z = -1 * np.array(field_calc(N, potential, s, dx))
        mag_E = np.sqrt(E_x**2 + E_y**2 + E_z**2)
        np.savetxt(f"{field}_{N}_output.csv", np.column_stack((pos_arr, dis_arr, pot_arr, E_x, E_y, E_z)) )
        plt.yscale('linear')
        plt.quiver(pos_arr.T[0], pos_arr.T[1], E_x/mag_E, E_y/mag_E,  angles='xy', scale_units='xy', scale=1)
        plt.title("Electric Field")
        plt.savefig('mag.png')
        #plt.show()
        
        plt.clf()
        plt.scatter(dis_arr, np.sqrt((E_x)**2 + (E_y)**2 + (E_z)**2), marker='+', s=17, c='purple')
        plt.ylabel("Electric field strength")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Distance")
        plt.savefig("el_arr.png")
        #plt.show()
        
        
    np.savetxt(f"{field}_2D_output.csv", potential[0:N+2, 0:N+2, N//2] )
    


def plot_slice(N, potential, rho, mode, nstep):
    plt.clf()
    plt.figure()
    plt.imshow(potential[0:(N+2), 0:(N+2), N//2], interpolation='gaussian', cmap='gnuplot')
    plt.colorbar()
    plt.savefig(f"N={N}_mode={mode}_nstep={nstep}.png")
    # plt.show()
    

 
def make_plots(N, s, dx, potential, rho, reset, mode, active_region, npad, white, black, omega, nstep=250, tol=1e-3, field="electric"):
    
    if mode == 'jacobian':
        # jacobi
        for sweep in tqdm(range(nstep)):
            potential, error = jacobi(potential, rho, active_region, npad)
            if np.isclose(error, 0, atol=tol):
                break 
            

        plot_slice(N, potential, rho, mode, nstep)
        

    elif mode == "gaussian":
        # gauss-seidel
        for sweep in tqdm(range(nstep)):
            potential, error = gauss_seidel(potential, rho, active_region, npad, white, black)
            if np.isclose(error, 0, atol=tol):
                break 
        save_data(N, potential, s, dx, field=field)
        plot_slice(N, potential, rho, mode, nstep)
        #plt.show()
        

    elif mode == "sor":
        # relaxation
        omega_arr = np.linspace(1, 2, 20, endpoint=False)
        data = np.zeros((20, 2))
        sweep_array = []
        
        for index in tqdm(range(20)):
            potential = np.copy(reset)
            for sweep in range(nstep):
                potential, error = sor(potential, rho, active_region, npad, white, black, omega_arr[index])
                if np.isclose(error, 0, atol=tol):
                    data[index] = omega_arr[index], sweep
                    sweep_array.append(sweep)
                    break 
        plt.plot(omega_arr, sweep_array)
        plt.ylabel("Sweep")
        plt.xlabel("Omega")
        #plt.show()
        plt.clf()
        np.savetxt("output_sor.csv", data)
        
        
def choose_params(N, nstep = int(250), field ="magnetic"):
    # choose parameters according to the mode and field
    shape = np.array((N+2, N+2, N+2))
    potential, reset, rho = np.zeros((3, *shape))
    black, white = create_checkerboard(shape)
    s = slice(1, N+1)
    
    if field =="electric":
        rho[N//2, N//2, N//2] = 1         # create point charge
        npad = ((1, 1), (1, 1), (1, 1))
        active_region = (s, s, s)
        
    elif field == "magnetic":
        rho[ N//2, N//2] = 1
        npad = ((1, 1), (1, 1), (0, 0))
        active_region = (s, s,)
        
    return potential, reset, rho, npad, active_region, black, white, shape, s

#%%        

def main():
    N = int(input("System size : "))
    dx = 1.
    u = 1.
    omega_opt = 1.89
    mode = str(input("Mode (gaussian, jacobian or sor): "))
    field = str(input("Field (magnetic or electric): "))
    nstep = int(8001) #int(input("Number of sweeps: "))
    
    potential_main, reset, rho, npad, active_region, black, white, shape, s = choose_params(N, nstep=nstep, field=field)
    tol = 1e-2

    make_plots(N, s, dx, potential_main, rho, reset, mode, active_region, npad, white, black, omega=omega_opt, nstep=nstep, tol=1e-3, field=field)

    
main()

# def simulation():
#     N = 50
#     dx = 1.
#     u = 1.
#     omega_opt = 1.89
#     mode = "gaussian"
#     field = "electric"
#     nstep = 25000
    
#     potential, reset, rho, npad, active_region, black, white, shape, s = choose_params(N, nstep=nstep, field=field)
    
#     tol = 1e-2
    

#     plt.figure()
#     plt.imshow(potential[:,:,N//2], animated=True, cmap='gnuplot') 
#     plt.colorbar()
#     free_energies = np.empty(int(nstep/100)+1)

#     for i, step in tqdm(enumerate(range(nstep))):
#         potential = gauss_seidel(potential, rho, active_region, npad, white, black)[0]
        
#         if i%10 == 0:

#             plt.cla()   
#             plt.title(step)
#             im = plt.imshow(potential[:,:,N//2], animated=True, interpolation='gaussian', cmap='gnuplot')
#             plt.draw()
#             plt.pause(0.0001)
            

