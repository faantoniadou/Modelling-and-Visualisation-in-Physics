
# CP1: The Ising Model
This program simulates the Ising model using different dynamics rules.
There are 2 steps to the simulation.
The first step is the animation of the simulation at a specific temperature, followed by 
plotting graphs of Energy, susceptibility, magnetisation and heat capacity for a range of temperatures.

## User input:
Upon running the program, the user is prompted to input the size of the system, temperature at which
the animated simulation will run as well as the dynamic rule (Glauber or Kawasaki).
The user will be prompted to choose the dynamic rule again in order to plot a graph of the parameters above.
The plot will use a range of temperatures which is fixed (1-3 in steps of 0.1 units).

## Code structure:
Functions:
* `total_energy`:

    Calculates total energy of a state by summing product of all spin pairs
    
    Parameters
    ----------
    state : array|state of which the total energy will be calculated
        
    Returns
    -------
    E: float | total energy of the given state

* `metropolis`:
    Monte Carlo move using Metropolis algorithm and Glauber dynamics

    
    Parameters
    ----------
    indices : array|index of the target spin to flip (or not)
    beta: float|beta = 1/kT
        
    Returns
    -------
    init_state: array|modified state to be fed into the simulation (either with spin flipped or not)

* `kawasaki`:
    Monte Carlo move using Kawasaki algorithm
    
    Parameters
    ----------
    indices : array|index of the target spin to flip (or not)
    beta: float|beta = 1/kT
        
    Returns
    -------
    init_state: array|modified state to be fed into the simulation (either with spin flipped or not)
    
* `simulation`:
    Animated simulation of the Ising model

    
    Parameters
    ----------
    method : string|method to use to move through spins and flip (or not)
    beta: float|beta = 1/kT
        
    Returns
    -------
    animated simulation

* `run_sim`:
    Runs simulation above according to method input

* `equil_metr`:
    Returns 1000 equilibrated states at a specific temperature
    
    Parameters
    ----------
    method : string|method to use to move through spins and flip (or not)
        
    Returns
    -------
    kT_array, M_avg, susc, E_array, H_capacity, M_errs, susc_errs, E_errs, H_errs : arrays| desired quantities to plot
    
* `plot_quant`:
    Plots desired quantities according to the method used.
