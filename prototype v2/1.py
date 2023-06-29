
from neuron import AdaptiveLIF
from neuron import plot_neuron
import numpy as np
from time import process_time
import random
#from plot_tools import plot_neuron
"""
tau_m = 20      # Membrane time constant (ms)
R = 1           # Membrane resistance (Mohm)
V_th = 1        # Threshold potential (mV)
V_reset = 0     # Reset potential (mV)
delta_t = 0.1   # Time step (ms)
t_sim = 1000    # Simulation time (ms)
tau_leak = 10   # Membrane leak time constant (ms)
"""

class resivour:
    def createCube(x, y, z):
        one_dim_shape = (x * y * z, 1)
        size = (x*y*z)
        print(f"Generation of {size} neurons begun...")
        
        tau_m_array = np.random.uniform(20.0, 50.0, one_dim_shape)
        tau_leak_array = np.random.uniform(10.0, 100.0, one_dim_shape)
        threshold_array = np.random.uniform(2.5, 6.0, one_dim_shape)
        reset_array = np.zeros(one_dim_shape)
        resistance_array = np.random.uniform(1.0, 5.0, one_dim_shape)
        neuron_IDS = np.arange(0.0, np.float32(size), 1.0).reshape(one_dim_shape)
        
        print(resistance_array.shape)
        print(neuron_IDS.shape)
        prep_array = np.hstack((
            neuron_IDS,
            tau_m_array,
            tau_leak_array,
            threshold_array,
            reset_array,
            resistance_array))

        # reshape the entire stacked array back to it's 3D form:
        # (5, 5, 5, 5)
        three_dim_shape = (x, y, z, 6)
        prep_array = np.reshape(prep_array, three_dim_shape)
        
        return prep_array
    

x = 2
y = 2
z = 2
generation_start = process_time()
cube_resivour = resivour.createCube(x, y, z)
generation_end = process_time()
#print(generation_end)
total_gen = generation_end - generation_start
print(f"Generation of {x*y*z} neurons took {total_gen*1000} ms!")

#print(cube_resivour[0])
#print(cube_resivour[0].shape)
#print(cube_resivour)
#print(cube_resivour.shape)
sim_time = 10 #miliseconds

sim_start = process_time()
for xpos in range(x):
    #print(xpos)
    for ypos in range(y):
        for zpos in range(z):
            neuron_start = process_time()
            ID, tau_m, tau_leak, thresh, reset, resistance = cube_resivour[xpos, ypos, zpos]
            neuron = AdaptiveLIF(tau_m, resistance, thresh, reset, 1.0, tau_leak)
            t, V, spikes = neuron.simulate(
                input_current = random.uniform(10., 50.),
                sim_ms=sim_time)
    #print(V)
    #raise

sim_end = process_time()
sim_total = sim_end - sim_start
print()
print("Done!")
print(f"Simulation of {x*y*z} neurons took {sim_total - sim_time//1000} seconds !")
