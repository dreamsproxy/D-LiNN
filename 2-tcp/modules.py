import numpy as np
import time

def LIF(tick = 0, gl=0.16, Cm=0.0049, mutable_return = None, tick_lim = 100):

    ######### Constants
    El      =   -0.065                      # resting membrane potential [V]
    thresh  =   -0.050                      # spiking threshold [V]

    # VOLTAGE
    V       =   np.array([])     # array for saving Voltage history
    V[0]    =   El                          # set initial to resting potential

    tick    = 0
    dt      = 1
    time    = [i for i in range(tick, tick_lim, 1)]
    I       = 0

    ######### Simulation
    I[tick]
    # use "I - V/R = C * dV/dT" to get this equation
    dV =  (I[tick] - gl*(V[tick-1]-El))/Cm
    V[tick] = V[tick-1] + dV*dt
    # in case we exceed threshold
    if V[tick] > thresh:
        V[tick-1] = 0.04   # set the last step to spike value
        V[tick] = El       # current step is resting membrane potential
        #spikes += 1     # count spike
    #time.sleep(1)
    #return
    #print(timestep.shape)
    #print(V.shape)
    #print(I.shape)

    mutable_return.append([tick, V, I])

    return