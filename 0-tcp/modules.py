#from matplotlib import pyplot as plt
from re import I
import numpy as np
import time

def LIF(gl=0.16, Cm=0.0049, mutable_return = None, tick_lim = 100):

    ######### Constants
    El      =   -0.065                      # resting membrane potential [V]
    thresh  =   -0.050                      # spiking threshold [V]

    # VOLTAGE
    V       =   np.array([])     # array for saving Voltage history
    V[0]    =   El                          # set initial to resting potential

    tick    = 0
    dt      = 1
    time       = [i for i in range(tick, tick_lim, 1)]
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
        spike = True
    #time.sleep(1)
    #return
    #print(timestep.shape)
    #print(V.shape)
    #print(I.shape)

    mutable_return.append([timestep, V, I, spike])

    return

"""
def plot_neuron(in_arr):
    plt.plot(in_arr[0], in_arr[1], c = "Blue")
    plt.plot(in_arr[0], in_arr[2], c = "Red")
    plt.xticks(np.arange(0.000, 0.500, 0.05))
    plt.xlim(left = 0, right = in_arr[0][-1])
    plt.ylim(top = 0.1, bottom = -0.09)
    plt.grid(visible = True)
    plt.show()
"""

class client:
    import socket
    import sys

    HOST, PORT = "localhost", 9999
    data = " ".join(sys.argv[1:])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    sock.sendto(bytes(data + "\n", "utf-8"), (HOST, PORT))
    received = str(sock.recv(1024), "utf-8")

    print("Sent:     {}".format(data))
    print("Received: {}".format(received))