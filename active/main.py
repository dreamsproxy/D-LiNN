import zmq
import numpy as np
import random
import string
import time

import numpy as np
import random
import zmq

class LIFNeuron:
    def __init__(self, id, tau, Vth, Vreset, Vinit, latency):
        self.id = id
        self.tau = tau
        self.Vth = Vth
        self.Vreset = Vreset
        self.V = Vinit
        self.last_spike_time = -np.inf
        self.latency = latency
        self.neighbors = []
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:5555")

    def update(self, I, t):
        resistance = self.latency * 0.0037  # 3.7% loss per 1 ms latency
        dVdt = (-self.V + I) / (self.tau * (1 + resistance))
        self.V += dVdt
        if self.V >= self.Vth:
            self.V = self.Vreset
            self.last_spike_time = t
            for neighbor in self.neighbors:
                self.socket.send_string(f"{self.id}:{neighbor.id}")
            return 1.0
        else:
            return 0.0

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

class LSM:
    def __init__(self, N, M, tau, Vth, Vreset, Vinit, num_neighbors):
        self.N = N
        self.M = M
        self.tau = tau
        self.Vth = Vth
        self.Vreset = Vreset
        self.Vinit = Vinit
        self.num_neighbors = num_neighbors
        self.W = np.random.normal(size=(M, N))
        self.reservoir = np.zeros(N)
        self.neurons = []
        for i in range(N):
            id = f"{i:016X}"
            neuron = LIFNeuron(id, tau, Vth, Vreset, Vinit, 0.0)
            self.neurons.append(neuron)
        for i, neuron in enumerate(self.neurons):
            neighbors = random.sample(self.neurons, num_neighbors)
            neuron.neighbors = neighbors
            for neighbor in neighbors:
                neighbor.add_neighbor(neuron)

    def update(self, I, t):
        spikes = np.zeros(self.N)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(I[i], t)
        self.reservoir = np.dot(self.W, spikes)
        return self.reservoir

def run_lsm():
    tau = 10
    Vth = 1
    Vreset = 0
    Vinit = 0
    num_neighbors = 50

    N = 1000  # Number of neurons
    M = 1000  # Number of outputs
    lsm = LSM(N, M, tau, Vth, Vreset, Vinit, num_neighbors)
    while True:
        t = time.time()
        I = np.random.normal(size=N)
        lsm.update(I, t)
        # Do something with the output, e.g. send it to another process or display it
        time.sleep(0.001)  # Delay to control the firing rate of the neurons

if __name__ == "main":
    run_lsm()