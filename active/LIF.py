import numpy as np
import random

class LIFNeuron:
    def __init__(self, tau, Vth, Vreset, Vinit, latency):
        self.tau = tau
        self.Vth = Vth
        self.Vreset = Vreset
        self.V = Vinit
        self.last_spike_time = -np.inf
        self.latency = latency
        self.neighbors = []

    def update(self, I, t):
        resistance = self.latency * 0.0037  # 3.7% loss per 1 ms latency
        dVdt = (-self.V + I) / (self.tau * (1 + resistance))
        self.V += dVdt
        if self.V >= self.Vth:
            self.V = self.Vreset
            self.last_spike_time = t
            for neighbor in self.neighbors:
                neighbor.receive_spike()
            return 1.0
        else:
            return 0.0

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def receive_spike(self):
        self.V += 0.1 * self.tau

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
        self.neurons = [LIFNeuron(tau, Vth, Vreset, Vinit) for _ in range(N)]
        for i in range(N):
            neighbors = random.sample(self.neurons, num_neighbors)
            self.neurons[i].neighbors = neighbors
            for neighbor in neighbors:
                neighbor.add_neighbor(self.neurons[i])
        
    def update(self, I, t):
        spikes = np.zeros(self.N)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(I[i], t)
        self.reservoir = np.dot(self.W, spikes)
        return self.reservoir
