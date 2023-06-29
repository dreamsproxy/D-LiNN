import time
import threading
import numpy as np
from p2pnetwork.node import Node

class AdExLIFNeuron(Node):
    def __init__(self, name, neighbors, ip, port,):
        super().__init__(ip, port, name)
        self.neighbors = neighbors
        self.current = 0.0
        self.voltage = 0.0
        self.threshold = -50.0
        self.reset_voltage = -60.0
        self.refractory_period = 0.0
        self.time_step = 0.1
        self.decay = 0.1

    def run(self):
        while True:
            if self.voltage >= self.threshold:
                self.send_currents()
                self.voltage = self.reset_voltage
                self.refractory_period = 20.0

            if self.refractory_period > 0:
                self.refractory_period -= self.time_step
            else:
                self.integrate_current()

            time.sleep(self.time_step)

    def integrate_current(self):
        self.voltage += (self.current - self.voltage) * self.time_step
        self.current *= (1.0 - self.decay)

    def send_currents(self):
        for neighbor in self.neighbors:
            neighbor.receive_current(self.current)

    def receive_current(self, current):
        self.current += current

if __name__ == '__main__':
    # Define the network topology
    neuron1 = AdExLIFNeuron("Neuron 1", [], "127.0.0.1", 5001)
    neuron2 = AdExLIFNeuron("Neuron 2", [neuron1], "127.0.0.1", 5002)
    neuron3 = AdExLIFNeuron("Neuron 3", [neuron1], "127.0.0.1", 5003)
    neuron4 = AdExLIFNeuron("Neuron 4", [neuron2, neuron3], "127.0.0.1", 5004)


    # Start the threads
    threads = []
    for neuron in [neuron1, neuron2, neuron3, neuron4]:
        thread = threading.Thread(target=neuron.run)
        thread.start()
        threads.append(thread)

    # Wait for KeyboardInterrupt (Ctrl+C) to stop the nodes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for thread in threads:
            thread.join()
