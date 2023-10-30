class AdaptiveLIFNeuron:
    def __init__(self, ID, threshold=1.0, tau_m=10.0, tau_s=1.0, tau_a=100.0, n_type = "core", rand_params = True):
        self.ID = str(ID)
        self.resting_potential = 0.0
        self.rand_params = rand_params
        self.n_type = n_type
        self.membrane_potential = 0.0
        self.potential_log = [self.membrane_potential]
        self.adaptation = 0.0
        
        if self.rand_params:
            self.threshold = np.random.uniform(1.0, 2.0)
            self.tau_m = np.random.uniform(5.0, 15.0)
            self.tau_s = np.random.uniform(0.5, 1.5)
            self.tau_a = np.random.uniform(50.0, 150.0)
            #self.potential_leak_rate = np.random.uniform(0.1, 0.9)
        else:
            self.tau_m = tau_m
            self.tau_s = tau_s
            self.tau_a = tau_a
        #   Below adaptation_exponential_rate specifies the
        #   amount of adaptation to keep while input IS present
        self.adaptation_increment_rate = 0.5
        #   Below adaptation_exponential_rate specifies the
        #   amount of adaptation to keep while input IS NOT present
        self.adaptation_exponential_rate = 0.8
        self.potential_leak_rate = 0.9

        self.connections = None
    
    def integrate_and_fire(self, inputs):
        if inputs != 0.0:
            if self.membrane_potential > self.resting_potential:
                self.membrane_potential *= self.potential_leak_rate
            else: self.membrane_potential = self.resting_potential
            
            # Input leak
            self.membrane_potential += (inputs - self.membrane_potential) / self.tau_m
            # Adaptation lock (Adaptive ignoring of input signals)
            self.adaptation += (-self.adaptation) / self.tau_a
            # Determine potential using updated Adaptation lock
            self.membrane_potential -= self.adaptation

            if self.membrane_potential >= self.threshold:
                self.adaptation += self.adaptation_increment_rate
                self.membrane_potential = self.threshold
                self.potential_log.append(self.membrane_potential)
                #self.membrane_potential *= self.potential_leak_rate
                return self.threshold
            else:
                self.potential_log.append(self.membrane_potential)
                return None
        
        else:
            # Exponentially decay the membrane potential until it has reached
            #   the resting potential of 0.0
            if self.membrane_potential > self.resting_potential:
                self.membrane_potential *= self.potential_leak_rate
            else: self.membrane_potential = self.resting_potential
            
            # Decrement adaptation until 0.0 if it is above 0.0
            if self.adaptation > 0.0:
                self.adaptation *= self.adaptation_exponential_rate
            else: self.adaptation = 0.0
            # Adaptation lock (Adaptive ignoring of input signals)
            self.adaptation -= (-self.adaptation) / self.tau_a
            ## Determine potential using updated Adaptation lock
            #self.membrane_potential += self.adaptation
            
            if self.membrane_potential >= self.threshold: 
                self.membrane_potential = self.resting_potential
                self.potential_log.append(self.membrane_potential)
                return self.threshold
            else:
                self.potential_log.append(self.membrane_potential)
                return None

import numpy as np
import zmq
import time
import threading
from random import random

# Function to simulate a single neuron
def simulate_neuron(neuron):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")

    # Simulate the neuron's activity for a certain duration
    duration = 10  # seconds
    start_time = time.time()
    while time.time() - start_time < duration:
        # Simulate random input signal for testing
        input_signal = random()

        # Integrate and fire the neuron
        spike = neuron.integrate_and_fire(input_signal)

        if spike:
            # Send spike event to other neurons
            socket.send(neuron.ID.encode('utf-8'))

        # Sleep for a small time step (e.g., 1 ms)
        time.sleep(0.001)

    socket.close()
    context.term()

# Function to receive spike events from other neurons
def receive_spikes():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    while True:
        spike = socket.recv().decode('utf-8')
        print(f"Received spike from neuron ID: {spike}")

def main():
    num_neurons = 5
    neurons = []

    # Create neurons and start their individual threads
    threads = []
    for i in range(num_neurons):
        neuron = AdaptiveLIFNeuron(ID=i)
        neurons.append(neuron)
        thread = threading.Thread(target=simulate_neuron, args=(neuron,))
        threads.append(thread)
        thread.start()

    # Start the receiver thread to collect spike events from other neurons
    receiver_thread = threading.Thread(target=receive_spikes)
    receiver_thread.start()

    # Wait for all threads to finish their simulations
    for thread in threads:
        thread.join()

    # After all threads have finished, terminate the receiver thread
    receiver_thread.join()

if __name__ == "__main__":
    main()
