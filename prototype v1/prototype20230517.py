import time
from queue import Queue, Empty
import threading
import json
from p2pnetwork.node import Node as P2PNode

import numpy as np
from time import sleep
import matplotlib.pyplot as plt


# Code created on 2023-05-17
# Prototype to demonstrate communication between node
# to be independent from neuron thingy. For Alan

class AdaptiveLIF:
    def __init__(self, tau_m: float, R: float, V_th: float, V_reset: float, delta_t: float, tau_leak: float):
        self.tau_m = tau_m
        self.R = R
        self.V_th = V_th
        self.V_reset = V_reset
        self.delta_t = delta_t
        self.tau_leak = tau_leak
        self.input_current_value = 0  # Initial input current value

        # Debug trace vars
        self.dV_calculated = False
        self.V_apenned = False
        self.time_apended = False
        self.input_trace = []

    def set_input_current(self, value: float):
        print(value)
        self.input_current_value = value
        return self.input_current_value
    
    def input_current(self, t: int):
        return self.input_current_value

    def dV_step(self, input_current: float, V: float):
        dV = (-(V[-1] - input_current * self.R) / self.tau_m) * self.delta_t - (V[-1] / self.tau_leak) * self.delta_t
        self.dV_calculated = True

        return dV

    def simulate(self):
        time = []
        V = [self.V_reset]  # Initialize with reset potential
        spikes = []

        t = 0
        try:
            self.dV_calculated = False
            self.time_apended = False

            input_current = 0

            self.input_trace.append(input_current)
            dV = self.dV_step(input_current, V)

            V.append(V[-1] + dV)
            self.V_apenned = True

            # Check for spike
            if V[-1] >= self.V_th:
                spike_check = V[-1]
                V[-1] = self.V_reset
                spikes.append(1)
            else:
                spikes.append(0)

            time.append(t)
            self.time_apended = True
            t += self.delta_t
            sleep(self.delta_t / 1000)  # Sleep to control the simulation speed

        except KeyboardInterrupt:
            # Ensure that time and V are the same length,
            #   Making sure that either V or time updates had executed at the same time constraint
            if len(time) < len(V):
                time.append(t)
                self.time_apended = True
                t += self.delta_t
            if len(V) < len(time):
                if self.dV_calculated:
                    V.append(V[-1] + dV)
                    self.V_apenned = True
            if len(spikes) < len(V):
                # Check for spike
                if V[-1] >= self.V_th:
                    V[-1] = self.V_reset
                    spikes.append(1)
                else:
                    spikes.append(0)

        #print(self.input_trace)
        return time, V, spikes, spike_check

class MyNeuronNode(threading.Thread):

    def __init__(self, id_: str, data_queue: Queue):
        super().__init__()

        self.id: str = id_
        self.q: Queue = data_queue
        self.counter: int = 0

        self.terminate_flag = threading.Event()

        return
    
    def check_spike(self, spike_check, threshold):
        if spike_check >= threshold:
            net1.send_to_nodes
    
    def run(self):
        while not self.terminate_flag.is_set():

            tau_m = 20      # Membrane time constant (ms)
            R = 1           # Membrane resistance (Mohm)
            V_th = 1        # Threshold potential (mV)
            V_reset = 0     # Reset potential (mV)
            delta_t = 0.1   # Time step (ms)
            t_sim = 1000    # Simulation time (ms)
            tau_leak = 10   # Membrane leak time constant (ms)
            
            neuron = AdaptiveLIF(tau_m, R, V_th, V_reset, delta_t, tau_leak)
            time, V, spikes, spike_check = neuron.simulate(t_sim)
            spike_check(spike_check, V_th)
            # Sleep to control the simulation speed
            sleep(self.delta_t / 1000)
            

            self.counter += 1

            try:
                data = self.q.get(timeout=0.1)
                print("[{}] ({}) I got data: {}".format(self.id, self.counter, data))
                neuron.input_current(5)
            except Empty as e:
                print("[{}] ({}) No data after 0.1 second wait!".format(self.id, self.counter))
                neuron.input_current(0)
            
            print("[{}] Done processing iteration: {}".format(self.id, self.counter))

        return
    
    def stop(self):
        self.terminate_flag.set()


class MyNetwork(P2PNode):

    def __init__(self, host: str, port: int, id_: str, data_queue: Queue, callback=None, max_connections=0):
        super().__init__(host, port, id_, callback, max_connections)
        self.q: Queue = data_queue
    
    def node_message(self, node: P2PNode, data):
        print("[{}] Received data from {}, putting data '{}' into queue".format(self.id, node.id, data))
        self.q.put(data)
        

def _main() -> None:

    # # Section 1: Setup Network <-> Neuron queue
    # 
    # When other node send us data, we put data into queue
    # for neuron to pull data for processing.    

    que1: Queue = Queue()
    neu1 = MyNeuronNode("Neuron1", que1)
    net1 = MyNetwork("127.0.0.1", 8001, "Network1", data_queue=que1)

    que2: Queue = Queue()
    neu2 = MyNeuronNode("Neuron2", que2)
    net2 = MyNetwork("127.0.0.1", 8002, "Network2", data_queue=que2)
    
    que3: Queue = Queue()
    neu3 = MyNeuronNode("Neuron3", que3)
    net3 = MyNetwork("127.0.0.1", 8003, "Network3", data_queue=que3)

    net1.start()
    net2.start()
    net3.start()

    neu1.start()
    neu2.start()
    neu3.start()

    time.sleep(1)
    # Here we expect out put to be:
    # "No data after 0.1 second wait!"
    # because:
    # - network have yet to connect to each other
    # - no data been put into queue for neuron to `get`` and process 

    debug = False
    net1.debug = debug
    net2.debug = debug
    net3.debug = debug

    net1.connect_with_node('127.0.0.1', 8002)
    net2.connect_with_node('127.0.0.1', 8003)
    net3.connect_with_node('127.0.0.1', 8001)

    time.sleep(1)

    # Now we have net1 set jobs to ALL other connected nodes
    net1.send_to_nodes(json.dumps([1,2,3,4]))
    net1.send_to_nodes(json.dumps([3,4,5,6]))

    time.sleep(2)

    net1.stop()
    net2.stop()
    net3.stop()

    neu1.stop()
    neu2.stop()
    neu3.stop()

    return

if __name__ == "__main__":
    _main()
