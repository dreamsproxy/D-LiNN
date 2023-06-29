import time
from queue import Queue, Empty
import threading
import json
from typing import List
from p2pnetwork.node import Node as P2PNode
import time

from neuron import AdaptiveLIF, plot_neuron

# Code created on 2023-05-17
# Prototype to demonstrate communication between node
# to be independent from neuron thingy. For Alan

class MyNeuronNode(threading.Thread):

    def __init__(self, id_: str, data_queue: Queue):
        super().__init__()

        self.id: str = id_
        self.q: Queue = data_queue
        self.counter: int = 0

        self.terminate_flag = threading.Event()

        tau_m = 20      # Membrane time constant (ms)
        R = 1           # Membrane resistance (Mohm)
        V_th = 1        # Threshold potential (mV)
        V_reset = 0     # Reset potential (mV)
        delta_t = 0.1   # Time step (ms)
        t_sim = 1000    # Simulation time (ms)
        tau_leak = 10   # Membrane leak time constant (ms)
        self.neuro_sama: AdaptiveLIF = AdaptiveLIF(tau_m, R, V_th, V_reset, delta_t, tau_leak)

        return
    
    def run(self):
        print("[{}] started".format(self.id))
        while not self.terminate_flag.is_set():

            node_input: List[float] = []

            # Simulate task which takes time
            time.sleep(self.neuro_sama.delta_t)

            self.counter += 1

            try:
                while True:
                    node_input.append(self.q.get(timeout=0))
            except Empty as e:
                print("[{}] ({}) got {} count of data".format(self.id, self.counter, len(node_input)))
                pass
            
            if len(node_input) == 0:
                node_input.append(0)

            self.neuro_sama.simulate_step(node_input)

            # print("[{}] Done processing iteration: {}".format(self.id, self.counter))

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
    
    def inbound_node_connected(self, node):
        print("asdfasdf")
        

def _main() -> None:

    que2: Queue = Queue()
    neu2 = MyNeuronNode("Neuron2", que2)
    net2 = MyNetwork("127.0.0.1", 8002, "Network2", data_queue=que2)
    
    net2.start()
    # neu2.start()

    time.sleep(1)

    net2.debug = False

    input("Press Enter when ready to receive data!")

    net2.connect_with_node('127.0.0.1', 8001)

    time.sleep(1)

    net2.stop()
    neu2.stop()

    plot_neuron(neu2.neuro_sama._time, neu2.neuro_sama._V, neu2.neuro_sama._spikes)

    return

if __name__ == "__main__":
    _main()