import zmq
import random
import sys
import time
from multiprocessing import Process
import threading

global DEBUG
DEBUG = False
thread_control = threading.Condition()
global periodical_array

class ServerThread(threading.Thread):
    def __init__(self, name, sender_id, source_branch_port):
        threading.Thread.__init__(self)
        self.name = name
        self.sender_id = sender_id
        self.source_branch_port = source_branch_port

    def run(self, DEBUG, source_branch_port, data, node_id, target_id):
        with open("out_nodes.csv", "r") as node_comms:
            direct_connections = node_comms.readlines()
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://*:{self.source_branch_port}")

        thread_control.acquire()

        outbound_data = f"[{target_id}, {target_branch_port}, {node_id}, {data}]"

        print(f"{target_id}, {outbound_data}")
        socket.send_string(f"{target_id}, {outbound_data}")
        time.sleep(1)


def ListenerThread(DEBUG, branch_port, listener_id):

    print(f"[INIT] Listener #{listener_id}")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    branch_port = int("5556")

    socket.connect ("tcp://localhost:%s" % branch_port)
    print (f"[LOG]\tListerner #{listener_id}\t\tbinded on branch-port {branch_port}")

    node_id = "10001"
    socket.setsockopt_string(zmq.SUBSCRIBE, node_id)

    # Process 5 updates
    total_value = 0
    for update_nbr in range (5):
        inbound_data = socket.recv()
        topic, node_data = inbound_data.split()

        #   DECODE DATA
        inbound_data = inbound_data.decode()
        topic = topic.decode()

        print(type(messagedata))


    if DEBUG:
        print(f"Topic Rate: {total_value / update_nbr}")
        print(f"Topic Rate: {type(int(total_value / update_nbr))}")
        print(f"\tType:\t{type(total_value)}")
        print(f"\tType:\t{type(update_nbr)}")

    print(f"Average messagedata value for topic {node_id} was {total_value / update_nbr}")
    exit()

def broadcast(branch_port):
    node_id = 0
    """Initialize Server Handler"""
    server_thread = threading.Thread(target=server, args=(DEBUG, branch_port), daemon=True)
    server_thread.start()
    print("[LOG]\tStarted Server Handler")

def main():
    #data_array = [["Sender ID", "Branch Port", "Target ID", "Branch Port", ["Voltage", "time taken"]]]
    #data_array = [["N0", "5565", "N1", "5565", ["-55", "30"]]]
    periodical_array = []
    global array_format
    array_format = f"[[{sender_id}, {source_branch_port}, {target_id}, {target_branch_port}, [{Voltage}, {delta_T}]]]"

    n_listeners = 3
    """Initialize Listener Handler"""
    i = 0
    client_thread = threading.Thread(target=ListenerThread, args=(DEBUG, branch_port, i), daemon=True)
    client_thread.start()
    print(f"[LOG]\tListener {i} Started")

    #print("Started All Listeners")

