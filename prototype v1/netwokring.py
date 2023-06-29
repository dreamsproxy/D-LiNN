import time
from queue import Queue, Empty
import threading
import json
from typing import List
from p2pnetwork.node import Node as P2PNode
import time

class MyNetwork(P2PNode):

    def inbound_node_connected(self, node):
        print("asdfasdf")
        

def _main() -> None:

    port = int(input("Port >>> "))
    net1 = MyNetwork("127.0.0.1", port, "Network1")
    
    core = input("Core? [y/n]")
    if core == "y":
        net1.connect_with_node('127.0.0.1', 8002)
        net1.debug = False

    input("Press Enter when ready to send data!")

    # Now we have net1 set jobs to ALL other connected nodes
    counter = 0
    while True:
        counter += 1
        try:
            net1.send_to_nodes(json.dumps(5))
            if counter % 300000 == 0:
                # print("Iterated {} times".format(counter))
                pass
        except KeyboardInterrupt as e:
            break

    net1.stop()

    return

if __name__ == "__main__":
    _main()