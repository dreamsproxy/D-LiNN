#import socket
from modules.activation import LIF
from modules import networking
from queue import Queue
import socket
import threading
import time
from random import uniform
"""
Server listen for conn
LIF run in another thread
If server recv data
    update I inside LIF as it is running but during time.sleep forced delay
else
    LIF continues to run without distrubance
"""

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 5550  # Port to listen on (non-privileged ports are > 1023)
neural_log = []
# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = (HOST, PORT)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

while True:
    print('\nwaiting to receive message')
    data, address = sock.recvfrom(2048)

    print(f'Signal from {address}')
    print(data)

    if data:
        sent = sock.sendto(data, address)
        print('sent %s bytes back to %s' % (sent, address))

    if str(address) not in neural_log:
        neural_log.append(str(address))
        #   log addr and assign weight by random

        networking.client()
        node_data = []

        t = threading.Thread(target=LIF, args = [0.006, 0.16, 0.0049, node_data])

        t.start()
        print(f"Active Threads:")
        print(threading.active_count())
        t.join()
        print(node_data)

except KeyboardInterrupt:
    print("Caught keyboard interrupt, exiting")
finally:
    sel.close()