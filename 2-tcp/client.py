import socket
import threading
from modules import LIF
import json
import numpy as np

#   TO BE IMPLEMENTED
#with open("./node_profiles/")
# Generate node ID and save node profile and log as json

node_id = input("Choose your node_id: ")

existing_nodes = []

# Connecting To Server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 55555))

# Listening to Server and Sending Nickname
def receive():
    tick = 0
    while True:
        try:
            message = client.recv(1024).decode('utf-8')
            if message == 'N-ID':
                client.send(node_id.encode('utf-8'))
            else:
                print(message)
                I = message

                dV =  (I - gl*(V[tick-1]-El))/Cm
                V.append(V[tick-1] + dV*dt)
                # in case we exceed threshold
                if V[tick] > thresh:
                    V[tick-1] = 0.04   # set the last step to spike value
                    V[tick] = El       # current step is resting membrane potential
                    return V[tick]
                tick += 1

        except Exception as e:
            client.close()
            # Close Connection When Error
            print(e)
            print("An error occured!")
            break

# Sending Messages To Server
def write():
    while True:
        message = '{}: {}'.format(node_id, input(''))
        client.send(message.encode('utf-8'))

#   Default Params
gl = 0.16
Cm = 0.0049
El = -0.065                      # resting membrane potential [V]
thresh = -0.050                      # spiking threshold [V]
V = np.array([])     # array for saving Voltage history
for n in range(4):
    V.append(El)
dt = 1

# Starting Threads For Listening And Writing
receive_thread = threading.Thread(target=receive)
receive_thread.start()

write_thread = threading.Thread(target=write)
write_thread.start()