import socket
import threading

host = '127.0.0.1'
port = 55555

# Starting Server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen()

# Lists For Clients and Their node_id_log
clients = []
node_id_log = []

# Sending Messages To All Connected Clients
def broadcast(message):
    for client in clients:
        client.send(message)

# Handling Messages From Clients
def handle(client):
    while True:
        try:
            # Broadcasting Messages
            message = client.recv(1024)
            broadcast(message)
        except:
            # Removing And Closing Clients
            index = clients.index(client)
            clients.remove(client)
            client.close()
            node_id = node_id_log[index]
            broadcast('{} disconnected!'.format(node_id).encode('utf-8'))
            node_id_log.remove(node_id)
            break

# Receiving / Listening Function
def receive():
    while True:
        # Accept Connection
        client, address = server.accept()
        print("Connected with {}".format(str(address)))

        # Request And Store Nickname
        client.send('N-ID'.encode('utf-8'))
        node_id = client.recv(1024).decode('utf-8')
        node_id_log.append(node_id)
        clients.append(client)

        # Print And Broadcast Nickname
        print("Node ID: {}".format(node_id))
        broadcast("{} linked!".format(node_id).encode('utf-8'))
        client.send('Connection Established'.encode('utf-8'))

        # Start Handling Thread For Client
        thread = threading.Thread(target=handle, args=(client,))
        thread.start()

receive()