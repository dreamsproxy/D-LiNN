import socket
import json

HOST = "127.0.0.1"
PORT = 5550
"""
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

sock.bind((HOST, PORT))

while True:
    try:
        data, addr = sock.recvfrom(1024, socket.MSG_DONTWAIT)
        print('received data', data)
        msg = json.loads(data.decode('ascii'))
        print('parsed message', msg)

        print('LX =', msg['LX'])
    except socket.timeout as e:
        pass
"""
import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
starttime = time.time()
try:
    m = sock.recv(100, socket.MSG_DONTWAIT)
except BlockingIOError as e:
    pass

endtime = time.time()
print(f'sock.recv(100, socket.MSG_DONTWAIT) took {endtime-starttime}s')