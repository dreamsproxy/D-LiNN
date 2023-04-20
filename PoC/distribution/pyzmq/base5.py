
import time
import zmq
from threading import Thread
from multiprocessing import Process
from random import randint

def task_to_do() -> str:
    # Just a demo task which return data
    
    target_id = "10001"
    target_branch_port = 12314
    node_id = "10000"
    data = str(randint(0,100000))

    return f"{target_id} {target_branch_port} {node_id} {data}"


def run_server(_name: str, _sender_id: str, _source_branch_port: int) -> None:
    # Note:
    # parameters that isn't used have underscore in front.

    SERVER_BIND_TO = "tcp://*:5555"

    print("ServerThread is running at {}".format(SERVER_BIND_TO))

    context = zmq.Context()
    
    socket = context.socket(zmq.PUB)
    socket.bind(SERVER_BIND_TO)
    
    # Server-side event loop
    while True:
        
        outbound_data = task_to_do()
        
        try:
            socket.send(outbound_data.encode('utf8'))
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                break
            else:
                raise

        time.sleep(0.1)

    return


def run_client(node_id_to_listen_for: str) -> None:

    context = zmq.Context()

    SERVER_ADDR = "tcp://localhost:5555"

    # Socket to talk to server
    print("Connecting to <{}>".format(SERVER_ADDR))
    socket = context.socket(zmq.SUB)
    socket.connect(SERVER_ADDR)
    socket.setsockopt(zmq.SUBSCRIBE, node_id_to_listen_for.encode("utf8"))

    # Process 5 updates
    total_value = 0
    for update_nbr in range (5):
        raw_data: bytes = socket.recv()
        data_as_str: str = raw_data.decode("utf8")

        recv_data = data_as_str.split()
        print("Recieved: <{}>".format(recv_data))

        topic: str = recv_data[0]
        branch_port: int = int(recv_data[1])
        node_id: str = recv_data[2]
        data: str = recv_data[3]

        # Do stuff with said data

    context.destroy()

    return


def main():

    node_id_to_listen_for = "10001"

    server = Process(target=run_server, args=(node_id_to_listen_for, "value", 1234), daemon=True)
    server.start()

    run_client(node_id_to_listen_for)

    return


if __name__ == "__main__":
    print("--- Starting ---")
    main()
    print("--- Completed ---")