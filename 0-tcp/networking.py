
import sys
import socket
import selectors
import types

class client:

    def __init__(self) -> None:
        self.sel = selectors.DefaultSelector()

    @staticmethod
    def start_connections(self, host, port, sel, message = [b"gg lol"]):
        server_addr = (host, port)
        print(f"Connecting to {server_addr}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.connect_ex(server_addr)
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        data = types.SimpleNamespace(
            msg_total=sum(len(m) for m in message),
            recv_total=0,
            messages=message.copy(),
            outb=b"",
        )
        sel.register(sock, events, data=data)

    @staticmethod
    def service_connection(self, key, mask, sel):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                print(f"Received {recv_data!r} from connection")
                data.recv_total += len(recv_data)
            if not recv_data or data.recv_total == data.msg_total:
                print(f"Closing connection")
                sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if not data.outb and data.messages:
                data.outb = data.messages.pop(0)
            if data.outb:
                print(f"Sending {data.outb!r}")
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]

    def send_data(self):
        self.sel = selectors.DefaultSelector()
        HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
        PORT = 5550  # Port to listen on (non-privileged ports are > 1023)
        self.start_connections(HOST, int(PORT), self.sel)
        try:
            while True:
                events = self.sel.select(timeout=1)
                if events:
                    for key, mask in events:
                        self.service_connection(key, mask, self.sel)
                # Check for a socket being monitored to continue.
                if not self.sel.get_map():
                    break
        except KeyboardInterrupt:
            print("Caught keyboard interrupt, exiting")
        finally:
            self.sel.close()

