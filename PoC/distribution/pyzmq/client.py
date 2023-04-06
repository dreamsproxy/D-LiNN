import sys
import zmq

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

if len(sys.argv) > 2:
    port1 =  sys.argv[2]
    int(port1)

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print ("Collecting updates from weather server...")
socket.connect ("tcp://localhost:%s" % port)

if len(sys.argv) > 2:
    socket.connect ("tcp://localhost:%s" % port1)

# Subscribe to zipcode, default is NYC, 10001
topicfilter = "10001"
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

# Process 5 updates
total_value = 0
for update_nbr in range (5):
    string = socket.recv()
    topic, messagedata = string.split()

    #   DECODE DATA
    messagedata = messagedata.decode()
    topic = topic.decode()

    total_value += int(messagedata)
    print(topic, messagedata)

DEBUG = False

if DEBUG:
    print(f"Topic Rate: {total_value / update_nbr}")
    print(f"Topic Rate: {type(int(total_value / update_nbr))}")
    print(f"\tType:\t{type(total_value)}")
    print(f"\tType:\t{type(update_nbr)}")

print(f"Average messagedata value for topic {topicfilter} was {total_value / update_nbr}")