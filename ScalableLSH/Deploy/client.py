import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
socket.send(b"http://10.3.2.61/~ubuntu/projects/001_backpage/dataset/corpus/ImagesNevada/Nevada_2012_10_10_1349862492000_4_0.jpg" + "\0")
message = socket.recv()
print message
