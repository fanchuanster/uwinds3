#! /usr/bin/python3
from socket import *
from _thread import *
serverName = ""
serverPort = 12000

# create a socket in the server program
serverSocket = socket(AF_INET, SOCK_STREAM)

# bind it to a given port of the hosting server name, empty means any available name of the server.
serverSocket.bind((serverName, serverPort))

# start listening on the local port, 1 (backlog) means 1 unaccepted new incoming connection is allowed in the waiting queue.
serverSocket.listen(1)
print("The server is ready to receive")
def multi_threaded_client(connectionSocket):
    # receive data from the connection.
    sentence = connectionSocket.recv(1024)
    # once there is incoming data received, read and process to the end.
    while sentence:
        capitalizedSentence = sentence.decode().upper()
        connectionSocket.send(capitalizedSentence.encode())
        print(sentence.decode())
        sentence = connectionSocket.recv(1024)
    # close the new connection when done.
    connectionSocket.close()

while True:
    # polling and welcoming new connection
    connectionSocket, addr = serverSocket.accept()

    # handle the new connection in a separate thread.
    start_new_thread(multi_threaded_client, (connectionSocket, ))