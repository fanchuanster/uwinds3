#!/usr/bin/python3
from socket import *
serverPort = 12000

# initialize a server socket and bind it to a given port number, so as to listen/welcome on the port.
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print('The server is ready to receive')

# loop on the port to receive incoming message.
while True:
    # receive message with maximum given size at a time
    message, clientAddress = serverSocket.recvfrom(2048)
    modifiedMessage = message.decode().upper()

    # answer the connection by sending message to the client.
    serverSocket.sendto(modifiedMessage.encode(), clientAddress)