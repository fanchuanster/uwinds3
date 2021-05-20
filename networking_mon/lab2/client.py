#!/usr/bin/python3
from socket import *
serverName = "10.211.36.170"
serverPort = 12000

# initialize a local socket which is similar to a local endpoint
clientSocket = socket(AF_INET, SOCK_DGRAM)
message = input("input your message:")

# through the local socket, send message to remote server on the given port.
clientSocket.sendto(message.encode(), (serverName, serverPort))

# then receive response from server.
modifiedMessage, serverAddress = clientSocket.recvfrom(2048)

# print the response message
print(modifiedMessage.decode())

# destruct the local socket in the end.
clientSocket.close()