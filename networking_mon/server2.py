#! /usr/bin/python3
from socket import *
from _thread import *
serverName = "137.207.82.53"
serverPort = 12000
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind((serverName, serverPort))
serverSocket.listen(1)
print("The server is ready to receive")
def multi_threaded_client(connectionSocket):
    sentence = connectionSocket.recv(1024)
    while sentence:
        capitalizedSentence = sentence.decode().upper()
        connectionSocket.send(capitalizedSentence.encode())
        print(sentence.decode())
        sentence = connectionSocket.recv(1024)
    connectionSocket.close()

while True:
    connectionSocket, addr = serverSocket.accept()
    start_new_thread(multi_threaded_client, (connectionSocket, ))