from socket import *
serverName = "10.199.62.80"
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect((serverName, serverPort))
sentence = input("Input lowercase sentence:")
while sentence:
    clientSocket.send(sentence.encode())
    S1="fff1"
    clientSocket.send(S1.encode())
    S2="fff2"
    clientSocket.send(S2.encode())
    S3="fff3"
    clientSocket.send(S3.encode())
    modifiedSentence = clientSocket.recv(1024)
    print("From Server: ", modifiedSentence.decode())
    sentence=input("Input your lowercase sentence:")
clientSocket.close()