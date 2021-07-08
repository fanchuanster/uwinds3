#!/usr/bin/python3

from Crypto.Hash import SHA512, MD5, SHA224


msg0="Harry Porter"
msg1="Alice in Wonderland"

#hash directly on messages 
h512=SHA512.new(msg0.encode())   #.new() initiaze a hash object; input should be byte string
h224=SHA224.new(msg0.encode())
h5=MD5.new(msg0.encode())

#hash serveral messages sequentially.  

mh5=MD5.new()
mh5.update(msg0.encode())   # msg0 is hashed. 
mh5.update(msg1.encode())  # now msg0+msg1 is hashed.  

print("SHA512({})={}".format(msg0, h512.hexdigest()))
print("SHA224({})={}".format(msg0, h224.hexdigest()))
print("MD5({})={}".format(msg0, h5.hexdigest()))
print("MD5({})={}".format(msg0+msg1, mh5.hexdigest()))

