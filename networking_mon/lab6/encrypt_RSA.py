#!/usr/bin/python3

from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

message=str.encode(input("input the message you would like to encrypt:"))

key=RSA.importKey(open('public.pem').read())


cipher=PKCS1_OAEP.new(key)

ciphertext=cipher.encrypt(message)

f=open('ciphertext.bin', 'wb')

f.write(ciphertext)

f.close()
