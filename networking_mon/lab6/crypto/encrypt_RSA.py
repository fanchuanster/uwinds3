#!/usr/bin/python3

from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

message=b'This is my message\n'

key=RSA.import_key(open('Test.pub').read())


cipher=PKCS1_OAEP.new(key)

ciphertext=cipher.encrypt(message)

f=open('ciphertext.bin', 'wb')

f.write(ciphertext)

f.close()
