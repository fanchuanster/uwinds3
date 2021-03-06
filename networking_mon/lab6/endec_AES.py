#!/usr/bin/python3

from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Util import Padding
from Crypto.Random import get_random_bytes

import binascii

def encryptRSA(message, keyfile):
    key=RSA.importKey(open(keyfile).read())
    cipher=PKCS1_OAEP.new(key)
    ciphertext=cipher.encrypt(message)
    return ciphertext

def decryptRSA(ciphertext, privatekeyfile):
    key_str = open(privatekeyfile).read()
    prikey = RSA.importKey(key_str, passphrase='111111')
    cipher = PKCS1_OAEP.new(prikey)
    message = cipher.decrypt(ciphertext)
    return message


key_hex_string = '00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF'
key = bytearray.fromhex(key_hex_string)

# a.
iv = get_random_bytes(16)
print("sk - {}".format(binascii.hexlify(bytearray(iv))))

# b.
c1 = encryptRSA(iv, "public.pem")
print("c1 - {}".format(binascii.hexlify(bytearray(c1))))

data = b'I find the solution for P not equal NP'

# c.
cipher = AES.new(key, AES.MODE_CBC, iv)
c2 = cipher.encrypt(Padding.pad(data, 16))
print("c2: {0}\n".format(binascii.hexlify(bytearray(c2))))


# d.
decrpted_message = decryptRSA(c1, "private.pem")
print("decrypted c1 (sk) - {}".format(binascii.hexlify(bytearray(decrpted_message))))

# Decrypt the ciphertext
cipher = AES.new(key, AES.MODE_CBC, iv)
plaintext = cipher.decrypt(c2)
print("decrypted msg - {0}".format(Padding.unpad(plaintext, 16)))

