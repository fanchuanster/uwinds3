#!/usr/bin/python3

from Crypto.Cipher import AES
from Crypto.Util import Padding
from Crypto.Random import get_random_bytes


key_hex_string = '00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF'
key = bytes.fromhex(key_hex_string)
iv = get_random_bytes(16)
data = b'COMP8677 CRYPTO Lecture'
print(data.hex())

# Encrypt the entire data
cipher = AES.new(key, AES.MODE_CBC, iv)
ciphertext = cipher.encrypt(Padding.pad(data, 16))
print("Ciphertext: {0}\n".format(ciphertext.hex()))

# Decrypt the ciphertext
cipher = AES.new(key, AES.MODE_CBC, iv)
plaintext = cipher.decrypt(ciphertext)
print("Plaintext: {0}".format(Padding.unpad(plaintext, 16)))

