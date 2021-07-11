#!/usr/bin/python3

from Crypto.Signature import pss
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

message = b'I owe you $3000'
signature= open('signature.bin', 'rb').read()
key = RSA.import_key(open('public.pem').read())
h = SHA256.new(message)
print(h.hexdigest())
verifier = pss.new(key)

try:
    verifier.verify(h, signature)   
    print("The signature is valid.")
except (ValueError, TypeError):
    print("The signature is NOT valid.")

