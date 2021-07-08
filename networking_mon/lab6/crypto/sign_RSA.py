#!/usr/bin/python3

from Crypto.Signature import pss
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

message=b'I owe you $3000'
key_str=open("private.pem").read()
key=RSA.import_key(key_str, passphrase="dees")
h=SHA256.new(message)
print(h.hexdigest())
signer=pss.new(key)
sig=signer.sign(h)
open("signature.bin", "wb").write(sig)


