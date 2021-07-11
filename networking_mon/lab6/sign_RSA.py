#!/usr/bin/python3

from Crypto.Signature import pss
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

message=b'I owe you $3000'
key_str=open("private.pem").read()
key=RSA.importKey(key_str, passphrase="111111")
h=SHA256.new(message)
print("hexdigest - '{}'".format(message), h.hexdigest())
signer=pss.new(key)
sig=signer.sign(h)
open("signature3000.bin", "wb").write(sig)


