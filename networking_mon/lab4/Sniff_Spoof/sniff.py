#!/usr/bin/python3
from scapy.all import *

print("SNIFFING PACKETS.........")

def print_pkt(pkt):    
      pkt.show2()

pkt = sniff(filter='icmp and host 10.5.40.12',prn=print_pkt)   








   # print("Destination IP:", pkt[IP].dst)
   # print("Protocol:", pkt[IP].proto)
   # print("\n")
