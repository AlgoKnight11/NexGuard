"""
Generates a realistic test PCAP file with Scapy containing:
- Baseline DNS/HTTP traffic
- Reconnaissance port scans
- SSH Brute Force authentication attempts
"""
import os
import time
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

def generate_test_pcap(output_path="sample_capture.pcap", duration_sec=15):
    packets = []
    base_time = time.time() - 3600 # 1 hour ago
    
    # 1. First 5 seconds: Normal Web & DNS traffic
    for t in range(5):
        pkt_time = base_time + t + 0.1
        # DNS query
        dns_pkt = Ether()/IP(src="192.168.1.50", dst="8.8.8.8", ttl=64)/UDP(sport=53210+t, dport=53)/Raw(load=b"test.local")
        dns_pkt.time = pkt_time
        packets.append(dns_pkt)
        
        # HTTP traffic
        for i in range(4):
            tcp_syn = Ether()/IP(src="192.168.1.50", dst="10.0.0.5", ttl=64)/TCP(sport=40000+t*10+i, dport=80, flags="S", window=29200)
            tcp_syn.time = pkt_time + 0.05 * i
            packets.append(tcp_syn)
            
            tcp_ack = Ether()/IP(src="10.0.0.5", dst="192.168.1.50", ttl=64)/TCP(sport=80, dport=40000+t*10+i, flags="SA", window=28960)
            tcp_ack.time = pkt_time + 0.05 * i + 0.01
            packets.append(tcp_ack)

    # 2. Next 5 seconds: Reconnaissance Port Scanning (high entropy)
    for t in range(5, 10):
        pkt_time = base_time + t + 0.05
        for p in range(10):
            target_port = 20 + t * 10 + p
            scan_pkt = Ether()/IP(src="192.168.1.100", dst="10.0.0.5", ttl=52)/TCP(sport=55000+p, dport=target_port, flags="S", window=1024)
            scan_pkt.time = pkt_time + 0.02 * p
            packets.append(scan_pkt)

    # 3. Next 5 seconds: SSH Brute Force attempts on Port 22
    for t in range(10, duration_sec):
        pkt_time = base_time + t + 0.02
        for attempt in range(8):
            client_port = 45000 + attempt
            # SYN
            p1 = Ether()/IP(src="192.168.1.100", dst="10.0.0.5", ttl=52)/TCP(sport=client_port, dport=22, flags="S", window=26883)
            p1.time = pkt_time + 0.05 * attempt
            # ACK
            p2 = Ether()/IP(src="192.168.1.100", dst="10.0.0.5", ttl=52)/TCP(sport=client_port, dport=22, flags="A", window=26883)
            p2.time = pkt_time + 0.05 * attempt + 0.01
            # PSH-ACK (auth payload)
            p3 = Ether()/IP(src="192.168.1.100", dst="10.0.0.5", ttl=52)/TCP(sport=client_port, dport=22, flags="PA", window=26883)/Raw(load=b"SSH-2.0-OpenSSH_8.2\n")
            p3.time = pkt_time + 0.05 * attempt + 0.02
            # RST (rejected)
            p4 = Ether()/IP(src="10.0.0.5", dst="192.168.1.100", ttl=64)/TCP(sport=22, dport=client_port, flags="R", window=0)
            p4.time = pkt_time + 0.05 * attempt + 0.03
            
            packets.extend([p1, p2, p3, p4])

    wrpcap(output_path, packets)
    print(f"Generated {len(packets)} packets in {output_path}")
    return output_path

if __name__ == "__main__":
    generate_test_pcap()
