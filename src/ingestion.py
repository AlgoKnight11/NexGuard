"""
NexGaurd - Input Ingestion and Multi-Level Feature Extraction Pipeline.
Supports both:
1. Flow-level telemetry (NetFlow / IPFIX / CIC-IDS CSVs)
2. Packet-level telemetry (Raw PCAP captures parsed with Scapy)
3. Streaming telemetry simulation and synthetic attack injection
"""

import os
import io
import math
import numpy as np
import pandas as pd
from collections import defaultdict

# Expected 20 features in exact order for the trained scaler and model
MODEL_FEATURE_COLS = [
    'flow_count', 'flow_duration_mean', 'tot_fwd_pkts_sum', 'tot_bwd_pkts_sum',
    'tot_len_fwd_pkts_sum', 'tot_len_bwd_pkts_sum', 'pkt_len_mean', 'flow_iat_mean',
    'init_fwd_win_mean', 'init_bwd_win_mean', 'port_entropy', 'unique_ports',
    'tcp_ratio', 'udp_ratio', 'syn_ratio', 'ack_ratio', 'rst_ratio',
    'psh_ratio', 'fin_ratio', 'urg_ratio'
]

def calculate_shannon_entropy(items):
    """Computes Shannon entropy for port access patterns."""
    if not items:
        return 0.0
    total = len(items)
    counts = defaultdict(int)
    for it in items:
        counts[it] += 1
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)

class PCAPFeatureExtractor:
    """
    Ingests raw PCAP/PCAPNG packet captures and extracts both:
    - Packet-level attributes (TTL variance, TCP window sizes, payload sizes, TCP flags)
    - Flow-level temporal aggregates binned into 1-second state vectors.
    """
    def __init__(self, window_sec=1):
        self.window_sec = window_sec

    def parse_pcap(self, pcap_source):
        """
        Parses a PCAP file path or file-like object.
        Returns:
            agg_df: DataFrame with 1-second aggregated states conforming to MODEL_FEATURE_COLS
            diagnostics: Packet-level telemetry diagnostics dictionary (TTL, window sizes, fragments)
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            raise ImportError("scapy is required for PCAP parsing. Run 'pip install scapy'.")

        packets = rdpcap(pcap_source)
        if not packets:
            raise ValueError("No packets found in PCAP source.")

        first_time = float(packets[0].time)
        
        # Diagnostics tracking
        ttl_values = []
        win_sizes = []
        payload_sizes = []
        retransmissions = 0
        fragmented_packets = 0
        seen_seq_ack = set()

        # Group packet events by 1-second time bins
        time_bins = defaultdict(lambda: {
            'flow_count': 0,
            'fwd_pkts': 0,
            'bwd_pkts': 0,
            'tot_len_fwd': 0,
            'tot_len_bwd': 0,
            'pkt_lens': [],
            'iats': [],
            'fwd_wins': [],
            'bwd_wins': [],
            'ports': [],
            'tcp_cnt': 0,
            'udp_cnt': 0,
            'syn_cnt': 0,
            'ack_cnt': 0,
            'rst_cnt': 0,
            'psh_cnt': 0,
            'fin_cnt': 0,
            'urg_cnt': 0,
            'flow_durations': []
        })

        prev_time = None
        flows = defaultdict(lambda: {'start': 0.0, 'last': 0.0, 'fwd_pkts': 0, 'bwd_pkts': 0})

        for pkt in packets:
            t = float(pkt.time)
            rel_t = t - first_time
            bin_idx = int(rel_t // self.window_sec)
            bin_data = time_bins[bin_idx]

            # Inter-arrival time (packet-level)
            if prev_time is not None:
                bin_data['iats'].append(max(0.0, (t - prev_time) * 1e6))
            prev_time = t

            pkt_len = len(pkt)
            bin_data['pkt_lens'].append(pkt_len)

            if IP in pkt:
                ip = pkt[IP]
                ttl_values.append(ip.ttl)
                # Check fragment flag (IP.flags & 0x01 or fragment offset > 0)
                if getattr(ip, 'flags', None) and ('MF' in str(ip.flags) or ip.frag > 0):
                    fragmented_packets += 1

                proto = ip.proto
                src_ip = ip.src
                dst_ip = ip.dst

                if TCP in pkt:
                    tcp = pkt[TCP]
                    bin_data['tcp_cnt'] += 1
                    bin_data['ports'].append(tcp.dport)
                    win_sizes.append(tcp.window)

                    flags = tcp.flags
                    if flags & 0x02: bin_data['syn_cnt'] += 1  # SYN
                    if flags & 0x10: bin_data['ack_cnt'] += 1  # ACK
                    if flags & 0x04: bin_data['rst_cnt'] += 1  # RST
                    if flags & 0x08: bin_data['psh_cnt'] += 1  # PSH
                    if flags & 0x01: bin_data['fin_cnt'] += 1  # FIN
                    if flags & 0x20: bin_data['urg_cnt'] += 1  # URG

                    # Retransmission heuristic
                    seq_key = (src_ip, tcp.sport, dst_ip, tcp.dport, tcp.seq)
                    if seq_key in seen_seq_ack:
                        retransmissions += 1
                    else:
                        seen_seq_ack.add(seq_key)

                    # Flow direction heuristic
                    flow_id = tuple(sorted([(src_ip, tcp.sport), (dst_ip, tcp.dport)]))
                    if flow_id not in flows:
                        flows[flow_id]['start'] = t
                        bin_data['fwd_wins'].append(tcp.window)
                    flows[flow_id]['last'] = t
                    flows[flow_id]['fwd_pkts'] += 1
                    bin_data['tot_len_fwd'] += pkt_len
                    bin_data['fwd_pkts'] += 1

                    # Payload size
                    payload = len(tcp.payload)
                    payload_sizes.append(payload)

                elif UDP in pkt:
                    udp = pkt[UDP]
                    bin_data['udp_cnt'] += 1
                    bin_data['ports'].append(udp.dport)
                    flow_id = tuple(sorted([(src_ip, udp.sport), (dst_ip, udp.dport)]))
                    if flow_id not in flows:
                        flows[flow_id]['start'] = t
                    flows[flow_id]['last'] = t
                    flows[flow_id]['fwd_pkts'] += 1
                    bin_data['tot_len_fwd'] += pkt_len
                    bin_data['fwd_pkts'] += 1
                    payload_sizes.append(len(udp.payload))

        # Flow duration summaries
        for flow_info in flows.values():
            dur = max(0.0, (flow_info['last'] - flow_info['start']) * 1e6)
            dur_bin = int((flow_info['start'] - first_time) // self.window_sec)
            time_bins[dur_bin]['flow_durations'].append(dur)
            time_bins[dur_bin]['flow_count'] += 1

        # Synthesize consecutive 1-second state records
        max_bin = max(time_bins.keys()) if time_bins else 0
        records = []
        for b_idx in range(max_bin + 1):
            data = time_bins[b_idx]
            fl_count = max(1, data['flow_count'])
            tot_pkts = max(1, data['fwd_pkts'] + data['bwd_pkts'])

            rec = {
                'flow_count': data['flow_count'],
                'flow_duration_mean': float(np.mean(data['flow_durations'])) if data['flow_durations'] else 0.0,
                'tot_fwd_pkts_sum': data['fwd_pkts'],
                'tot_bwd_pkts_sum': data['bwd_pkts'],
                'tot_len_fwd_pkts_sum': data['tot_len_fwd'],
                'tot_len_bwd_pkts_sum': data['tot_len_bwd'],
                'pkt_len_mean': float(np.mean(data['pkt_lens'])) if data['pkt_lens'] else 0.0,
                'flow_iat_mean': float(np.mean(data['iats'])) if data['iats'] else 0.0,
                'init_fwd_win_mean': float(np.mean(data['fwd_wins'])) if data['fwd_wins'] else 0.0,
                'init_bwd_win_mean': float(np.mean(data['bwd_wins'])) if data['bwd_wins'] else 0.0,
                'port_entropy': calculate_shannon_entropy(data['ports']),
                'unique_ports': len(set(data['ports'])),
                'tcp_ratio': data['tcp_cnt'] / tot_pkts,
                'udp_ratio': data['udp_cnt'] / tot_pkts,
                'syn_ratio': data['syn_cnt'] / tot_pkts,
                'ack_ratio': data['ack_cnt'] / tot_pkts,
                'rst_ratio': data['rst_cnt'] / tot_pkts,
                'psh_ratio': data['psh_cnt'] / tot_pkts,
                'fin_ratio': data['fin_cnt'] / tot_pkts,
                'urg_ratio': data['urg_cnt'] / tot_pkts
            }
            records.append(rec)

        agg_df = pd.DataFrame(records)[MODEL_FEATURE_COLS]

        diagnostics = {
            "total_packets": len(packets),
            "total_seconds": max_bin + 1,
            "avg_ttl": float(np.mean(ttl_values)) if ttl_values else 64.0,
            "ttl_variance": float(np.var(ttl_values)) if ttl_values else 0.0,
            "avg_window_size": float(np.mean(win_sizes)) if win_sizes else 29200.0,
            "retransmissions": retransmissions,
            "fragmented_packets": fragmented_packets,
            "avg_payload_bytes": float(np.mean(payload_sizes)) if payload_sizes else 0.0
        }

        return agg_df, diagnostics


class CSVFeatureExtractor:
    """
    Ingests NetFlow / CIC-IDS style CSV logs and aligns columns to the 20-feature schema.
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def process_csv(self, file_path_or_buffer):
        """Processes CSV and returns aggregated feature DataFrame."""
        if isinstance(file_path_or_buffer, str):
            df = pd.read_csv(file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)

        df.columns = df.columns.str.strip()

        # Check if already aggregated to MODEL_FEATURE_COLS
        if all(col in df.columns for col in MODEL_FEATURE_COLS):
            return df[MODEL_FEATURE_COLS]

        # Otherwise aggregate using DataPreprocessor
        raw_df = self.preprocessor.clean_and_load(file_path_or_buffer)
        agg_df = self.preprocessor.aggregate_to_states(raw_df)
        return agg_df[MODEL_FEATURE_COLS]


class SyntheticTelemetryGenerator:
    """
    Generates realistic sequential network telemetry states for live simulation,
    what-if stress-testing, and zero-day attack trajectory modeling.
    """
    @staticmethod
    def generate_scenario(scenario_type="ftp_bruteforce", length_sec=30):
        """
        Synthesizes a realistic sequence of states.
        Scenarios:
        - 'benign': Healthy enterprise traffic
        - 'recon_to_ssh': Reconnaissance port scan transitioning to SSH brute force
        - 'ftp_bruteforce': FTP brute force password guessing attack progression
        - 'ddos_syn': Rapid SYN flood progression
        """
        records = []
        np.random.seed(42)

        for t in range(length_sec):
            if scenario_type == "benign":
                # Healthy web/DNS traffic
                rec = {
                    'flow_count': int(np.random.normal(25, 5)),
                    'flow_duration_mean': float(np.random.uniform(1e5, 5e6)),
                    'tot_fwd_pkts_sum': int(np.random.normal(120, 20)),
                    'tot_bwd_pkts_sum': int(np.random.normal(110, 15)),
                    'tot_len_fwd_pkts_sum': int(np.random.normal(50000, 8000)),
                    'tot_len_bwd_pkts_sum': int(np.random.normal(80000, 10000)),
                    'pkt_len_mean': float(np.random.uniform(400, 700)),
                    'flow_iat_mean': float(np.random.uniform(2e5, 8e5)),
                    'init_fwd_win_mean': 29200.0,
                    'init_bwd_win_mean': 28960.0,
                    'port_entropy': float(np.random.uniform(0.8, 1.4)),
                    'unique_ports': int(np.random.randint(4, 10)),
                    'tcp_ratio': float(np.random.uniform(0.7, 0.9)),
                    'udp_ratio': float(np.random.uniform(0.1, 0.3)),
                    'syn_ratio': float(np.random.uniform(0.02, 0.08)),
                    'ack_ratio': float(np.random.uniform(0.6, 0.8)),
                    'rst_ratio': 0.01,
                    'psh_ratio': float(np.random.uniform(0.2, 0.4)),
                    'fin_ratio': float(np.random.uniform(0.01, 0.05)),
                    'urg_ratio': 0.0
                }
            elif scenario_type == "recon_to_ssh":
                if t < 10:
                    # Early reconnaissance: high port entropy, port scanning
                    rec = {
                        'flow_count': int(np.random.normal(40, 8)),
                        'flow_duration_mean': float(np.random.uniform(5e4, 2e5)),
                        'tot_fwd_pkts_sum': int(np.random.normal(80, 15)),
                        'tot_bwd_pkts_sum': 10,
                        'tot_len_fwd_pkts_sum': 4000,
                        'tot_len_bwd_pkts_sum': 1000,
                        'pkt_len_mean': 60.0,
                        'flow_iat_mean': float(np.random.uniform(1e4, 5e4)),
                        'init_fwd_win_mean': 1024.0,
                        'init_bwd_win_mean': 0.0,
                        'port_entropy': float(np.random.uniform(3.2, 4.5)), # High port entropy
                        'unique_ports': int(np.random.randint(25, 40)),
                        'tcp_ratio': 0.98,
                        'udp_ratio': 0.02,
                        'syn_ratio': float(np.random.uniform(0.65, 0.85)), # SYN probe storm
                        'ack_ratio': 0.1,
                        'rst_ratio': float(np.random.uniform(0.3, 0.5)),
                        'psh_ratio': 0.0,
                        'fin_ratio': 0.0,
                        'urg_ratio': 0.0
                    }
                else:
                    # SSH Brute Force Infiltration (Targeting port 22)
                    ramp = min(1.0, (t - 10) / 10.0)
                    rec = {
                        'flow_count': int(np.random.normal(70 + ramp * 50, 10)),
                        'flow_duration_mean': float(np.random.uniform(1e6, 4e6)),
                        'tot_fwd_pkts_sum': int(np.random.normal(300, 40)),
                        'tot_bwd_pkts_sum': int(np.random.normal(250, 30)),
                        'tot_len_fwd_pkts_sum': 25000,
                        'tot_len_bwd_pkts_sum': 35000,
                        'pkt_len_mean': 110.0,
                        'flow_iat_mean': 50000.0,
                        'init_fwd_win_mean': 26883.0,
                        'init_bwd_win_mean': 26883.0,
                        'port_entropy': 0.05, # Port 22 locked
                        'unique_ports': 1,
                        'tcp_ratio': 1.0,
                        'udp_ratio': 0.0,
                        'syn_ratio': 0.15,
                        'ack_ratio': 0.85,
                        'rst_ratio': float(0.05 + ramp * 0.15),
                        'psh_ratio': float(0.4 + ramp * 0.3),
                        'fin_ratio': 0.05,
                        'urg_ratio': 0.0
                    }
            elif scenario_type == "ftp_bruteforce":
                # FTP Brute Force (Port 21 password guessing)
                ramp = min(1.0, t / 15.0)
                rec = {
                    'flow_count': int(np.random.normal(50 + ramp * 60, 8)),
                    'flow_duration_mean': float(np.random.uniform(2e6, 8e6)),
                    'tot_fwd_pkts_sum': int(np.random.normal(200 + ramp * 100, 25)),
                    'tot_bwd_pkts_sum': int(np.random.normal(180 + ramp * 80, 20)),
                    'tot_len_fwd_pkts_sum': 18000,
                    'tot_len_bwd_pkts_sum': 22000,
                    'pkt_len_mean': 85.0,
                    'flow_iat_mean': 80000.0,
                    'init_fwd_win_mean': 64240.0,
                    'init_bwd_win_mean': 64240.0,
                    'port_entropy': 0.02, # Concentrated on port 21
                    'unique_ports': 1,
                    'tcp_ratio': 1.0,
                    'udp_ratio': 0.0,
                    'syn_ratio': float(0.1 + ramp * 0.2),
                    'ack_ratio': 0.9,
                    'rst_ratio': float(0.1 + ramp * 0.25),
                    'psh_ratio': float(0.35 + ramp * 0.35),
                    'fin_ratio': 0.1,
                    'urg_ratio': 0.0
                }
            else: # ddos_syn
                rec = {
                    'flow_count': int(np.random.normal(300, 30)),
                    'flow_duration_mean': 1000.0,
                    'tot_fwd_pkts_sum': 800,
                    'tot_bwd_pkts_sum': 0,
                    'tot_len_fwd_pkts_sum': 48000,
                    'tot_len_bwd_pkts_sum': 0,
                    'pkt_len_mean': 60.0,
                    'flow_iat_mean': 100.0,
                    'init_fwd_win_mean': 1024.0,
                    'init_bwd_win_mean': 0.0,
                    'port_entropy': 2.8,
                    'unique_ports': 15,
                    'tcp_ratio': 1.0,
                    'udp_ratio': 0.0,
                    'syn_ratio': 0.95,
                    'ack_ratio': 0.05,
                    'rst_ratio': 0.0,
                    'psh_ratio': 0.0,
                    'fin_ratio': 0.0,
                    'urg_ratio': 0.0
                }
            records.append(rec)

        return pd.DataFrame(records)[MODEL_FEATURE_COLS]
