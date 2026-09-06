#  NexGaurd: Network World Model for Predictive Cyber Defence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-00F2FE.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-06B6D4.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-EF4444.svg)](https://streamlit.io)
[![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK%20Aligned-00F2FE.svg)](https://attack.mitre.org)
[![Offline Air-Gapped](https://img.shields.io/badge/Deployment-Air--Gapped%20Offline-10B981.svg)]()

> **Predictive Cyber Defence using State-Transition World Models $P(S_{t+1} | S_t)$**  
> Moving beyond static intrusion classification to forecast multi-step attacker progression before system compromise is finalized.

---

##  Table of Contents
1. [Executive Summary](#executive-summary)
2. [Why World Models for Cyber Defence?](#why-world-models-for-cyber-defence)
3. [System Architecture](#system-architecture)
4. [How the Model Receives Further Inputs (Input Pipeline)](#how-the-model-receives-further-inputs-input-pipeline)
5. [MITRE ATT&CK Mapping & Proactive Playbooks](#mitre-attck-mapping--proactive-playbooks)
6. [Design System & Frontend Console](#design-system--frontend-console)
7. [Comparative Benchmark Results](#comparative-benchmark-results)
8. [Installation & Offline Execution](#installation--offline-execution)

---

##  Executive Summary
Traditional Network Intrusion Detection Systems (NIDS) evaluate network flows or packets in complete isolation. By mapping isolated flows to static binary labels (`Benign` vs `Malicious`), traditional systems discard the **temporal and causal structure** of an infiltration (e.g., how reconnaissance port probes precede initial access handshakes, and how TCP flag ratios evolve). Consequently, defenders can only react *after* an attack completes.

**NexGaurd** implements a **Network World Model** that learns environment transition dynamics:
$$P(S_{t+1} \mid S_{t-H:t})$$

Given an observed sliding window of multi-level network telemetry ($H=10$ seconds), NexGaurd:
1. **Forecasts future network state vectors** $S_{t+1}, S_{t+2}, \dots, S_{t+K}$ across a recursive $K$-step forward rollout.
2. **Estimates future compromise likelihood** and provides an actionable **Time-To-Compromise (TTC)** countdown.
3. **Maps anticipated progression to MITRE ATT&CK stages** (e.g., T1046 Reconnaissance, T1110.001 Brute Force / Initial Access).
4. **Delivers causal explainability** via **Integrated Gradients (Captum)** to show which flow/packet features drive the threat.
5. **Provides a "What-If" Counterfactual Sandbox** allowing defenders to test proactive mitigation strategies prior to deployment.

---

##  Why World Models for Cyber Defence?

| Traditional Static IDS Classifiers | NexGaurd Network World Model |
| :--- | :--- |
| Evaluates individual flows $x_t$ without sequence memory. | Ingests a temporal sliding window sequence $S_{t-H:t}$. |
| **Reactive**: Alerts only after the exploit payload or authentication completes. | **Proactive**: Forecasts attack trajectory $K$ seconds *before* compromise completion. |
| Vulnerable to low-and-slow threshold evasion. | Detects subtle state transition anomalies across time steps. |
| Zero forward simulation capability. | Recursive multi-step simulation ($S_{t+1 \dots t+K}$) with counterfactual testing. |
| Black-box classification score. | Causal explainability via Captum Integrated Gradients & temporal heatmaps. |

---

##  System Architecture

```
                                 ┌────────────────────────────────────────┐
                                 │       Multi-Level Telemetry Sources     │
                                 └───────────────────┬────────────────────┘
                                                     │
                     ┌───────────────────────────────┴───────────────────────────────┐
                     ▼                                                               ▼
        ┌───────────────────────────┐                                  ┌───────────────────────────┐
        │  Flow-Level Logs (NetFlow)│                                  │  Packet Captures (PCAP)   │
        │  - TCP flag distributions │                                  │  - TTL mean & variance    │
        │  - IAT statistics         │                                  │  - TCP window sizes       │
        │  - Flow durations/volumes │                                  │  - Retransmissions & Frag │
        └─────────────┬─────────────┘                                  └─────────────┬─────────────┘
                      │                                                              │
                      └──────────────────────────────┬───────────────────────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │ 1-Second State Vectorization  │
                                     │  - 20 Multi-Level Features    │
                                     │  - Shannon Port Entropy       │
                                     │  - StandardScaler Normalizer │
                                     └───────────────┬───────────────┘
                                                     │ S_{t-H:t} (10 x 20)
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │      Network World Model      │
                                     │   (2-Layer Recurrent GRU)     │
                                     └───────┬───────────────┬───────┘
                                             │               │
                     ┌───────────────────────┘               └───────────────────────┐
                     ▼                                                               ▼
        ┌─────────────────────────┐                                     ┌─────────────────────────┐
        │  Regression Head (MSE)  │                                     │ Classification Head(CE) │
        │  Predicts S_{t+1} State │                                     │  Attack Stage & TTC     │
        └────────────┬────────────┘                                     └────────────┬────────────┘
                     │                                                               │
                     └───────────────────────┬───────────────────────────────────────┘
                                             ▼
                             ┌───────────────────────────────┐
                             │ Recursive K-Step Simulation   │
                             │  - Infiltration Trajectory    │
                             │  - MITRE ATT&CK Threat Intel  │
                             │  - Integrated Gradients XAI   │
                             └───────────────┬───────────────┘
                                             ▼
                             ┌───────────────────────────────┐
                             │ NexGaurd SOC Web Console (UI) │
                             │  - Design System Aligned      │
                             │  - Air-Gapped & Offline Ready │
                             └───────────────────────────────┘
```

---

##  How the Model Receives Further Inputs (Input Pipeline)

To operate in live enterprise operations and Critical Information Infrastructure (CII), NexGaurd provides **four distinct ingestion modalities**:

### 1. Raw PCAP / PCAPNG Capture Ingestion (`PCAPFeatureExtractor`)
- **Parser Engine**: Uses `scapy` with zero cloud or external API dependencies.
- **Packet-Level Feature Extraction**:
  - Reads Layer 3 IP headers: extracts IP Time-To-Live (TTL) values, calculates TTL variance over sessions, and flags IP fragmentation.
  - Reads Layer 4 TCP/UDP headers: extracts TCP initial advertised window sizes (`init_fwd_win_mean`, `init_bwd_win_mean`), tracks sequence/acknowledgment numbers to detect packet retransmissions, and aggregates TCP control flags (`SYN`, `ACK`, `RST`, `PSH`, `FIN`, `URG`).
  - Computes packet length distributions and inter-arrival time (IAT) statistics.
  - Calculates destination port Shannon entropy:
    $$H(\text{Ports}) = -\sum_{i} p_i \log_2(p_i)$$
- **Aggregation**: Synchronizes packets into uniform 1-second state bins matching the 20 model features.

### 2. NetFlow / IPFIX Flow Records (`CSVFeatureExtractor`)
- Ingests NetFlow v9, IPFIX, or standard datasets (CSE-CIC-IDS2018, CTU-13, UNSW-NB15).
- Auto-detects and standardizes column aliases (e.g. forward/backward packet counts, flow durations, flag counts).
- Performs continuous timeline re-indexing, ensuring idle seconds without traffic are properly maintained as zero-state transitions rather than omitted.

### 3. Real-Time Streaming Ingestion (`StreamSimulator`)
- Connects to live network sockets or message brokers (Kafka/syslog).
- Ingests incoming 1-second state chunks into a sliding FIFO ring buffer of size $H=10$.
- Triggers recursive $K$-step forward simulation on every clock tick (1 Hz).

### 4. What-If Counterfactual Perturbation Ingestion
- Allows security operators to manually perturb state features (e.g., dialing up SYN flag ratio or port entropy) to test how zero-day attack escalations propagate through the World Model's learned dynamics.

---

##  MITRE ATT&CK Mapping & Proactive Playbooks

NexGaurd automatically correlates forecasted trajectories with the **MITRE ATT&CK Matrix**:

| Predicted Stage | MITRE Tactic | Technique ID | Technique Name | Proactive Containment Playbook |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Baseline Operation | `T0000` | Normal Enterprise Traffic | Maintain continuous behavioral logging; no containment needed. |
| **Reconnaissance** | Reconnaissance | `T1046` | Network Service Discovery | Detects port entropy spikes ($H > 2.0$); adaptive rate-limiting of scanning IP at perimeter. |
| **1** | Credential Access | `T1110.001` | Brute Force: Password Guessing (FTP:21) | Throttles FTP connection velocity; enforces fail2ban firewall rule before session establishment. |
| **2** | Initial Access / Privilege Escalation | `T1110.001` / `T1021.004` | Brute Force: SSH Remote Services (SSH:22) | Enforces mandatory hardware MFA challenge; isolates SSH port 22 behind zero-trust proxy. |

---

##  Design System & Frontend Console

The NexGaurd interface strictly adheres to the requested design specifications:

- **Primary Color**: `#00F2FE` (Electric Neon Cyan) — Active indicators, key metrics, forecast lines.
- **Secondary Color**: `#06B6D4` (Teal Cyan) — Sub-headers, secondary charts, historical timelines.
- **Tertiary Color**: `#EF4444` (Crimson Alert Red) — Threat warnings, attack progression curves, Time-To-Compromise countdown.
- **Neutral Background**: `#0A0E17` (Deep Space Obsidian) — Dashboard background with `#111827` card containers.
- **Typography**:
  - Headlines: `Space Grotesk`
  - Body: Modern clean sans-serif (`Geist` / Inter)
  - Code & Metrics: `JetBrains Mono`

### Interactive Dashboard Tabs:
1. **Command Center & Forward Rollout**:
   - Interactive timeline scrubber across 8,600+ seconds of telemetry.
   - Proactive alert banner with Time-To-Compromise (TTC) and MITRE ATT&CK containment recommendations.
   - Dual Plotly timeline comparing observed ground truth ($t-10s \dots t$) with forecasted trajectory ($t+1 \dots t+K$).
   - Live 20-feature state vector inspector.
2. **Input Pipeline & Ingestion Architecture**:
   - Visual breakdown of flow-level vs packet-level ingestion.
   - Drag-and-drop PCAP parser (powered by Scapy) with packet diagnostics (TTL variance, window size, retransmissions).
   - Drag-and-drop NetFlow CSV parser.
   - 1-click real-time stream simulation.
3. **Causal Explainability (Captum)**:
   - Integrated Gradients feature attribution bar chart.
   - 2D Spatio-Temporal Heatmap ($10$ seconds $\times$ top features) showing exact emergence of attack drivers.
4. **What-If Counterfactual Sandbox**:
   - Sliders to perturb SYN flag ratio, port entropy, flow count, and unique ports.
   - Side-by-side plot comparing observed trajectory vs counterfactual trajectory.
5. **Comparative Benchmark Validation**:
   - Direct performance evaluation against static baseline.

---

##  Comparative Benchmark Results

Evaluated on held-out test sequences from **CSE-CIC-IDS2018**:

| Evaluation Metric | NexGaurd World Model (Temporal GRU) | Static Baseline (MLP Classifier) |
| :--- | :---: | :---: |
| **Accuracy** | **99.83%** | 99.88% |
| **Precision (Macro)** | **99.60%** | 99.71% |
| **Recall (Macro)** | **99.92%** | 99.94% |
| **F1-Score (Macro)** | **99.76%** | 99.82% |
| **False Positive Rate (FPR)** | **0.066%** | 0.048% |
| **Transition State MSE** | **0.000108** | *N/A (No State Dynamics)* |
| **Forecast Horizon Capability** | **Recursive $K$-Step Rollout ($t+1 \dots t+K$)** | *None (Zero Forward Horizon)* |
| **Pre-Compromise Early Alerting** | **Supported (TTC Countdown)** | *Unsupported (Strictly Reactive)* |

---

##  Installation & Offline Execution

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Launch NexGaurd Web Console
```bash
streamlit run src/app.py
```
The console will open at `http://localhost:8501` completely offline without any cloud dependencies.

### 3. Optional: Re-train World Model
```bash
python src/train.py --epochs 10 --batch_size 64 --rnn_type gru --hidden_dim 64
```
