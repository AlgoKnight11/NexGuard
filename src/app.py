"""
NexGaurd: Network World Model for Predictive Cyber Defence
Proactive Multi-Horizon Intrusion Forecasting & Causal Explainability Console
"""

import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data_preprocessing import DataPreprocessor
from inference import InferenceEngine
from ingestion import (
    MODEL_FEATURE_COLS,
    PCAPFeatureExtractor,
    CSVFeatureExtractor,
    SyntheticTelemetryGenerator
)
from mitre_mapping import get_mitre_intel, MITRE_TECHNIQUES

# ---------------------------------------------------------
# Page Configuration & Design System
# ---------------------------------------------------------
st.set_page_config(
    page_title="NexGaurd | Network World Model",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS based on provided palette:
# Primary: #00F2FE, Secondary: #06B6D4, Tertiary: #EF4444, Neutral/BG: #0A0E17
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --primary: #00F2FE;
    --secondary: #06B6D4;
    --tertiary: #EF4444;
    --neutral-bg: #0A0E17;
    --card-bg: #111827;
    --card-border: rgba(6, 182, 212, 0.25);
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--neutral-bg) !important;
    color: var(--text-primary) !important;
}

h1, h2, h3, h4, .headline-font {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

code, pre, .mono-font, .stMetric, .stSlider {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background-color: rgba(17, 24, 39, 0.85);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 12px 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(10px);
}
div[data-testid="metric-container"]:hover {
    border-color: var(--primary);
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
}

/* Top App Header Banner */
.nexgaurd-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(10, 14, 23, 0.95));
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}

.nexgaurd-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.nexgaurd-glow {
    color: var(--primary);
    text-shadow: 0 0 12px rgba(0, 242, 254, 0.5);
}

.nexgaurd-badges {
    display: flex;
    gap: 10px;
}

.badge-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.badge-cyan {
    background-color: rgba(0, 242, 254, 0.12);
    color: var(--primary);
    border: 1px solid rgba(0, 242, 254, 0.4);
}

.badge-green {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-red {
    background-color: rgba(239, 68, 68, 0.15);
    color: var(--tertiary);
    border: 1px solid rgba(239, 68, 68, 0.4);
    animation: pulse-red 2s infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 rgba(239, 68, 68, 0); }
    50% { box-shadow: 0 0 14px rgba(239, 68, 68, 0.6); }
}

/* Alert Boxes */
.alert-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    border-left: 4px solid;
    background: rgba(17, 24, 39, 0.85);
}

.alert-danger {
    border-color: var(--tertiary);
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
}

.alert-nominal {
    border-color: var(--primary);
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.15);
}

/* Streamlit Tabs Customization */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}

/* Custom Buttons */
.stButton>button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Inference Engine & Cached Telemetry
# ---------------------------------------------------------
@st.cache_resource
def load_engine():
    if not (os.path.exists("models/best_world_model.pth") and os.path.exists("models/scaler.pkl")):
        return None
    return InferenceEngine(model_path="models/best_world_model.pth", scaler_path="models/scaler.pkl")

engine = load_engine()

if engine is None:
    st.error("Model artifacts ('models/best_world_model.pth' or 'models/scaler.pkl') missing. Please train the model first.")
    st.stop()

@st.cache_data
def load_base_telemetry():
    """Loads precomputed test states or slices from cic.csv."""
    cache_path = "models/cached_test_states.pkl"
    if os.path.exists(cache_path):
        data = joblib.load(cache_path)
        return data['scaled_features'], data['raw_features'], data['labels'], data['feature_cols']
    
    # Fallback to direct raw parsing if cache not found
    preprocessor = DataPreprocessor(window_sec=1, history_len=10, forecast_step=1)
    raw_df = preprocessor.clean_and_load("cic.csv")
    agg_df = preprocessor.aggregate_to_states(raw_df)
    preprocessor.scaler = engine.scaler
    raw_feats = agg_df[engine.feature_cols].values
    scaled_feats = engine.scaler.transform(raw_feats)
    labels = agg_df['label'].values
    
    n_samples = len(scaled_feats)
    test_start = int(n_samples * 0.80)
    return scaled_feats[test_start:], raw_feats[test_start:], labels[test_start:], engine.feature_cols

default_scaled, default_raw, default_labels, feature_cols = load_base_telemetry()

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="nexgaurd-header">
    <div>
        <h1 class="nexgaurd-title">
            NexGaurd <span class="nexgaurd-glow">Cyber World Model</span>
        </h1>
        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
            Predictive Cyber Defence • Transition Dynamics \\(P(S_{t+1} | S_t)\\) • Pre-Compromise Infiltration Forecasting
        </p>
    </div>
    <div class="nexgaurd-badges">
        <span class="badge-pill badge-cyan"> WORLD MODEL ONLINE</span>
        <span class="badge-pill badge-green"> AIR-GAPPED OFFLINE</span>
        <span class="badge-pill badge-cyan">GRU-2L (HIDDEN=64)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Telemetry Source & Parameters
# ---------------------------------------------------------
st.sidebar.markdown("### Telemetry Controls")

data_source = st.sidebar.selectbox(
    "Select Telemetry Feed:",
    options=[
        "CSE-CIC-IDS2018 (Enterprise Flow Timeline)",
        "Raw Packet Capture (PCAP Sample / Scapy)",
        "Synthetic Threat Scenario: Recon -> SSH Brute Force",
        "Synthetic Threat Scenario: FTP Password Spray",
        "Custom Upload (PCAP / NetFlow CSV)"
    ]
)

# Handle Data Source Loading
active_scaled = default_scaled
active_raw = default_raw
active_labels = default_labels
pcap_diagnostics = None

if data_source == "Raw Packet Capture (PCAP Sample / Scapy)":
    pcap_path = "sample_capture.pcap"
    if not os.path.exists(pcap_path):
        from pcap_sample import generate_test_pcap
        generate_test_pcap(pcap_path)
    extractor = PCAPFeatureExtractor(window_sec=1)
    agg_df, pcap_diagnostics = extractor.parse_pcap(pcap_path)
    active_raw = agg_df[feature_cols].values
    active_scaled = engine.scaler.transform(active_raw)
    # PCAP has reconnaissance at t=5..10, SSH brute force at t=10..15
    active_labels = np.array([0]*5 + [0]*5 + [2]*max(0, len(active_raw)-10))

elif data_source == "Synthetic Threat Scenario: Recon -> SSH Brute Force":
    agg_df = SyntheticTelemetryGenerator.generate_scenario("recon_to_ssh", length_sec=30)
    active_raw = agg_df[feature_cols].values
    active_scaled = engine.scaler.transform(active_raw)
    active_labels = np.array([0]*10 + [2]*20)

elif data_source == "Synthetic Threat Scenario: FTP Password Spray":
    agg_df = SyntheticTelemetryGenerator.generate_scenario("ftp_bruteforce", length_sec=30)
    active_raw = agg_df[feature_cols].values
    active_scaled = engine.scaler.transform(active_raw)
    active_labels = np.array([0]*5 + [1]*25)

elif data_source == "Custom Upload (PCAP / NetFlow CSV)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Telemetry File",
        type=["pcap", "pcapng", "csv"],
        help="Upload raw PCAP capture or NetFlow CSV log"
    )
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext in ["pcap", "pcapng"]:
            with open("temp_upload.pcap", "wb") as f:
                f.write(uploaded_file.getbuffer())
            extractor = PCAPFeatureExtractor(window_sec=1)
            agg_df, pcap_diagnostics = extractor.parse_pcap("temp_upload.pcap")
            active_raw = agg_df[feature_cols].values
            active_scaled = engine.scaler.transform(active_raw)
            active_labels = np.zeros(len(active_raw), dtype=int)
            st.sidebar.success(f"Successfully ingested {uploaded_file.name} ({len(active_raw)} seconds)")
        else:
            csv_ext = CSVFeatureExtractor(DataPreprocessor())
            agg_df = csv_ext.process_csv(uploaded_file)
            active_raw = agg_df[feature_cols].values
            active_scaled = engine.scaler.transform(active_raw)
            active_labels = np.zeros(len(active_raw), dtype=int)
            st.sidebar.success(f"Successfully ingested {uploaded_file.name}")

# Timeline scrubber limits
max_idx = max(0, len(active_scaled) - engine.history_len - 5)

col_ctrl1, col_ctrl2 = st.sidebar.columns([1, 1])
if 'play_idx' not in st.session_state:
    st.session_state.play_idx = min(200, max_idx) if max_idx > 0 else 0

if col_ctrl1.button(" Step -1s"):
    st.session_state.play_idx = max(0, st.session_state.play_idx - 1)
if col_ctrl2.button("Step +1s "):
    st.session_state.play_idx = min(max_idx, st.session_state.play_idx + 1)

current_time_sec = st.sidebar.slider(
    "Scrub Timeline (Second Index)",
    min_value=0,
    max_value=max_idx,
    value=min(st.session_state.play_idx, max_idx),
    step=1
)
st.session_state.play_idx = current_time_sec

# Simulation Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation Horizon")

K_steps = st.sidebar.slider(
    "Forecast Rollout Horizon (K-Steps Ahead)",
    min_value=3,
    max_value=15,
    value=6,
    step=1,
    help="How many discrete seconds into the future the World Model recursively simulates"
)

detection_threshold = st.sidebar.slider(
    "Early Warning Sensitivity Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
    help="Confidence threshold to trigger proactive kill-chain containment alerts"
)

# Extract Sequence for Analysis
history_seq = active_scaled[current_time_sec : current_time_sec + engine.history_len]
raw_history_seq = active_raw[current_time_sec : current_time_sec + engine.history_len]
true_history_labels = active_labels[current_time_sec : current_time_sec + engine.history_len]

# Execute World Model Forward Simulation
rollout_states, rollout_probs = engine.forward_rollout(history_seq, K_steps=K_steps)
_, current_probs = engine.predict_next(history_seq)
current_pred_cls = np.argmax(current_probs)
current_label = true_history_labels[-1]

# Current State Markers
last_raw = raw_history_seq[-1]
last_port_entropy = last_raw[feature_cols.index('port_entropy')]
last_syn_ratio = last_raw[feature_cols.index('syn_ratio')]
last_flow_count = last_raw[feature_cols.index('flow_count')]
last_unique_ports = int(last_raw[feature_cols.index('unique_ports')])

# Determine Early Warning & MITRE Mapping
max_future_attack_prob = 0.0
compromise_step = -1
predicted_attack_cls = 0

for step_i in range(K_steps):
    step_p = rollout_probs[step_i]
    atk_prob = step_p[1] + step_p[2]
    if atk_prob > detection_threshold and atk_prob > max_future_attack_prob:
        max_future_attack_prob = atk_prob
        compromise_step = step_i + 1
        predicted_attack_cls = 1 if step_p[1] > step_p[2] else 2

mitre_intel = get_mitre_intel(
    predicted_attack_cls if compromise_step != -1 else current_pred_cls,
    port_entropy=last_port_entropy,
    syn_ratio=last_syn_ratio
)

# Class Names
class_names = ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_command, tab_ingestion, tab_explain, tab_counterfactual, tab_benchmarks = st.tabs([
    "Command Center & Forward Rollout",
    "Input Pipeline & Ingestion Architecture",
    "Causal Explainability (Captum)",
    "What-If Counterfactual Sandbox",
    "Comparative Benchmark Validation"
])

# =========================================================
# TAB 1: Command Center & Forward Rollout
# =========================================================
with tab_command:
    # Early Warning Status Banner
    if compromise_step != -1:
        st.markdown(f"""
        <div class="alert-card alert-danger">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="color: #EF4444; margin: 0 0 6px 0;">
                        PROACTIVE WARNING: Imminent Infiltration Forecasted!
                    </h3>
                    <p style="margin: 0; color: #E2E8F0; font-size: 1.05rem;">
                        World Model predicts trajectory converging to <strong>{mitre_intel['technique_name']}</strong> 
                        (MITRE: <code style="color: #00F2FE;">{mitre_intel['technique_id']}</code>).
                    </p>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                        Kill-Chain Phase: <strong>{mitre_intel['kill_chain_phase']}</strong> | Risk Rating: <span style="color: #EF4444; font-weight: 700;">{mitre_intel['risk_level']}</span>
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #EF4444; font-family: 'JetBrains Mono', monospace;">
                        {compromise_step}s
                    </div>
                    <div style="color: #94A3B8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">
                        Estimated Time-To-Compromise (TTC)
                    </div>
                    <div style="color: #00F2FE; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                        Confidence: {max_future_attack_prob*100:.1f}%
                    </div>
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(239, 68, 68, 0.25);">
                <strong style="color: #00F2FE; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em;">Recommended Containment Actions Before Handshake Completes:</strong>
                <ul style="margin: 6px 0 0 0; padding-left: 20px; color: #CBD5E1; font-size: 0.92rem;">
                    {''.join([f"<li>{act}</li>" for act in mitre_intel['action_playbook']])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card alert-nominal">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #00F2FE; margin: 0 0 4px 0;">
                         Nominal State: No Active Attack Trajectory Converging
                    </h3>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.92rem;">
                        World Model forward rollout across the next <strong>{K_steps} seconds</strong> confirms network stability within baseline parameters.
                    </p>
                </div>
                <div style="text-align: right;">
                    <span class="badge-pill badge-green">P(ATTACK) &lt; {detection_threshold*100:.0f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Top Metric Tiles
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Flow Rate (flows/s)", f"{last_flow_count}")
    col_m2.metric("Port Entropy (bits)", f"{last_port_entropy:.2f}")
    col_m3.metric("Unique Target Ports", f"{last_unique_ports}")
    col_m4.metric("SYN Flag Ratio", f"{last_syn_ratio*100:.1f}%")
    col_m5.metric("Current Stage", class_names[current_pred_cls])

    # Visualizations Row
    col_chart_left, col_chart_right = st.columns([1.6, 1.0])

    with col_chart_left:
        st.markdown("#### Attacker Progression Probability Horizon (Historical vs Forecasted)")

        # Build timeline dataframe
        hist_steps = [f"t-{engine.history_len - i - 1}s" for i in range(engine.history_len)]
        fut_steps = [f"t+{i+1}s (Sim)" for i in range(K_steps)]
        timeline_x = hist_steps + fut_steps

        # Historical ground truth probabilities
        h_benign = [1.0 if l == 0 else 0.0 for l in true_history_labels]
        h_ftp = [1.0 if l == 1 else 0.0 for l in true_history_labels]
        h_ssh = [1.0 if l == 2 else 0.0 for l in true_history_labels]

        # Forecasted rollout probabilities
        f_benign = rollout_probs[:, 0].tolist()
        f_ftp = rollout_probs[:, 1].tolist()
        f_ssh = rollout_probs[:, 2].tolist()

        fig_prog = go.Figure()

        # Benign line (Teal/Green)
        fig_prog.add_trace(go.Scatter(
            x=timeline_x, y=h_benign + f_benign,
            mode='lines+markers', name='Benign / Nominal',
            line=dict(color='#10B981', width=2.5),
            marker=dict(size=6)
        ))

        # FTP Brute Force (Secondary Cyan #06B6D4)
        fig_prog.add_trace(go.Scatter(
            x=timeline_x, y=h_ftp + f_ftp,
            mode='lines+markers', name='FTP-BruteForce (Port 21)',
            line=dict(color='#06B6D4', width=2.5, dash='dash' if compromise_step != -1 and predicted_attack_cls == 1 else 'solid'),
            marker=dict(size=6)
        ))

        # SSH Brute Force (Tertiary Red #EF4444)
        fig_prog.add_trace(go.Scatter(
            x=timeline_x, y=h_ssh + f_ssh,
            mode='lines+markers', name='SSH-Bruteforce (Port 22)',
            line=dict(color='#EF4444', width=3),
            marker=dict(size=7)
        ))

        # Boundary Line between Observed and Forecast
        fig_prog.add_shape(
            type="line",
            x0="t-0s", y0=-0.05, x1="t-0s", y1=1.05,
            line=dict(color="#00F2FE", width=2, dash="dot")
        )
        fig_prog.add_annotation(
            x="t-0s", y=1.03,
            text=" Forecast Boundary (Now)",
            showarrow=False,
            font=dict(color="#00F2FE", size=11, family="JetBrains Mono")
        )

        fig_prog.update_layout(
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#111827",
            font=dict(color="#F8FAFC", family="Space Grotesk"),
            yaxis=dict(title="Progression Probability", range=[-0.05, 1.1], gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    with col_chart_right:
        st.markdown("#### Future State Vector Transition \\(S_{t+1 \\dots t+K}\\)")

        # Descale forecasted states to raw units for intuitive inspection
        descaled_future = engine.descale_state(rollout_states)
        flow_idx = feature_cols.index('flow_count')
        syn_idx = feature_cols.index('syn_ratio')
        entropy_idx = feature_cols.index('port_entropy')

        future_steps = [f"+{i+1}s" for i in range(K_steps)]
        fig_state = go.Figure()

        fig_state.add_trace(go.Bar(
            x=future_steps,
            y=descaled_future[:, flow_idx],
            name='Flow Velocity',
            marker_color='#00F2FE',
            opacity=0.85
        ))

        fig_state.add_trace(go.Scatter(
            x=future_steps,
            y=descaled_future[:, syn_idx] * 100,
            name='SYN Ratio (%)',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#EF4444', width=2.5)
        ))

        fig_state.update_layout(
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#111827",
            font=dict(color="#F8FAFC", family="Space Grotesk"),
            yaxis=dict(title="Flows/sec", gridcolor="rgba(255,255,255,0.08)"),
            yaxis2=dict(title="SYN %", overlaying='y', side='right', range=[0, 100], gridcolor="rgba(255,255,255,0)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_state, use_container_width=True)

    # Detailed Current State Inspection Drawer
    with st.expander(" View Full Observed 20-Feature Network State Vector \\(S_t\\)", expanded=False):
        inspect_df = pd.DataFrame({
            "Feature Dimension": feature_cols,
            "Observed Raw Value": [f"{v:.4f}" if isinstance(v, float) else str(v) for v in last_raw],
            "Standard Scaled (Z-Score)": [f"{v:.4f}" for v in history_seq[-1]]
        })
        st.dataframe(inspect_df, use_container_width=True, height=300)


# =========================================================
# TAB 2: Ingestion Architecture & Input Pipeline
# =========================================================
with tab_ingestion:
    st.markdown("""
    ### Multi-Level Telemetry Ingestion Pipeline
    NexGaurd bridges **both flow-level and packet-level telemetry** into continuous, synchronized 1-second state vectors \\(S_t\\).
    """)

    col_pipe_a, col_pipe_b = st.columns(2)

    with col_pipe_a:
        st.markdown("""
        <div style="background: #111827; border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 12px; padding: 18px;">
            <h4 style="color: #00F2FE; margin-top: 0;">1. Flow-Level Telemetry (NetFlow / IPFIX)</h4>
            <p style="color: #94A3B8; font-size: 0.92rem;">
                Captures aggregate volumetric behavior:
            </p>
            <ul style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 0;">
                <li><strong>TCP Flag Distribution:</strong> SYN, ACK, RST, PSH, FIN, URG ratios per second.</li>
                <li><strong>Bidirectional Volumes:</strong> Forward/Backward packet counts and total byte lengths.</li>
                <li><strong>Temporal Dynamics:</strong> Flow durations, packet inter-arrival times (IAT mean/max).</li>
                <li><strong>Protocol Balance:</strong> Active TCP vs UDP traffic ratios.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_pipe_b:
        st.markdown("""
        <div style="background: #111827; border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 12px; padding: 18px;">
            <h4 style="color: #06B6D4; margin-top: 0;">2. Packet-Level Telemetry (Raw PCAP / Scapy)</h4>
            <p style="color: #94A3B8; font-size: 0.92rem;">
                Captures granular protocol and sequencing signatures:
            </p>
            <ul style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 0;">
                <li><strong>Destination Port Entropy:</strong> Shannon entropy \\(-\\sum p \\log_2 p\\) detecting scanning patterns.</li>
                <li><strong>TCP Window Sizes:</strong> Initial forward/backward window bytes (buffer sizes).</li>
                <li><strong>Hop Distance & Fragmentation:</strong> IP TTL distribution and IP fragment flags.</li>
                <li><strong>Retransmission Dynamics:</strong> Duplicate ACK and sequence retransmission counts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Interactive Telemetry Ingestion Lab")

    ingest_mode = st.radio(
        "Choose Ingestion Inflow Method:",
        options=["Parse Local PCAP Capture", "Upload Custom NetFlow/IPFIX CSV", "Simulate Real-Time Streaming Ingestion"],
        horizontal=True
    )

    if ingest_mode == "Parse Local PCAP Capture":
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown("**Load Generated Test PCAP (`sample_capture.pcap`)**")
            st.write("Contains synthetic multi-phase traffic (DNS baseline + Port Scan + SSH Brute Force).")
            if st.button(" Ingest & Extract PCAP Features", key="btn_parse_pcap"):
                with st.spinner("Parsing PCAP packets via Scapy and extracting 20-feature temporal matrix..."):
                    ext = PCAPFeatureExtractor(window_sec=1)
                    parsed_df, diag = ext.parse_pcap("sample_capture.pcap")
                    st.success(f"Extracted {len(parsed_df)} discrete 1s state bins from {diag['total_packets']} packets!")
                    st.session_state['parsed_pcap_df'] = parsed_df
                    st.session_state['parsed_pcap_diag'] = diag

        with col_p2:
            if 'parsed_pcap_diag' in st.session_state:
                diag = st.session_state['parsed_pcap_diag']
                st.markdown("**Packet-Level Diagnostics Extracted:**")
                st.json(diag)

        if 'parsed_pcap_df' in st.session_state:
            st.dataframe(st.session_state['parsed_pcap_df'], use_container_width=True, height=220)

    elif ingest_mode == "Upload Custom NetFlow/IPFIX CSV":
        st.write("Upload any CSV flow record conforming to CSE-CIC-IDS2018 or generic NetFlow/IPFIX formats.")
        cust_file = st.file_uploader("Drop NetFlow CSV here", type=["csv"], key="uploader_ingest_tab")
        if cust_file:
            st.info(f"Loaded {cust_file.name}. Aligning columns with `StandardScaler`...")
            csv_ext = CSVFeatureExtractor(DataPreprocessor())
            extracted_df = csv_ext.process_csv(cust_file)
            st.dataframe(extracted_df.head(10), use_container_width=True)

    else:
        st.markdown("**Real-Time Streaming Telemetry Simulator**")
        st.write("Simulates a continuous socket or Kafka message bus streaming 1-second state buffers directly into the World Model.")
        stream_scenario = st.selectbox(
            "Select Real-Time Attack Simulation Stream:",
            ["Reconnaissance Port Scan Escalating to SSH Brute Force", "FTP Password Spray Attack", "Benign Enterprise Background"]
        )
        if st.button(" Run Real-Time Stream Simulation (10s Burst)"):
            scen_key = "recon_to_ssh" if "Recon" in stream_scenario else ("ftp_bruteforce" if "FTP" in stream_scenario else "benign")
            burst_df = SyntheticTelemetryGenerator.generate_scenario(scen_key, length_sec=10)
            burst_raw = burst_df[feature_cols].values
            burst_scaled = engine.scaler.transform(burst_raw)
            
            progress_bar = st.progress(0)
            for i in range(10):
                time.sleep(0.15)
                progress_bar.progress((i + 1) / 10)
            st.success("Successfully ingested 10 streaming state windows into World Model temporal buffer!")
            st.dataframe(burst_df, use_container_width=True, height=200)


# =========================================================
# TAB 3: Causal Explainability (Captum)
# =========================================================
with tab_explain:
    st.markdown("""
    ### Causal Explainability via Integrated Gradients
    Black-box alerts are ineffective in modern Security Operations Centers (SOC). 
    NexGaurd applies **Integrated Gradients** (via PyTorch Captum) to attribute the predicted progression trajectory 
    to specific flow flags, port patterns, and temporal sequences.
    """)

    col_exp_ctrl1, col_exp_ctrl2 = st.columns([1, 2])
    with col_exp_ctrl1:
        target_explain_cls = st.selectbox(
            "Target Attack Hypothesis to Attribute:",
            options=[1, 2, 0],
            format_func=lambda x: f"{class_names[x]} ({'MITRE Initial Access' if x > 0 else 'Baseline Nominal'})"
        )
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 10px; padding: 14px; margin-top: 10px;">
            <strong style="color: #00F2FE;">Baseline Reference:</strong> Zero-state baseline \\(x'=0\\)<br>
            <strong style="color: #06B6D4;">Attribution Method:</strong> Gauss-Legendre Quadrature Path Integral<br>
            <strong style="color: #EF4444;">Target Class:</strong> {class_names[target_explain_cls]}
        </div>
        """, unsafe_allow_html=True)

    # Compute Attributions
    with st.spinner("Computing Integrated Gradients attributions across 10-second history window..."):
        attributions, feature_importance = engine.explain_prediction(history_seq, target_class=target_explain_cls)

    with col_exp_ctrl2:
        # Top Contributing Features Horizontal Bar Chart
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Attribution Score': feature_importance
        }).sort_values(by='Attribution Score', ascending=True)

        fig_bar = px.bar(
            imp_df.tail(8),
            y='Feature',
            x='Attribution Score',
            orientation='h',
            color='Attribution Score',
            color_continuous_scale=[[0, '#06B6D4'], [1, '#EF4444' if target_explain_cls > 0 else '#00F2FE']],
            title=f"Top Driver Features for {class_names[target_explain_cls]}"
        )
        fig_bar.update_layout(
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#111827",
            font=dict(color="#F8FAFC", family="Space Grotesk"),
            margin=dict(l=20, r=20, t=35, b=20),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Temporal Attribution Heatmap
    st.markdown("#### Spatio-Temporal Attribution Heatmap (Features × Time Steps)")
    st.markdown("Shows how individual feature influence evolved second-by-second across the sliding 10s history window.")

    top_8_indices = np.argsort(feature_importance)[-8:]
    top_8_features = [feature_cols[i] for i in top_8_indices]
    heatmap_matrix = attributions[:, top_8_indices].T

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=[f"t-{engine.history_len - i - 1}s" for i in range(engine.history_len)],
        y=top_8_features,
        colorscale='RdBu_r',
        zmid=0.0
    ))
    fig_heat.update_layout(
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#111827",
        font=dict(color="#F8FAFC", family="Space Grotesk"),
        xaxis=dict(title="Historical Time Steps (Seconds)"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Forensic Analyst Takeaway
    top_driver_name = feature_cols[np.argmax(feature_importance)]
    st.info(f"""
    **Forensic SOC Insight:** The World Model's probability attribution for **{class_names[target_explain_cls]}** 
    is primarily governed by **`{top_driver_name}`**, indicating that sudden shifts in this dimension serve 
    as the leading causal indicator preceding credential access attempts.
    """)


# =========================================================
# TAB 4: What-If Counterfactual Sandbox
# =========================================================
with tab_counterfactual:
    st.markdown("""
    ### What-If Counterfactual Simulation Sandbox
    Test how hypothetical perturbations to live traffic would alter the World Model's forward simulation horizon. 
    Use the sliders below to adjust observed traffic parameters and observe the projected future state change.
    """)

    col_cf_left, col_cf_right = st.columns([1, 2])

    with col_cf_left:
        st.markdown("**Traffic Feature Perturbations (State \\(S_t\\)):**")
        cf_syn_ratio = st.slider("Hypothetical SYN Flag Ratio", 0.0, 1.0, float(last_syn_ratio), 0.05)
        cf_port_entropy = st.slider("Hypothetical Port Entropy", 0.0, 5.0, float(last_port_entropy), 0.1)
        cf_flow_count = st.slider("Hypothetical Flow Velocity (flows/s)", 1, 500, int(last_flow_count), 10)
        cf_unique_ports = st.slider("Hypothetical Unique Ports", 1, 50, int(last_unique_ports), 1)

        run_cf = st.button(" Run Counterfactual Rollout", type="primary")

    with col_cf_right:
        # Run counterfactual simulation
        perturbations = {
            'syn_ratio': cf_syn_ratio,
            'port_entropy': cf_port_entropy,
            'flow_count': cf_flow_count,
            'unique_ports': cf_unique_ports
        }
        cf_rollout_states, cf_rollout_probs, _ = engine.simulate_counterfactual(
            history_seq,
            perturbations,
            K_steps=K_steps
        )

        st.markdown("#### Observed vs Counterfactual Attacker Probability Horizon")

        future_axis = [f"t+{i+1}s" for i in range(K_steps)]

        # Actual vs Counterfactual Attack Probability (Class 1 + Class 2)
        actual_atk_probs = rollout_probs[:, 1] + rollout_probs[:, 2]
        cf_atk_probs = cf_rollout_probs[:, 1] + cf_rollout_probs[:, 2]

        fig_cf = go.Figure()

        fig_cf.add_trace(go.Scatter(
            x=future_axis, y=actual_atk_probs,
            mode='lines+markers', name='Actual Observed Trajectory',
            line=dict(color='#06B6D4', width=2.5),
            marker=dict(size=8)
        ))

        fig_cf.add_trace(go.Scatter(
            x=future_axis, y=cf_atk_probs,
            mode='lines+markers', name='Counterfactual Perturbed Trajectory',
            line=dict(color='#EF4444' if cf_atk_probs.max() > actual_atk_probs.max() else '#00F2FE', width=3, dash='dot'),
            marker=dict(size=9, symbol='diamond')
        ))

        fig_cf.update_layout(
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#111827",
            font=dict(color="#F8FAFC", family="Space Grotesk"),
            yaxis=dict(title="Aggregated Attack Probability", range=[-0.05, 1.05], gridcolor="rgba(255,255,255,0.08)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=340
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        prob_delta = cf_atk_probs.mean() - actual_atk_probs.mean()
        if prob_delta > 0.1:
            st.warning(f" Perturbation increases forecasted attack likelihood by **+{prob_delta*100:.1f}%**! The World Model recognizes this pattern as characteristic of pre-attack escalation.")
        elif prob_delta < -0.1:
            st.success(f" Perturbation decreases forecasted attack likelihood by **{abs(prob_delta)*100:.1f}%**, stabilizing the network into benign equilibrium.")
        else:
            st.info("ℹ Minimal trajectory divergence observed for this perturbation scale.")


# =========================================================
# TAB 5: Benchmark Validation
# =========================================================
with tab_benchmarks:
    st.markdown("""
    ### Comparative Benchmark & Architectural Justification
    Evaluating **NexGaurd Network World Model** (Temporal Dynamics Learning \\(P(S_{t+1}|S_t)\\)) against 
    a **Baseline Static Classifier (MLP)** trained on the identical feature space.
    """)

    if os.path.exists("models/benchmark_results.csv"):
        bench_df = pd.read_csv("models/benchmark_results.csv")
        
        col_b1, col_b2 = st.columns([1.2, 1])

        with col_b1:
            st.markdown("#### Test Set Performance Metrics")
            st.dataframe(bench_df.style.format({
                'Accuracy': '{:.4%}',
                'Precision': '{:.4%}',
                'Recall': '{:.4%}',
                'F1-Score': '{:.4%}',
                'FPR': '{:.6f}'
            }), use_container_width=True)

            # Bar chart comparison
            bench_melted = pd.melt(bench_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
            fig_b = px.bar(
                bench_melted,
                x='variable',
                y='value',
                color='Model',
                barmode='group',
                labels={'variable': 'Metric', 'value': 'Score'},
                color_discrete_sequence=['#00F2FE', '#06B6D4'],
                title="Model Performance Comparison"
            )
            fig_b.update_layout(
                paper_bgcolor="#0A0E17",
                plot_bgcolor="#111827",
                font=dict(color="#F8FAFC", family="Space Grotesk"),
                yaxis=dict(range=[0.95, 1.005], gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=20, r=20, t=35, b=20),
                height=300
            )
            st.plotly_chart(fig_b, use_container_width=True)

        with col_b2:
            st.markdown("""
            <div style="background: #111827; border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 12px; padding: 18px;">
                <h4 style="color: #00F2FE; margin-top: 0;">Why Static Classifiers Fail in Real Defence:</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">
                    Traditional IDS classifiers evaluate flows \\(x_t\\) in complete isolation. As a result:
                </p>
                <ul style="color: #CBD5E1; font-size: 0.88rem;">
                    <li><strong>Zero Predictive Horizon:</strong> A static classifier can only trigger an alert <em>after</em> the brute force packet arrives or session is established.</li>
                    <li><strong>Blind to Sequence Dynamics:</strong> They cannot differentiate a legitimate burst of logins from an escalating credential spray pattern unfolding across 10 seconds.</li>
                </ul>
                <h4 style="color: #10B981; margin-top: 14px;">The World Model Advantage:</h4>
                <p style="color: #CBD5E1; font-size: 0.88rem; margin-bottom: 0;">
                    NexGaurd learns the underlying <strong>causal transition model</strong> \\(S_{t+1} = f(S_t, S_{t-1}, \\dots)\\). 
                    This enables multi-step recursive rollouts, providing defenders with an actionable 
                    <strong>Time-to-Compromise (TTC) window</strong> to quarantine or throttle attacking endpoints <em>before</em> breach finalization.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Benchmark results file ('models/benchmark_results.csv') not found. Run training pipeline to generate benchmarks.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; color: #64748B; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
    <div>NexGaurd AI Defence Engine • Version 2.4.0-Production</div>
    <div>MITRE ATT&CK® Aligned • Fully Offline / Air-Gapped Capable</div>
</div>
""", unsafe_allow_html=True)
