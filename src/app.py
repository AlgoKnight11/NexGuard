import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import os

from data_preprocessing import DataPreprocessor
from inference import InferenceEngine

# Set Page Config
st.set_page_config(
    page_title="Network World Model Dashboard",
    page_icon="",
    layout="wide"
)

st.title("Network World Model for Predictive Cyber Defence")
st.markdown("""
This interface demonstrates a **World Model** AI system that learns network traffic transition dynamics 
and simulates attacker progression stages over a future time horizon.
""")

# Check if model files exist
model_exists = os.path.exists("models/best_world_model.pth") and os.path.exists("models/scaler.pkl")

if not model_exists:
    st.warning("**Trained model weights or scaler not found!**")
    st.info("""
    Before running inference, please train the World Model and Baseline using the dataset.
    Run the following command in your terminal:
    ```bash
    python src/train.py --epochs 10 --batch_size 128
    ```
    """)
    st.stop()

# Initialize Inference Engine
@st.cache_resource
def get_inference_engine():
    return InferenceEngine(model_path="models/best_world_model.pth", scaler_path="models/scaler.pkl")

try:
    engine = get_inference_engine()
except Exception as e:
    st.error(f"Error loading inference engine: {e}")
    st.stop()

# Load Test Data Segment for simulation
@st.cache_data
def load_test_data_segment(csv_path="cic.csv"):
    try:
        # Load and preprocess using the saved preprocessor parameters
        preprocessor = DataPreprocessor(
            window_sec=engine.window_sec,
            history_len=engine.history_len,
            forecast_step=1
        )
        # Load a subset of raw CSV to keep load times snappy in UI
        st.write("Loading test data segment for interactive simulation...")
        raw_df = preprocessor.clean_and_load(csv_path)
        agg_df = preprocessor.aggregate_to_states(raw_df)
        
        # Fit-transform scaler (using the same scaler as engine)
        preprocessor.scaler = engine.scaler
        features = agg_df[engine.feature_cols].values
        scaled_features = engine.scaler.transform(features)
        labels = agg_df['label'].values
        
        # Get test segment (last 15% of the timeline)
        n_samples = len(scaled_features)
        test_start_idx = int(n_samples * 0.85)
        
        test_feats = scaled_features[test_start_idx:]
        test_labels = labels[test_start_idx:]
        raw_test_feats = features[test_start_idx:]
        
        return test_feats, test_labels, raw_test_feats, engine.feature_cols
    except Exception as e:
        st.error(f"Error loading test data segment: {e}")
        return None, None, None, None

test_feats, test_labels, raw_test_feats, feature_cols = load_test_data_segment()

if test_feats is None:
    st.stop()

# Sidebar Configuration
st.sidebar.header("Simulation Controls")

# Selection index
max_idx = len(test_feats) - engine.history_len - 5
selected_idx = st.sidebar.slider(
    "Select Simulation Start Second",
    min_value=0,
    max_value=max_idx,
    value=min(200, max_idx),
    step=1
)

# Rollout parameters
K_steps = st.sidebar.slider(
    "Forecast Horizon (K-steps ahead)",
    min_value=3,
    max_value=15,
    value=5,
    step=1
)

detection_threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# Tabs
tab_predict, tab_explain, tab_benchmarks = st.tabs([
    "Infiltration Prediction & Rollout",
    "Explainability & Attributions",
    "Benchmarks & Performance"
])

# Map labels to classes
class_names = ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]
mitre_stages = {
    0: "Benign (No Malicious Activity)",
    1: "Credential Access / Initial Access (FTP Brute Force)",
    2: "Credential Access / Initial Access (SSH Brute Force)"
}

# Extract selected sequence
history_seq = test_feats[selected_idx : selected_idx + engine.history_len]
raw_history_seq = raw_test_feats[selected_idx : selected_idx + engine.history_len]
true_history_labels = test_labels[selected_idx : selected_idx + engine.history_len]

# Run Forward Rollout
rollout_states, rollout_probs = engine.forward_rollout(history_seq, K_steps=K_steps)

# Current State Prediction
_, current_probs = engine.predict_next(history_seq)
current_pred_cls = np.argmax(current_probs)
current_label = true_history_labels[-1]

with tab_predict:
    st.subheader("Simulated Timeline & Prediction Rollout")
    
    # Alert Box
    # Check if future predictions cross detection threshold for malicious classes (1 and 2)
    max_attack_prob = 0.0
    compromise_step = -1
    compromise_cls = 0
    
    for step in range(K_steps):
        step_probs = rollout_probs[step]
        attack_prob = step_probs[1] + step_probs[2]
        if attack_prob > detection_threshold and attack_prob > max_attack_prob:
            max_attack_prob = attack_prob
            compromise_step = step + 1
            compromise_cls = 1 if step_probs[1] > step_probs[2] else 2
            
    if compromise_step != -1:
        st.error(f"""
        ### PROACTIVE ALERT: Potential Compromise Predicted!
        * **Target Attack Stage:** {mitre_stages[compromise_cls]} (MITRE ATT&CK: Credential Access)
        * **Estimated Time to Compromise:** {compromise_step} seconds from now
        * **Attacker Confidence:** {max_attack_prob * 100:.1f}%
        """)
    else:
        st.success("**No attack signature detected in the predicted future states.**")
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot rollout probability progression
        st.markdown("**Predicted Attacker Progression Probability Timeline**")
        
        # Build timeline dataframe
        steps_axis = [f"t-{engine.history_len - i - 1}s" for i in range(engine.history_len)] + [f"t+{i+1}s (Pred)" for i in range(K_steps)]
        
        # Historical labels mapping (one-hot representing actual states)
        history_benign_probs = [1.0 if l == 0 else 0.0 for l in true_history_labels]
        history_ftp_probs = [1.0 if l == 1 else 0.0 for l in true_history_labels]
        history_ssh_probs = [1.0 if l == 2 else 0.0 for l in true_history_labels]
        
        # Predicted probabilities
        pred_benign_probs = rollout_probs[:, 0].tolist()
        pred_ftp_probs = rollout_probs[:, 1].tolist()
        pred_ssh_probs = rollout_probs[:, 2].tolist()
        
        timeline_df = pd.DataFrame({
            'Time': steps_axis,
            'Benign': history_benign_probs + pred_benign_probs,
            'FTP-BruteForce': history_ftp_probs + pred_ftp_probs,
            'SSH-Bruteforce': history_ssh_probs + pred_ssh_probs,
            'Type': ['Historical'] * engine.history_len + ['Forecasted'] * K_steps
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_df['Time'], y=timeline_df['Benign'],
            mode='lines+markers', name='Benign', line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=timeline_df['Time'], y=timeline_df['FTP-BruteForce'],
            mode='lines+markers', name='FTP Brute Force', line=dict(color='orange', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=timeline_df['Time'], y=timeline_df['SSH-Bruteforce'],
            mode='lines+markers', name='SSH Brute Force', line=dict(color='red', width=2)
        ))
        
        # Add transition boundary shape and annotation manually for categorical x-axis compatibility
        fig.add_shape(
            type="line",
            x0="t-0s", y0=-0.05, x1="t-0s", y1=1.05,
            line=dict(color="grey", width=1.5, dash="dash")
        )
        fig.add_annotation(
            x="t-0s", y=1.0,
            text="Forecast Start",
            showarrow=False,
            yshift=10,
            font=dict(color="grey")
        )
        
        fig.update_layout(
            yaxis_title="Probability",
            yaxis_range=[-0.05, 1.05],
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Current Observed State Features (t-0s)**")
        # Format the last state features as a dictionary and display as dataframe
        last_raw_feat = raw_history_seq[-1]
        feat_df = pd.DataFrame({
            'Feature': feature_cols,
            'Value': [f"{v:.4f}" if isinstance(v, float) else str(v) for v in last_raw_feat]
        })
        st.dataframe(feat_df, height=350, use_container_width=True)

with tab_explain:
    st.subheader("Causal Explainability: Integrated Gradients")
    st.markdown("""
    This panel shows **feature attribution** scores computed via **Integrated Gradients**. 
    It identifies which specific feature dimensions in our network state sequence contributed most to the prediction.
    """)
    
    target_exp_cls = st.selectbox(
        "Select Target Attack Stage to Explain:",
        options=[1, 2],
        format_func=lambda x: class_names[x]
    )
    
    # Run attribution
    attributions, feature_importance = engine.explain_prediction(history_seq, target_class=target_exp_cls)
    
    col_exp1, col_exp2 = st.columns([1, 1])
    
    with col_exp1:
        st.markdown(f"**Top Contributing Features for {class_names[target_exp_cls]}**")
        
        # Create dataframe of importance
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            imp_df,
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            color_continuous_scale='Reds',
            height=450
        )
        fig_imp.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_exp2:
        st.markdown("**Temporal Attribution Heatmap**")
        st.markdown("Shows how attributions of key features changed over the past 10 seconds.")
        
        # Find indices of top 8 most important features to keep heatmap legible
        top_indices = np.argsort(feature_importance)[-8:]
        top_features = [feature_cols[i] for i in top_indices]
        
        # Filter attributions matrix
        heatmap_data = attributions[:, top_indices].T  # shape: (8, history_len)
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"t-{engine.history_len - i - 1}s" for i in range(engine.history_len)],
            y=top_features,
            colorscale='RdBu',
            zmid=0.0
        ))
        fig_heat.update_layout(
            xaxis_title="Time Steps",
            margin=dict(l=20, r=20, t=20, b=20),
            height=450
        )
        st.plotly_chart(fig_heat, use_container_width=True)

with tab_benchmarks:
    st.subheader("Comparative Benchmarks")
    st.markdown("""
    This table compares the **World Model (Temporal GRU)** against a **Baseline Static Classifier (MLP)** 
    trained on the exact same dataset features.
    """)
    
    if os.path.exists("models/benchmark_results.csv"):
        bench_df = pd.read_csv("models/benchmark_results.csv")
        st.dataframe(bench_df, use_container_width=True)
        
        # Display as Bar Chart
        bench_melted = pd.melt(bench_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR'])
        fig_bench = px.bar(
            bench_melted,
            x='variable',
            y='value',
            color='Model',
            barmode='group',
            labels={'variable': 'Evaluation Metric', 'value': 'Score'},
            color_discrete_sequence=['teal', 'coral'],
            height=400
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        
        st.info("""
        **Key Observation:** The World Model achieves better performance because it learns sequence dynamics 
        (e.g. recognizing the speed, port pattern, and volume changes over time) rather than treating each state in isolation.
        """)
    else:
        st.info("No benchmark results found. Please run the training script to generate benchmarks.")
