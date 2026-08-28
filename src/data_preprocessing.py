import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class NetworkDataset(Dataset):
    """
    Custom PyTorch Dataset for sequential network state inputs.
    Returns:
        x: sequence of historical states of shape (history_len, feature_dim)
        y_next: next state vector of shape (feature_dim,) - target for World Model regression
        y_label: attack label for current time window (classification target)
    """
    def __init__(self, states, labels, indices, history_len=10, forecast_step=1):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = indices
        self.history_len = history_len
        self.forecast_step = forecast_step
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve the sequence start index
        seq_idx = self.indices[idx]
        
        # Historical sequence of states S_{t-H:t}
        x = self.states[seq_idx : seq_idx + self.history_len]
        # Label at time t (end of the history window)
        y_label = self.labels[seq_idx + self.history_len - 1]
        # Target next state S_{t+1} (or S_{t+k})
        y_next = self.states[seq_idx + self.history_len + self.forecast_step - 1]
        
        return x, y_next, y_label

class DataPreprocessor:
    def __init__(self, window_sec=1, history_len=10, forecast_step=1):
        self.window_sec = window_sec
        self.history_len = history_len
        self.forecast_step = forecast_step
        self.scaler = StandardScaler()
        self.feature_cols = []

    def clean_and_load(self, file_path):
        """Loads and cleans raw flow logs."""
        print(f"Loading dataset from {file_path}...")
        # Read essential columns first
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        print("Cleaning timestamps and filtering years...")
        # Parse timestamps
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        # Drop rows with invalid timestamps or from non-2018 years
        df = df[df['Timestamp'].notnull()]
        df = df[df['Timestamp'].dt.year == 2018]
        
        # Replace infinities and NaNs
        print("Handling missing/infinite values...")
        df = df.replace([np.inf, -np.inf], np.nan)
        # Fill numeric NaNs with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Sort chronologically
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        return df

    def compute_port_entropy(self, ports):
        """Computes entropy of destination ports in a window to detect scans."""
        if len(ports) == 0:
            return 0.0
        counts = pd.Series(ports).value_counts()
        probs = counts / len(ports)
        return -np.sum(probs * np.log2(probs))

    def aggregate_to_states(self, df):
        """
        Aggregates flow records into window_sec state bins using fast vectorized operations.
        """
        print(f"Aggregating network flows into {self.window_sec}s state windows...")
        
        # Map labels to integers
        # 0: Benign, 1: FTP-BruteForce, 2: SSH-Bruteforce
        label_map = {'Benign': 0, 'FTP-BruteForce': 1, 'SSH-Bruteforce': 2}
        df['Label_int'] = df['Label'].map(label_map).fillna(0).astype(int)
        
        # Binary indicator columns for protocols
        df['is_tcp'] = (df['Protocol'] == 6).astype(int)
        df['is_udp'] = (df['Protocol'] == 17).astype(int)
        
        # Create floor timestamp bins
        freq = f"{self.window_sec}s"
        df['TimeBin'] = df['Timestamp'].dt.floor(freq)
        
        # Set up a continuous index of time bins to avoid missing seconds
        min_time = df['TimeBin'].min()
        max_time = df['TimeBin'].max()
        all_bins = pd.date_range(start=min_time, end=max_time, freq=freq)
        
        # Perform fast group aggregation
        agg_res = df.groupby('TimeBin').agg(
            flow_count=('Label_int', 'count'),
            tcp_cnt=('is_tcp', 'sum'),
            udp_cnt=('is_udp', 'sum'),
            syn_cnt=('SYN Flag Cnt', 'sum'),
            ack_cnt=('ACK Flag Cnt', 'sum'),
            rst_cnt=('RST Flag Cnt', 'sum'),
            psh_cnt=('PSH Flag Cnt', 'sum'),
            fin_cnt=('FIN Flag Cnt', 'sum'),
            urg_cnt=('URG Flag Cnt', 'sum'),
            dst_ports=('Dst Port', list),
            flow_duration_mean=('Flow Duration', 'mean'),
            tot_fwd_pkts_sum=('Tot Fwd Pkts', 'sum'),
            tot_bwd_pkts_sum=('Tot Bwd Pkts', 'sum'),
            tot_len_fwd_pkts_sum=('TotLen Fwd Pkts', 'sum'),
            tot_len_bwd_pkts_sum=('TotLen Bwd Pkts', 'sum'),
            pkt_len_mean=('Pkt Len Mean', 'mean'),
            flow_iat_mean=('Flow IAT Mean', 'mean'),
            init_fwd_win_mean=('Init Fwd Win Byts', 'mean'),
            init_bwd_win_mean=('Init Bwd Win Byts', 'mean'),
            label=('Label_int', 'max')
        )
        
        # Compute port entropy and unique ports
        agg_res['port_entropy'] = agg_res['dst_ports'].apply(self.compute_port_entropy)
        agg_res['unique_ports'] = agg_res['dst_ports'].apply(lambda l: len(set(l)))
        agg_res = agg_res.drop(columns=['dst_ports'])
        
        # Reindex to continuous timeline bins (fills gaps with NaNs)
        agg_res = agg_res.reindex(all_bins)
        
        # Fill missing values for times with no network activity
        agg_res['flow_count'] = agg_res['flow_count'].fillna(0).astype(int)
        agg_res['label'] = agg_res['label'].fillna(0).astype(int)
        
        # Calculate ratios and replace remaining NaNs with 0
        for prefix in ['tcp', 'udp', 'syn', 'ack', 'rst', 'psh', 'fin', 'urg']:
            cnt_col = f'{prefix}_cnt'
            ratio_col = f'{prefix}_ratio'
            if cnt_col in agg_res.columns:
                agg_res[ratio_col] = np.where(agg_res['flow_count'] > 0, agg_res[cnt_col] / agg_res['flow_count'], 0.0)
                agg_res = agg_res.drop(columns=[cnt_col])
                
        # Fill other numeric columns (means, sums) with 0 for idle periods
        agg_res = agg_res.fillna(0.0)
        
        self.feature_cols = [col for col in agg_res.columns if col != 'label']
        return agg_res.reset_index(drop=True)

    def process_and_split(self, agg_df, train_ratio=0.7, val_ratio=0.15, block_sec=600):
        """Splits sequence start indices using a block temporal split and fits scaler."""
        print("Splitting dataset indices and fitting scaler...")
        
        features = agg_df[self.feature_cols].values
        labels = agg_df['label'].values
        n_samples = len(agg_df)
        
        # Max starting index for a valid sequence
        max_start_idx = n_samples - self.history_len - self.forecast_step
        if max_start_idx < 0:
            raise ValueError("Dataset is too small for the given history_len and forecast_step.")
            
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx in range(max_start_idx + 1):
            in_block_pos = idx % block_sec
            if in_block_pos < int(block_sec * train_ratio):
                train_indices.append(idx)
            elif in_block_pos < int(block_sec * (train_ratio + val_ratio)):
                val_indices.append(idx)
            else:
                test_indices.append(idx)
                
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        # Collect all state indices that are visible during training (to fit the scaler)
        train_state_indices = set()
        for idx in train_indices:
            for offset in range(self.history_len):
                train_state_indices.add(idx + offset)
        train_state_indices = sorted(list(train_state_indices))
        
        # Fit scaler ONLY on training states
        self.scaler.fit(features[train_state_indices])
        
        # Transform all features
        scaled_features = self.scaler.transform(features)
        
        print(f"Sequence Split Sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        return (scaled_features, labels, train_indices), (scaled_features, labels, val_indices), (scaled_features, labels, test_indices)

    def get_dataloaders(self, train_data, val_data, test_data, batch_size=64):
        """Creates PyTorch DataLoaders."""
        train_ds = NetworkDataset(train_data[0], train_data[1], train_data[2], self.history_len, self.forecast_step)
        val_ds = NetworkDataset(val_data[0], val_data[1], val_data[2], self.history_len, self.forecast_step)
        test_ds = NetworkDataset(test_data[0], test_data[1], test_data[2], self.history_len, self.forecast_step)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def save_scaler(self, path):
        """Saves scaler and feature list for deployment."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'window_sec': self.window_sec,
            'history_len': self.history_len
        }, path)
        print(f"Scaler saved to {path}")
