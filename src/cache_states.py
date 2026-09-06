"""
Precomputes aggregated states from cic.csv and caches them to models/cached_test_states.pkl
so the Streamlit dashboard loads in milliseconds.
"""
import os
import joblib
import pandas as pd
from data_preprocessing import DataPreprocessor

def build_cache(csv_path="cic.csv", output_path="models/cached_test_states.pkl"):
    print("Building state cache from cic.csv...")
    # Load metadata
    metadata = joblib.load("models/scaler.pkl")
    scaler = metadata['scaler']
    feature_cols = metadata['feature_cols']
    
    preprocessor = DataPreprocessor(window_sec=1, history_len=10, forecast_step=1)
    raw_df = preprocessor.clean_and_load(csv_path)
    agg_df = preprocessor.aggregate_to_states(raw_df)
    
    features = agg_df[feature_cols].values
    scaled_features = scaler.transform(features)
    labels = agg_df['label'].values
    
    # Save test segment (last 20% of the timeline, plus a representative sample)
    n_samples = len(scaled_features)
    test_start = max(0, int(n_samples * 0.80))
    
    cache_data = {
        'scaled_features': scaled_features[test_start:],
        'raw_features': features[test_start:],
        'labels': labels[test_start:],
        'feature_cols': feature_cols,
        'total_time_bins': len(agg_df)
    }
    
    joblib.dump(cache_data, output_path)
    print(f"Cached {len(cache_data['scaled_features'])} states to {output_path}")

if __name__ == "__main__":
    build_cache()
