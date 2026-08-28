import joblib
import torch
import torch.nn as nn
import numpy as np
from models import NetworkWorldModel
from captum.attr import IntegratedGradients

# A wrapper module to expose only classification logits for Captum
class ModelClsWrapper(nn.Module):
    def __init__(self, model):
        super(ModelClsWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        _, logits = self.model(x)
        return logits

class InferenceEngine:
    def __init__(self, model_path="models/best_world_model.pth", scaler_path="models/scaler.pkl", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load preprocessor metadata and scaler
        metadata = joblib.load(scaler_path)
        self.scaler = metadata['scaler']
        self.feature_cols = metadata['feature_cols']
        self.window_sec = metadata['window_sec']
        self.history_len = metadata['history_len']
        
        # Initialize model
        feature_dim = len(self.feature_cols)
        # Note: We assume standard hyperparameters (hidden_dim=64, num_layers=2, num_classes=3)
        self.model = NetworkWorldModel(
            feature_dim=feature_dim,
            hidden_dim=64,
            num_layers=2,
            num_classes=3
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Explainability wrapper
        self.wrapper_model = ModelClsWrapper(self.model)
        self.ig = IntegratedGradients(self.wrapper_model)
        
    def predict_next(self, history_sequence):
        """
        Predict next state and current attack stage from history sequence.
        Args:
            history_sequence: numpy array of shape (history_len, feature_dim) - SCALED features
        Returns:
            next_state_pred: numpy array of shape (feature_dim,) - SCALED features
            probs: numpy array of shape (num_classes,) - class probabilities
        """
        x_tensor = torch.tensor(history_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            next_state_tensor, logits_tensor = self.model(x_tensor)
            probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()[0]
            next_state_pred = next_state_tensor.cpu().numpy()[0]
            
        return next_state_pred, probs

    def forward_rollout(self, history_sequence, K_steps=5):
        """
        Perform recursive forward simulation for K steps into the future.
        Args:
            history_sequence: numpy array of shape (history_len, feature_dim) - SCALED features
            K_steps: number of steps to simulate forward
        Returns:
            rollout_states: numpy array of shape (K_steps, feature_dim) - SCALED features
            rollout_probs: numpy array of shape (K_steps, num_classes) - class probabilities
        """
        seq = history_sequence.copy()
        
        rollout_states = []
        rollout_probs = []
        
        for k in range(K_steps):
            # Feed current sequence
            next_state_pred, probs = self.predict_next(seq)
            
            rollout_states.append(next_state_pred)
            rollout_probs.append(probs)
            
            # Slide window: discard oldest state, append predicted next state
            seq = np.vstack([seq[1:], next_state_pred])
            
        return np.array(rollout_states), np.array(rollout_probs)

    def explain_prediction(self, history_sequence, target_class=1):
        """
        Compute Integrated Gradients attributions for a given sequence with respect to a target class.
        Args:
            history_sequence: numpy array of shape (history_len, feature_dim) - SCALED features
            target_class: index of target class (1: FTP-BruteForce, 2: SSH-Bruteforce)
        Returns:
            attributions: numpy array of shape (history_len, feature_dim)
            feature_importance: numpy array of shape (feature_dim,) - summed attribution over time steps
        """
        x_tensor = torch.tensor(history_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Enable gradients
        x_tensor.requires_grad = True
        
        # Baselines: zero baseline
        baseline = torch.zeros_like(x_tensor).to(self.device)
        
        # Compute attributions
        attributions, delta = self.ig.attribute(x_tensor, baseline, target=target_class, return_convergence_delta=True)
        
        attributions = attributions.detach().cpu().numpy()[0]  # shape: (history_len, feature_dim)
        
        # Sum attributions over the history sequence to find total feature contribution
        feature_importance = np.sum(np.abs(attributions), axis=0)
        
        return attributions, feature_importance

    def descale_state(self, scaled_state):
        """Converts scaled features back to original scale."""
        if len(scaled_state.shape) == 1:
            return self.scaler.inverse_transform(scaled_state.reshape(1, -1))[0]
        return self.scaler.inverse_transform(scaled_state)

    def scale_state(self, raw_state):
        """Converts raw features to scaled features."""
        if len(raw_state.shape) == 1:
            return self.scaler.transform(raw_state.reshape(1, -1))[0]
        return self.scaler.transform(raw_state)
