import torch
import torch.nn as nn

class NetworkWorldModel(nn.Module):
    """
    A sequence model that learns network state transitions and predicts attacker progression.
    It takes a history sequence of network state vectors S_{t-H:t} and simultaneously outputs:
      1. The predicted next state vector S_{t+1} (regression)
      2. The predicted current/future attack stage class logits (classification)
    """
    def __init__(self, feature_dim, hidden_dim=64, num_layers=2, num_classes=3, rnn_type='gru', dropout=0.2):
        super(NetworkWorldModel, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        
        # Sequential feature extractor
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:  # default to GRU
            self.rnn = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            
        # Regression head: predicts S_{t+1}
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Classification head: predicts attack stage probability
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_dim)
        rnn_out, _ = self.rnn(x)
        
        # Use the last hidden state of the sequence for prediction
        last_hidden = rnn_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # Next state regression prediction
        next_state = self.regression_head(last_hidden)  # shape: (batch_size, feature_dim)
        
        # Attack stage logits prediction
        class_logits = self.classification_head(last_hidden)  # shape: (batch_size, num_classes)
        
        return next_state, class_logits


class BaselineModel(nn.Module):
    """
    A static (non-sequence) MLP classifier that makes predictions based on a single
    time window observation S_t in isolation, ignoring temporal history.
    This serves as our static classification baseline.
    """
    def __init__(self, feature_dim, hidden_dim=64, num_classes=3, dropout=0.2):
        super(BaselineModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Input can be a single state S_t of shape (batch_size, feature_dim)
        # or we take the last state of a sequence: x[:, -1, :]
        if len(x.shape) == 3:
            x = x[:, -1, :]  # isolate current state
        return self.network(x)
