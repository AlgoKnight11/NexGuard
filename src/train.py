import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from data_preprocessing import DataPreprocessor
from models import NetworkWorldModel, BaselineModel

def train_world_model(model, train_loader, val_loader, epochs, lr, alpha, beta, class_weights, device):
    print("Starting training of NetworkWorldModel...")
    criterion_reg = nn.MSELoss()
    # Apply class weights to handle imbalance
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_reg_loss = 0.0
        train_cls_loss = 0.0
        train_total_loss = 0.0
        
        for x, y_next, y_label in train_loader:
            x, y_next, y_label = x.to(device), y_next.to(device), y_label.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_next_state, pred_logits = model(x)
            
            # Compute losses
            loss_reg = criterion_reg(pred_next_state, y_next)
            loss_cls = criterion_cls(pred_logits, y_label)
            total_loss = alpha * loss_reg + beta * loss_cls
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_reg_loss += loss_reg.item()
            train_cls_loss += loss_cls.item()
            train_total_loss += total_loss.item()
            
        # Validation phase
        model.eval()
        val_reg_loss = 0.0
        val_cls_loss = 0.0
        val_total_loss = 0.0
        
        with torch.no_grad():
            for x, y_next, y_label in val_loader:
                x, y_next, y_label = x.to(device), y_next.to(device), y_label.to(device)
                pred_next_state, pred_logits = model(x)
                
                loss_reg = criterion_reg(pred_next_state, y_next)
                loss_cls = criterion_cls(pred_logits, y_label)
                total_loss = alpha * loss_reg + beta * loss_cls
                
                val_reg_loss += loss_reg.item()
                val_cls_loss += loss_cls.item()
                val_total_loss += total_loss.item()
                
        # Normalize losses
        train_total_loss /= len(train_loader)
        val_total_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {train_total_loss:.4f} (Reg: {train_reg_loss/len(train_loader):.4f}, Cls: {train_cls_loss/len(train_loader):.4f}) | "
              f"Val Loss: {val_total_loss:.4f} (Reg: {val_reg_loss/len(val_loader):.4f}, Cls: {val_cls_loss/len(val_loader):.4f})")
              
        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), "models/best_world_model.pth")
            print(f"--> Saved best world model weights (Val Loss: {best_val_loss:.4f})")
            
    print("Finished training NetworkWorldModel.")
    # Load best weights
    model.load_state_dict(torch.load("models/best_world_model.pth"))
    return model

def train_baseline_model(model, train_loader, val_loader, epochs, lr, class_weights, device):
    print("Starting training of BaselineModel (static MLP)...")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, _, y_label in train_loader:
            x, y_label = x.to(device), y_label.to(device)
            
            optimizer.zero_grad()
            
            # Predict class using ONLY current state (last element in the sequence)
            pred_logits = model(x)
            loss = criterion(pred_logits, y_label)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _, y_label in val_loader:
                x, y_label = x.to(device), y_label.to(device)
                pred_logits = model(x)
                loss = criterion(pred_logits, y_label)
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Cls Loss: {train_loss:.4f} | Val Cls Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_baseline_model.pth")
            print(f"--> Saved best baseline weights (Val Loss: {best_val_loss:.4f})")
            
    print("Finished training BaselineModel.")
    model.load_state_dict(torch.load("models/best_baseline_model.pth"))
    return model

def evaluate_model(model, test_loader, model_name, device, is_world_model=True):
    model.eval()
    all_preds = []
    all_labels = []
    total_reg_loss = 0.0
    criterion_reg = nn.MSELoss()
    
    with torch.no_grad():
        for x, y_next, y_label in test_loader:
            x, y_next = x.to(device), y_next.to(device)
            
            if is_world_model:
                pred_next_state, pred_logits = model(x)
                total_reg_loss += criterion_reg(pred_next_state, y_next).item()
            else:
                pred_logits = model(x)
                
            preds = torch.argmax(pred_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_label.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate Metrics
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # False Positive Rate calculation per class
    fprs = []
    for i in range(cm.shape[0]):
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        tn = cm.sum() - (tp + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr)
        
    avg_fpr = np.mean(fprs)
    
    results = {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1_score': report['macro avg']['f1-score'],
        'fpr': avg_fpr,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    if is_world_model:
        results['state_mse'] = total_reg_loss / len(test_loader)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Train Network World Model and Baseline")
    parser.add_argument("--data_path", type=str, default="cic.csv", help="Path to raw CSV dataset")
    parser.add_argument("--window_sec", type=int, default=1, help="Aggregation window size in seconds")
    parser.add_argument("--history_len", type=int, default=10, help="Temporal sequence history steps")
    parser.add_argument("--forecast_step", type=int, default=1, help="Forecasting steps ahead")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="MSE state loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="CrossEntropy classification loss weight")
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"], help="RNN backbone type")
    parser.add_argument("--hidden_dim", type=int, default=64, help="RNN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN layers")
    
    args = parser.parse_args()
    
    # Create models folder
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load and aggregate data
    preprocessor = DataPreprocessor(
        window_sec=args.window_sec,
        history_len=args.history_len,
        forecast_step=args.forecast_step
    )
    
    raw_df = preprocessor.clean_and_load(args.data_path)
    agg_df = preprocessor.aggregate_to_states(raw_df)
    
    # 2. Process and Split
    train_data, val_data, test_data = preprocessor.process_and_split(agg_df)
    
    # Save the scaler and preprocessor configurations
    preprocessor.save_scaler("models/scaler.pkl")
    
    # 3. Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(
        train_data, val_data, test_data, batch_size=args.batch_size
    )
    
    # Calculate class weights from training sequence labels (visible to model)
    train_seq_labels = np.array([train_data[1][idx + args.history_len - 1] for idx in train_data[2]])
    class_counts = np.bincount(train_seq_labels)
    num_classes = len(class_counts)
    class_weights = len(train_seq_labels) / (num_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Calculated class counts: {class_counts}")
    print(f"Calculated class weights: {class_weights}")
    
    # 4. Initialize Models
    feature_dim = len(preprocessor.feature_cols)
    print(f"Features dimension: {feature_dim}")
    
    world_model = NetworkWorldModel(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        rnn_type=args.rnn_type
    ).to(device)
    
    baseline_model = BaselineModel(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    # 5. Train Models
    world_model = train_world_model(
        world_model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        alpha=args.alpha, beta=args.beta,
        class_weights=class_weights_tensor,
        device=device
    )
    
    baseline_model = train_baseline_model(
        baseline_model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        class_weights=class_weights_tensor,
        device=device
    )
    
    # 6. Evaluate Models on Test Set
    print("\n--- EVALUATING NETWORK WORLD MODEL ---")
    wm_results = evaluate_model(world_model, test_loader, "World Model", device, is_world_model=True)
    
    print("\n--- EVALUATING BASELINE STATIC MODEL ---")
    base_results = evaluate_model(baseline_model, test_loader, "Baseline Model", device, is_world_model=False)
    
    # Print Comparison Table
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'World Model (Temporal)':<22} | {'Baseline (Static)':<18}")
    print("-"*66)
    print(f"{'Accuracy':<20} | {wm_results['accuracy']:<22.4f} | {base_results['accuracy']:<18.4f}")
    print(f"{'Precision (Macro)':<20} | {wm_results['precision']:<22.4f} | {base_results['precision']:<18.4f}")
    print(f"{'Recall (Macro)':<20} | {wm_results['recall']:<22.4f} | {base_results['recall']:<18.4f}")
    print(f"{'F1-Score (Macro)':<20} | {wm_results['f1_score']:<22.4f} | {base_results['f1_score']:<18.4f}")
    print(f"{'False Positive Rate':<20} | {wm_results['fpr']:<22.4f} | {base_results['fpr']:<18.4f}")
    if 'state_mse' in wm_results:
        print(f"{'Transition State MSE':<20} | {wm_results['state_mse']:<22.6f} | {'N/A':<18}")
    print("="*50)
    
    # Save benchmark results
    benchmark_df = pd.DataFrame({
        'Model': ['World Model', 'Baseline Static'],
        'Accuracy': [wm_results['accuracy'], base_results['accuracy']],
        'Precision': [wm_results['precision'], base_results['precision']],
        'Recall': [wm_results['recall'], base_results['recall']],
        'F1-Score': [wm_results['f1_score'], base_results['f1_score']],
        'FPR': [wm_results['fpr'], base_results['fpr']]
    })
    benchmark_df.to_csv("models/benchmark_results.csv", index=False)
    print("Benchmark results saved to models/benchmark_results.csv")

if __name__ == "__main__":
    main()
