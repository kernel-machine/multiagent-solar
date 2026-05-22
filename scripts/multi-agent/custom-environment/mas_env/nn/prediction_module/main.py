import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from prediction import GHIPredictorLSTM

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[(i + lookback):(i + lookback + horizon)])
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="LSTM for GHI prediction")
    parser.add_argument("dataset", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--lookback", type=int, default=96, help="Lookback window size (es. 96 per 24h)")
    parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon (es. 24 per 6h)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use-log1p", action="store_true", help="Apply log1p transformation to GHI")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the time series
    df = pd.read_csv(args.dataset)
    ghi_values = df['ghi'].values.astype(np.float32)

    if args.use_log1p:
        print("Applying log1p transformation...")
        ghi_values = np.log1p(ghi_values)

    # MinMax Scaling
    min_val, max_val = np.min(ghi_values), np.max(ghi_values)
    if max_val > min_val:
        ghi_scaled = (ghi_values - min_val) / (max_val - min_val)
    else:
        ghi_scaled = ghi_values

    # Create sequences (X = context, y = prediction horizon)
    X, y = create_sequences(ghi_scaled, args.lookback, args.horizon)
    X = X.reshape(-1, args.lookback, 1)  # shape (samples, lookback, 1)

    # Split Train/Test (80% train, 20% test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    model = GHIPredictorLSTM(input_size=1, hidden_size=64, num_layers=3, output_size=args.horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.6f}")
	
	# Save model
    torch.save(model.state_dict(), "ghi_predictor_lstm.pth")
    print("Starting Evaluation...")
    model.eval()
    test_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # Inverse transform
    predictions = predictions * (max_val - min_val) + min_val
    targets = targets * (max_val - min_val) + min_val
    
    if args.use_log1p:
        predictions = np.expm1(predictions)
        targets = np.expm1(targets)

    # Plot the comparison for multiple random batch elements in the test set
    num_plots = 4
    indices_to_plot = np.random.choice(len(targets), num_plots, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices_to_plot):
        plt.subplot(2, 2, i + 1)
        plt.plot(targets[idx], label="Real Values (GHI)", marker='o')
        plt.plot(predictions[idx], label="Predictions (GHI)", marker='x')
        plt.title(f"Comparison (Test Set) - Sample {idx}")
        plt.xlabel("Time Horizon")
        plt.ylabel("GHI Value")
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig("ghi_prediction_comparison.png")
    print("Comparison saved in 'ghi_prediction_comparison.png'")

if __name__ == "__main__":
    main()
