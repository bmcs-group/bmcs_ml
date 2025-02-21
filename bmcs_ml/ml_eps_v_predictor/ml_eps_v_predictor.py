import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


class ViscoelasticDataset(Dataset):
    def __init__(self, data_root_dir, normalize=True):
        self.data_root_dir = Path(data_root_dir)
        self.scenario_folders = [f for f in os.listdir(self.data_root_dir) if os.path.isdir(self.data_root_dir / f)]
        self.X_data, self.y_data = self.load_data()
        
        # Compute mean and std for normalization
        if normalize:
            self.mean = np.mean(self.X_data, axis=0)
            self.std = np.std(self.X_data, axis=0)
            self.X_data = (self.X_data - self.mean) / (self.std + 1e-8)  # Avoid division by zero
        else:
            self.mean = None
            self.std = None
    
    def load_data(self):
        X_data, y_data = [], []
        
        for scenario in self.scenario_folders:
            scenario_path = self.data_root_dir / scenario
            file_pattern = scenario_path / f"Pi_data_{scenario}.npy"
            
            if file_pattern.exists():
                print(f"Loading data from: {file_pattern}")
                Pi_data = np.load(file_pattern)  # Load numpy array
                
                eps_t = Pi_data[:, 0]    # Total strain history
                d_eps_t = Pi_data[:, 1]  # Strain rate history
                eps_v_t = Pi_data[:, 2]  # Viscoelastic strain history
                d_t = Pi_data[:, 3]      # Time step history
                
                # Define target: next step viscoelastic strain
                y = np.append(eps_v_t[1:], eps_v_t[-1])  # Set last row's target to itself
                X = np.column_stack([eps_t, d_eps_t, eps_v_t, d_t])  # Use full dataset
                
                X_data.append(X)
                y_data.append(y)
            else:
                print(f"Warning: No data file found in {scenario_path}")
        
        return np.vstack(X_data), np.hstack(y_data)
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X_data[idx], dtype=torch.float32), torch.tensor(self.y_data[idx], dtype=torch.float32)


class VE_TimeIntegrationPredictor(nn.Module):
    def __init__(self):
        super(VE_TimeIntegrationPredictor, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

# Training function with shuffle option
def train_nn(dataset, epochs=100, batch_size=32, initial_lr=0.01, lr_decay=0.99, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    model = VE_TimeIntegrationPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (inputs, target) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()  # Adjust learning rate
    
    # Plot training loss evolution
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Evolution")
    plt.legend()
    plt.grid()
    plt.show()
    
    torch.save(model.state_dict(), "ve_timeintegration_predictor.pth")
    print("Surrogate model saved as ve_timeintegration_predictor.pth")
    return model

# Model evaluation function
def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    criterion = nn.MSELoss()
    all_targets = []
    all_predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, target in dataloader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, target)
            total_loss += loss.item()
            
            all_targets.append(target.numpy())
            all_predictions.append(outputs.numpy())
    
    avg_loss = total_loss / len(dataloader)
    print(f"Model Evaluation Loss: {avg_loss}")

    # Convert lists to numpy arrays
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    # Plot Predictions vs. Ground Truth
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5, label="Predicted vs Actual")
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', label="Ideal Fit")
    plt.xlabel("Actual Viscoelastic Strain")
    plt.ylabel("Predicted Viscoelastic Strain")
    plt.title("Predicted vs Actual Viscoelastic Strain")
    plt.legend()
    plt.grid()
    plt.show()

    # Residual Plot (Errors)
    residuals = all_predictions - all_targets
    plt.figure(figsize=(6, 4))
    plt.scatter(all_targets, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Actual Viscoelastic Strain")
    plt.ylabel("Residual (Error)")
    plt.title("Residual Plot")
    plt.grid()
    plt.show()

    # Histogram of Residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Residuals (Prediction Error)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid()
    plt.show()

    return avg_loss
