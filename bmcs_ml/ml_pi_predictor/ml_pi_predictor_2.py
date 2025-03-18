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
            self.X_data = (self.X_data - self.mean) / self.std
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
                pi_n1 = Pi_data[:, 4]    # Total energy history
                
                y = pi_n1
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
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        
        # Improved activation functions
        self.swish = lambda x: x * torch.sigmoid(x)  # Swish activation for smooth nonlinear transitions
        self.elu = nn.ELU(alpha=1.0)  # ELU to prevent vanishing gradients
        self.dropout = nn.Dropout(p=0.2)  # Dropout 
    
    def forward(self, x):
        x = self.swish(self.fc1(x))
        x = self.dropout(self.swish(self.fc2(x)))
        x = self.swish(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        return self.fc6(x)

# Training function with shuffle option
def train_nn(dataset, epochs=100, batch_size=32, initial_lr=0.01, lr_decay=0.99, shuffle=True, model_filename="ve_pi_p_e100_b32.pth"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    model = VE_TimeIntegrationPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)  # AdamW for better weight regularization
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
    
    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Surrogate model saved as {model_filename}")
    
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
    
    return avg_loss
