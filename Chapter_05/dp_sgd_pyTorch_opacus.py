import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

def train_dp_model():
    # 1. Setup Dummy Data and Model
    # ---------------------------------------------------------
    print("Setting up data and model...")
    # 1000 samples, 10 features each
    X = torch.randn(1000, 10) 
    # Binary classification target
    y = torch.randint(0, 2, (1000, 1)).float() 

    dataset = TensorDataset(X, y)
    # Note: Opacus handles batching securely, but requires standard DataLoaders
    data_loader = DataLoader(dataset, batch_size=32)

    # A simple Multi-Layer Perceptron (MLP)
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )

    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    # 2. Attach the Privacy Engine (Opacus)
    # ---------------------------------------------------------
    import warnings
    # Suppress the UserWarning about Secure RNG being turned off for experimentation
    warnings.filterwarnings("ignore", message="Secure RNG turned off", category=UserWarning)

    print("Initializing Privacy Engine...")
    privacy_engine = PrivacyEngine()

    # 'make_private' wraps the model, optimizer, and dataloader.
    # It automatically handles per-sample gradient computation, clipping, and noising.
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.2, # Controls the amount of Gaussian noise added
        max_grad_norm=1.0,    # The clipping threshold (C) for per-sample gradients
    )

    # 3. Training Loop
    # ---------------------------------------------------------
    epochs = 5
    delta = 1e-5 # Target delta (usually set to < 1/N, where N is dataset size)

    print("\n--- Starting DP-SGD Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in data_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass (Opacus intercepts this to compute per-sample gradients)
            loss.backward()
            
            # Optimizer step (Opacus clips and adds noise here)
            optimizer.step()
            
            total_loss += loss.item()

        # Calculate the privacy budget (epsilon) spent so far
        epsilon = privacy_engine.get_epsilon(delta)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(data_loader):.4f} | "
              f"Privacy spent: (ε = {epsilon:.2f}, δ = {delta})")

if __name__ == "__main__":
    train_dp_model()