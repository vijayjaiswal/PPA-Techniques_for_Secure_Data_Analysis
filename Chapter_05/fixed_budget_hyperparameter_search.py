import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import itertools
import warnings

# Suppress the UserWarning about Secure RNG being turned off for experimentation
warnings.filterwarnings("ignore", message="Secure RNG turned off", category=UserWarning)
# Suppress the UserWarning about Optimal order being the largest alpha
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha", category=UserWarning)

def evaluate_model(model, data_loader):
    """Simple evaluation loop."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def train_and_evaluate_dp(max_grad_norm, lr, target_epsilon, epochs, train_loader, val_loader):
    """Trains a model with a fixed epsilon and returns validation accuracy."""
    
    # 1. Define Model and Optimizer (Must be fresh for each run)
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 2. Attach Privacy Engine with a FIXED Epsilon
    privacy_engine = PrivacyEngine()
    
    # This automatically calculates the correct noise_multiplier!
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=1e-5,
        max_grad_norm=max_grad_norm,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 128))
    )

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

    # 4. Evaluate and return
    val_acc = evaluate_model(model, val_loader)
    
    # Extract the noise multiplier that Opacus calculated under the hood
    calculated_noise = optimizer.noise_multiplier
    
    return val_acc, calculated_noise

def run_hyperparameter_search():
    # Setup dummy data
    X = torch.randn(2000, 10)
    y = torch.randint(0, 2, (2000, 1)).float()
    
    # Split into train/val
    train_loader = DataLoader(TensorDataset(X[:1500], y[:1500]), batch_size=64)
    val_loader = DataLoader(TensorDataset(X[1500:], y[1500:]), batch_size=64)

    # Define our constraints and grid
    TARGET_EPSILON = 3.0
    EPOCHS = 5
    
    # The Grid
    grad_norms = [0.1, 1.0, 5.0]
    learning_rates = [0.01, 0.1, 0.5]
    
    print(f"--- Starting DP-SGD Grid Search (Fixed ε = {TARGET_EPSILON}) ---")
    best_acc = 0
    best_params = None

    for norm, lr in itertools.product(grad_norms, learning_rates):
        val_acc, noise_mult = train_and_evaluate_dp(
            max_grad_norm=norm, 
            lr=lr, 
            target_epsilon=TARGET_EPSILON, 
            epochs=EPOCHS, 
            train_loader=train_loader, 
            val_loader=val_loader
        )
        
        print(f"Norm: {norm:<4} | LR: {lr:<5} | Calc. Noise: {noise_mult:.2f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = (norm, lr)

    print("\n--- Search Complete ---")
    print(f"Best Accuracy  : {best_acc:.4f}")
    print(f"Best Parameters: max_grad_norm = {best_params[0]}, learning_rate = {best_params[1]}")

if __name__ == "__main__":
    run_hyperparameter_search()