import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import warnings

# Suppress the UserWarning about Secure RNG being turned off for experimentation
warnings.filterwarnings("ignore", message="Secure RNG turned off", category=UserWarning)
# Suppress the UserWarning about Optimal order being the largest alpha
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha", category=UserWarning)

# 1. Setup Data and Evaluation Metric
X_train, y_train = torch.randn(2000, 10), torch.randint(0, 2, (2000, 1)).float()
X_val, y_val = torch.randn(500, 10), torch.randint(0, 2, (500, 1)).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = (model(x) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 2. Define the Trade-off Experiment
target_epsilons = [0.5, 1.0, 3.0, 5.0, 10.0]
results = {}

print("--- Evaluating Utility-Privacy Trade-off ---")

for eps in target_epsilons:
    # Reset model and optimizer for each epsilon
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        epochs=10, target_epsilon=eps, target_delta=1e-5, max_grad_norm=1.0
    )
    
    # Train
    model.train()
    for epoch in range(10):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = nn.BCELoss()(model(x), y)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    acc = evaluate(model, val_loader)
    results[eps] = acc
    print(f"Target ε: {eps:>4.1f} | Achieved Acc: {acc:.4f}")

# Train Non-Private Baseline
baseline_model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
baseline_opt = optim.SGD(baseline_model.parameters(), lr=0.1)
for epoch in range(10):
    for x, y in train_loader:
        baseline_opt.zero_grad()
        nn.BCELoss()(baseline_model(x), y).backward()
        baseline_opt.step()
        
baseline_acc = evaluate(baseline_model, val_loader)
print(f"Baseline (ε=∞): {baseline_acc:.4f}")