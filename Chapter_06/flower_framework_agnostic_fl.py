"""
Flower 1.x (Federated App) - Framework-Agnostic & Production-Ready
------------------------------------------------------------------
This script is updated for the latest Flower 1.x App architecture (Compatible with 1.27.0+).

To run this simulation without warnings:
$ flwr run Chapter_06/

Key Features:
1. Framework-Agnostic: Using NumPy-based transitions.
2. Federated App: Decoupled ClientApp and ServerApp logic.
3. Bug Fix: Proper use of torch DataLoader and dataset length.
"""

import os
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

# --- 1. MODEL DEFINITION (Framework-Agnostic via NumPy) ---

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# --- 2. FLOWER CLIENT (Framework-Agnostic) ---

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        for _ in range(1): # 1 Epoch
            for data, target in self.trainloader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()
        # count number of examples in the dataset (Fix: self.trainloader.dataset)
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = 0.0
        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                loss += nn.MSELoss()(output, target).item()
        return loss / len(self.testloader.dataset), len(self.testloader.dataset), {}

# --- 3. FLOWER APP LOGIC (1.x App Architecture) ---

def client_fn(context: fl.common.Context) -> fl.client.Client:
    """Function to create a client instance."""
    model = SimpleModel()
    
    # 1. Prepare Data (Mocking Non-IID or local silos)
    x = torch.randn(64, 10).float()
    y = torch.randn(64, 1).float()
    
    # Use proper DataLoader to support .dataset attribute for length check
    dataset = TensorDataset(x, y)
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(dataset, batch_size=32)
    
    # Convert NumPyClient to Client
    return FlowerClient(model, trainloader, testloader).to_client()

def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    """Function to configure the server components."""
    # 2. Define Strategy (Production-ready FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,      # Sample all clients
        fraction_evaluate=1.0, # Evaluate on all clients
        min_fit_clients=2,     # Minimum 2 clients to proceed
        min_available_clients=2,
    )
    
    # Configure rounds
    num_rounds = context.run_config.get("num-rounds", 3)
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

# --- 4. DEPLOYMENT ENTRY POINTS ---

# In Flower 1.x, we define ClientApp and ServerApp.
# These are used by 'flwr run' to orchestrate the simulation or deployment.
client_app = fl.client.ClientApp(client_fn=client_fn)
server_app = fl.server.ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("-" * 60)
    print("Flower 1.x Federated App: Framework-Agnostic & Production-Ready")
    print("-" * 60)
    print("\n[IMPORTANT] Simulation Warning Fix:")
    print("To run this federated simulation without deprecation warnings:")
    print("1. Ensure 'pyproject.toml' is in this directory.")
    print("2. Run the following command from the project root:")
    print("   $ flwr run Chapter_06/")
    print("\nNote: Manual 'python script.py' with start_simulation() is deprecated")
    print("in favor of the App-based 'flwr run' CLI.")
    
    # Optional: If you STILL want to run it via 'python flower_framework_agnostic_fl.py',
    # we can use a basic simulation setup, but the WARNING will persist.
    # To fix the TypeError, we use the LEGACY arguments for the LEGACY function.
    from flwr.simulation import start_simulation
    
    # Re-wrap client_fn to match legacy signature if needed for start_simulation
    def legacy_client_fn(cid: str):
        # We manually create a mock context for the new client_fn
        # Flower 1.27.0+ requires run_id and state in Context
        mock_context = fl.common.Context(
            run_id=0,
            node_id=int(cid),
            node_config={},
            run_config={},
            state=fl.common.RecordSet(),
        )
        return client_fn(mock_context)

    print("\nRunning legacy simulation mode (for quick local test)...")
    start_simulation(
        client_fn=legacy_client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(min_available_clients=2, min_fit_clients=2),
    )
