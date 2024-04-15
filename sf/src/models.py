import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


class AffiNETy_PL_L():
    return

class AffiNETy():
    def __init__(self, dataset=None, ):
        self.model = AtomMPNN(n_message=n_message, n_rbf=n_rbf, n_neuron=n_neuron, n_embed=n_embed, r_cut=r_cut)
        self.dataset = dataset
        return

    def train(self, dataset=None, epochs=100, batch_size=32, lr=0.01, split_percent=0.8, verbose=False):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_dataset = self.dataset[:int(len(self.dataset)*split_percent)]
        test_dataset = self.dataset[int(len(self.dataset)*split_percent):]
        print(f"Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples")
        print(train_dataset[0], test_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"Batch size: {batch_size}")
        for epoch in range(epochs):
            for batch in train_loader:
                for data in batch.to_data_list():
                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(data.x, data.edge_index, data.edge_attr, R=data.R, molecule_ind=data.molecule_ind, total_charge=data.total_charge)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
            for batch in test_loader:
                for data in batch.to_data_list():
                    self.model.eval()
                    out = self.model(data.x, data.edge_index)
                    test_loss = criterion(out, data.y)
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {test_loss.item()}")
            print(f"Epoch {epoch+1}/{epochs}")
        return

