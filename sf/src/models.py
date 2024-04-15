import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ViSNet


class AffiNETy_PL_L(nn.Module):
    def __init__(self, visnet_pl=ViSNet(), visnet_l=ViSNet(), temperature=298.0,):
        """
        Initialize the model with two ViSNet instances.

        Parameters:
        - visnet_pl: Pretrained ViSNet model for PL graphs.
        - visnet_l: Pretrained ViSNet model for L graphs.

        Objectives:
        AffiNETy_PL_L is designed to take in N PL graphs, and M L graphs.
        ViSNet will be evaluated on each graph with results.
        data should be like:
        the model will initially start with pre-trained ViSNet models,
        but will use ViSNet on each pl set of graphs and ligand set of graphs
        before performing a sum to predict the final ouptut value.
        """
        super(AffiNETy_PL_L, self).__init__()
        self.visnet_pl = visnet_pl
        self.visnet_l = visnet_l

    def forward(self, data):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.

            _d = Data(
                # pl
                pl_x=pls["x"],
                pl_edge_index=pls["edge_index"],
                pl_edge_attr=pls["edge_attr"],
                pl_z=pls["z"],
                pl_pos=pls["pos"],
                # l
                l_x=ls["x"],
                l_edge_index=ls["edge_index"],
                l_edge_attr=ls["edge_attr"],
                l_z=ls["z"],
                l_pos=ls["pos"],
            )
        """
        # pl_results = []
        # l_results = []
        print(data)
        print(data.pl_z)
        pl_energy = 0
        for i in range(len(data.pl_z)):
            pl_out = self.visnet_pl(
                z = data.pl_z[i], pos = data.pl_z[i], batch = data.batch
            )
            pl_energy += pl_out
        for i in range(len(data.l_z)):
            l_out = self.visnet_l(
                z = data.l_z[i], pos = data.l_z[i], batch = data.batch
            )
            l_energy += l_out
        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        result = -torch.log(pl_energy - l_energy / RT)
        return result


# class AffiNETy_PL_L:
#
#     return


class AffiNETy:
    def __init__(
        self, dataset=None, model=AffiNETy_PL_L, visnet_pl=ViSNet(), visnet_l=ViSNet(), lr=1e-4,
    ):
        self.model = model(visnet_pl, visnet_l)
        self.dataset = dataset
        return

    def train(
        self, epochs=100, batch_size=32, lr=0.01, split_percent=0.8, verbose=False
    ):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided!")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_dataset = self.dataset[: int(len(self.dataset) * split_percent)]
        test_dataset = self.dataset[int(len(self.dataset) * split_percent) :]
        print(
            f"Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples"
        )
        print(train_dataset[0], test_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"Batch size: {batch_size}")
        for epoch in range(epochs):
            for batch in train_loader:
                print(f"{batch = }")
                for data in batch.to_data_list():
                    optimizer.zero_grad()
                    out = self.model(
                        data
                    )
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
            for batch in test_loader:
                for data in batch.to_data_list():
                    self.model.eval()
                    out = self.model(
                        data
                    )
                    test_loss = criterion(out, data.y)
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {test_loss.item()}")
            print(f"Epoch {epoch+1}/{epochs}")
        return
