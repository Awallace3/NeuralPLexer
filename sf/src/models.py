import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ViSNet


class AffiNETy_PL_L(nn.Module):
    def __init__(self, visnet_pl=ViSNet, visnet_l=ViSNet):
        """
        Initialize the model with two ViSNet instances.

        Parameters:
        - visnet_pl: Pretrained ViSNet model for PL graphs.
        - visnet_l: Pretrained ViSNet model for L graphs.
        """
        super(AffiNETy_PL_L, self).__init__()
        self.visnet_pl = visnet_pl
        self.visnet_l = visnet_l

    def forward(self, data_list):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        results = []
        for data in data_list:
            # Process PL graphs
            pl_out = self.visnet_pl(
                data.pl_x, data.pl_edge_index, data.pl_edge_attr, data.pl_pos
            )
            # Process L graphs
            l_out = self.visnet_l(
                data.l_x, data.l_edge_index, data.l_edge_attr, data.l_pos
            )
            # Combine results from both models
            combined_result = pl_out.sum() + l_out.sum()
            # Store the result for this sample
            results.append(combined_result)
        # Convert list of results to a tensor
        return torch.stack(results)


# class AffiNETy_PL_L:
#     """
#     AffiNETy_PL_L is designed to take in N PL graphs, and M L graphs.
#     ViSNet will be evaluated on each graph with results.
#     data should be like:
#         _d = Data(
#             # pl
#             pl_x=pls["x"],
#             pl_edge_index=pls["edge_index"],
#             pl_edge_attr=pls["edge_attr"],
#             pl_z=pls["z"],
#             pl_pos=pls["pos"],
#             # l
#             l_x=ls["x"],
#             l_edge_index=ls["edge_index"],
#             l_edge_attr=ls["edge_attr"],
#             l_z=ls["z"],
#             l_pos=ls["pos"],
#             y=self.power_ranking_dict[i]
#         )
#     the model will initially start with pre-trained ViSNet models,
#     but will use ViSNet on each pl set of graphs and ligand set of graphs
#     before performing a sum to predict the final ouptut value.
#     """
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
                for data in batch.to_data_list():
                    optimizer.zero_grad()
                    out = self.model(
                        data.pl_x,
                        data.pl_edge_index,
                        data.pl_edge_attr,
                        data.pl_z,
                        data.pl_pos,
                        data.l_x,
                        data.l_edge_index,
                        data.l_edge_attr,
                        data.l_z,
                        data.l_pos,
                    )
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
            for batch in test_loader:
                for data in batch.to_data_list():
                    self.model.eval()
                    out = self.model(
                        data.pl_x,
                        data.pl_edge_index,
                        data.pl_edge_attr,
                        data.pl_z,
                        data.pl_pos,
                        data.l_x,
                        data.l_edge_index,
                        data.l_edge_attr,
                        data.l_z,
                        data.l_pos,
                    )
                    test_loss = criterion(out, data.y)
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {test_loss.item()}")
            print(f"Epoch {epoch+1}/{epochs}")
        return
