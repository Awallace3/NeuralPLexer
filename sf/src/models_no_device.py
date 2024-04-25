import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ViSNet

import torch.distributed as dist


class AffiNETy_PL_P_L(nn.Module):
    def __init__(
        self,
        visnet_pl=ViSNet(),
        visnet_p=ViSNet(),
        visnet_l=ViSNet(),
        temperature=298.0,
    ):
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
        super(AffiNETy_PL_P_L, self).__init__()
        self.visnet_pl = visnet_pl
        self.visnet_p = visnet_p
        self.visnet_l = visnet_l
        self.temperature = temperature

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
        print(data)
        pl_es = torch.zeros(len(data.pl_z), dtype=torch.float)
        p_es = torch.zeros(len(data.p_z), dtype=torch.float)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_z[i]), dtype=torch.int64, )
            l_es[i] = self.visnet_l(
                z=data.l_z[i],
                pos=data.l_pos[i],
                batch=batch,
            )[0]
        for i in range(len(data.pl_z)):
            batch = torch.zeros(len(data.pl_z[i]), dtype=torch.int64, )
            pl_es[i] = self.visnet_pl(
                z=data.pl_z[i],
                pos=data.pl_pos[i],
                batch=batch,
            )[0]
        for i in range(len(data.p_z)):
            batch = torch.zeros(len(data.p_z[i]), dtype=torch.int64, )
            p_es[i] = self.visnet_p(
                z=data.p_z[i],
                pos=data.p_pos[i],
                batch=batch,
            )[0]

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        result = (
            torch.mean(pl_es) - torch.mean(p_es) - torch.mean(l_es) / -RT
        )
        return result
"""
criterion = nn.L1Loss()
if os.path.exists("./ViSNet_task1_energy"):
    model_energy = ViSNet()
    # Load the saved state dictionary
    model_energy = torch.load('./ViSNet_task1_energy')
    eval_loss = 0.
    with torch.no_grad():
        for batch in test_loader_energy:
            z = batch.z.to(device)
            pos = batch.pos.to(device)
            b = batch.batch.to(device)
            output = model_energy(z, pos, b)
            e = batch.energy.view(-1, 1).to(device)
            loss = criterion(output[0], e)
            eval_loss += loss.item()
    print("Test MAE:", eval_loss / len(test_loader_energy))
else:
    model_energy = ViSNet()
    optimizer = torch.optim.Adam(model_energy.parameters(), lr=0.0001)
    if gpu_enabled:
        model_energy = model_energy.cuda()
        model_energy = model_energy.to(device)
    lowest_test_error = 1000
    for epoch in range(50):
        train_loss = 0.
        eval_loss = 0.
        model_energy.train()
        for batch in train_loader_energy:
            optimizer.zero_grad()
            z = batch.z.to(device)
            pos = batch.pos.to(device)
            b = batch.batch.to(device)
            output = model_energy(z, pos, b)
            e = batch.energy.view(-1, 1).to(device)
            loss = criterion(output[0], e)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model_energy.eval()
        with torch.no_grad():
            for batch in test_loader_energy:
                z = batch.z.to(device)
                pos = batch.pos.to(device)
                b = batch.batch.to(device)
                output = model_energy(z, pos, b)
                e = batch.energy.view(-1, 1).to(device)
                loss = criterion(output[0], e)
                eval_loss += loss.item()
                if loss < lowest_test_error:
                    mod_out = model_energy
                    torch.save(mod_out, "ViSNet_task1_energy")
        print("Epoch: {} | Train Loss: {:.4f} | Eval Loss: {:.4f}".format(
            epoch, train_loss/len(train_loader_energy), eval_loss/len(test_loader_energy)
        ))

    if gpu_enabled:
        mod_out.to("cpu")
"""

from torch.distributed.elastic.multiprocessing.errors import record
import os

class AffiNETy:
    def __init__(
        self,
        dataset=None,
        model=AffiNETy_PL_P_L,
        visnet_pl=ViSNet(),
        visnet_p=ViSNet(),
        visnet_l=ViSNet(),
        lr=1e-4,
        use_GPU=False,
    ):
        self.model = model(visnet_pl, visnet_p, visnet_l)
        self.dataset = dataset
        self.use_GPU = use_GPU
        return

    @record
    def train(
        self, epochs=100, batch_size=4, lr=0.01, split_percent=0.8, verbose=True
    ):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided!")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_dataset = self.dataset[: int(len(self.dataset) * split_percent)]
        test_dataset = self.dataset[int(len(self.dataset) * split_percent) :]
        print(
            f"Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples"
        )
        # print(train_dataset[0], test_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        lowest_test_error = 1000000

        gpu_enabled = False
        dist.init_process_group(backend='gloo')
        self.model = nn.parallel.DistributedDataParallel(self.model)
        local_rank = int(os.environ["LOCAL_RANK"])
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        print("running on the CPU")
        print("Starting training...")
        for epoch in range(epochs):
            train_loss = 0.
            eval_loss = 0.
            for batch in train_loader:
                preds, true = [], []
                optimizer.zero_grad()
                for data in batch.to_data_list():
                    out = self.model(data)
                    preds.append(out.item())
                    true.append(data.y[0])
                preds = torch.tensor(preds, requires_grad=True)
                true = torch.tensor(true, requires_grad=True)
                loss = criterion(preds, true)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            with torch.no_grad():
                for batch in test_loader:
                    preds, true = [], []
                    for data in batch.to_data_list():
                        self.model.eval()
                        out = self.model(data)
                        preds.append(out.item())
                        true.append(data.y[0])
                    loss = criterion(preds, true)
                    eval_loss += loss.item()
                    if loss < lowest_test_error:
                        torch.save(model, "models/AffiNETy")
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {test_loss.item()}")
            print(f"Epoch {epoch+1}/{epochs}")
        return
