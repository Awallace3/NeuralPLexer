import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ViSNet, GraphSAGE

import torch.distributed as dist


class AffiNETy_PL_P_L(nn.Module):
    def __init__(
        self,
        # could try to reduce cutoff to 3.0, hidden channels 128->64,
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

    def forward(self, data, device):
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
        pl_es = torch.zeros(len(data.pl_z), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_z), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_z[i]), dtype=torch.int64, device=device)
            l_es[i] = self.visnet_l(
                z=data.l_z[i].to(device),
                pos=data.l_pos[i].to(device),
                batch=batch,
            )[0]
        for i in range(len(data.pl_z)):
            batch = torch.zeros(len(data.pl_z[i]), dtype=torch.int64, device=device)
            pl_es[i] = self.visnet_pl(
                z=data.pl_z[i].to(device),
                pos=data.pl_pos[i].to(device),
                batch=batch,
            )[0]
        for i in range(len(data.p_z)):
            batch = torch.zeros(len(data.p_z[i]), dtype=torch.int64, device=device)
            p_es[i] = self.visnet_p(
                z=data.p_z[i].to(device),
                pos=data.p_pos[i].to(device),
                batch=batch,
            )[0]

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        result = torch.mean(pl_es) - torch.mean(p_es) - torch.mean(l_es) / -RT
        return result


class AffiNETy_graphSage(nn.Module):
    def __init__(
        self,
        pl_model=GraphSAGE(-1, 5, 3),
        p_model=GraphSAGE(-1, 5, 3),
        l_model=GraphSAGE(-1, 5, 3),
        # l_model=ViSNet(),
        temperature=298.0,
        pl_in=2,
        p_in=2,
        l_in=20,
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
        super(AffiNETy_graphSage, self).__init__()
        self.pl_model = pl_model
        self.p_model = p_model
        self.l_model = l_model
        self.temperature = temperature

        self.pl_n1 = nn.Linear(pl_in, 32)
        self.pl_relu1 = nn.ReLU()
        self.pl_n2 = nn.Linear(32, 32)
        self.pl_relu2 = nn.ReLU()
        self.pl_n3 = nn.Linear(32, 1)
        self.pl_relu3 = nn.ReLU()

        self.p_n1 = nn.Linear(p_in, 32)
        self.p_relu1 = nn.ReLU()
        self.p_n2 = nn.Linear(32, 32)
        self.p_relu2 = nn.ReLU()
        self.p_n3 = nn.Linear(32, 1)
        self.p_relu3 = nn.ReLU()

        self.l_n1 = nn.Linear(l_in, 32)
        self.l_relu1 = nn.ReLU()
        self.l_n2 = nn.Linear(32, 32)
        self.l_relu2 = nn.ReLU()
        self.l_n3 = nn.Linear(32, 1)
        self.l_relu3 = nn.ReLU()

    def forward(self, data, device):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        # print(data)
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        # for i in range(len(data.l_z)):
        #     batch = torch.zeros(len(data.l_z[i]), dtype=torch.int64, device=device)
        #     l_es[i] = self.l_model(
        #         z=data.l_z[i].to(device),
        #         pos=data.l_pos[i].to(device),
        #         batch=batch,
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                # x=x,
                # edge_index=edge_index,
                # edge_attr=edge_attr,
                # z=data.l_z[i].to(device),
                # pos=data.l_pos[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            # x = torch.tensor(data.pl_x[i], dtype=torch.float).to(device)
            # edge_index = torch.tensor(data.pl_edge_index[i], dtype=torch.int64).to(
            #     device
            # )
            # edge_attr = torch.tensor(data.pl_edge_attr[i], dtype=torch.float).to(device)
            # print(x)
            # print(edge_index)
            # print(edge_attr)

            # pl_es[i] = self.pl_model(
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                # x=x,
                # edge_index=edge_index,
                # edge_attr=edge_attr,
                # z=data.pl_z[i].to(device),
                # pos=data.pl_pos[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
            # print(f'\n{pl_es[i] = }\n')
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            # x = torch.tensor(data.pl_x[i], dtype=torch.float).to(device)
            # edge_index = torch.tensor(data.pl_edge_index[i], dtype=torch.int64).to(
            #     device
            # )
            # edge_attr = torch.tensor(data.pl_edge_attr[i], dtype=torch.float).to(device)
            out = self.p_model(
                # x=x,
                # edge_index=edge_index,
                # edge_attr=edge_attr,
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                # z=data.p_z[i].to(device),
                # pos=data.p_pos[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)

        # print(f"{pl_es = }\n{p_es = }\n{l_es}\n\n")

        pl_e_avg =  self.pl_relu3(self.pl_n3(self.pl_relu2(self.pl_n2(self.pl_relu1(self.pl_n1(pl_es))))))
        p_e_avg =  self.p_relu3(self.p_n3(self.p_relu2(self.p_n2(self.p_relu1(self.p_n1(p_es))))))
        l_e_avg =  self.l_relu3(self.l_n3(self.l_relu2(self.l_n2(self.l_relu1(self.l_n1(l_es))))))

        # print(f"{pl_e_avg = }\n{p_e_avg = }\n{l_e_avg}\n\n")

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        # result = -torch.log((pl_e_avg - p_e_avg - l_e_avg) / -RT)
        # result = -torch.log(pl_e_avg - p_e_avg - l_e_avg)
        # result = torch.log(-(pl_e_avg - p_e_avg - l_e_avg))
        result = (pl_e_avg - p_e_avg - l_e_avg)
        return result


from torch.distributed.elastic.multiprocessing.errors import record
import os


class AffiNETy:
    def __init__(
        self,
        dataset=None,
        model=AffiNETy_graphSage,
        pl_model=GraphSAGE(
            in_channels=-1, hidden_channels=5, num_layers=3, out_channels=1, jk="lstm"
        ),
        p_model=GraphSAGE(
            in_channels=-1, hidden_channels=5, num_layers=3, out_channels=1, jk="lstm"
        ),
        # l_model=ViSNet(),
        l_model=GraphSAGE(
            in_channels=-1, hidden_channels=5, num_layers=3, out_channels=1, jk="lstm"
        ),
        lr=1e-4,
        use_GPU=False,
        num_workers=1,
    ):
        self.model = model(pl_model, p_model, l_model)
        self.dataset = dataset
        self.use_GPU = use_GPU
        self.num_workers = num_workers
        return

    @record
    def train(self, epochs=100, batch_size=2, split_percent=0.8, verbose=True):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided!")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print(self.model.parameters())
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        criterion = nn.MSELoss()
        train_dataset = self.dataset[: int(len(self.dataset) * split_percent)]
        test_dataset = self.dataset[int(len(self.dataset) * split_percent) :]
        print(
            f"Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples"
        )
        print(train_dataset[0], test_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        lowest_test_error = 1000000

        if torch.cuda.is_available() and self.use_GPU:
            gpu_enabled = True
            device = torch.device("cuda:0")
            print("running on the GPU")
            # self.model = self.model.to(device)
            self.model = self.model.cuda()
            self.model = self.model.to(device)
            # dist.init_process_group(backend='nccl')
            # self.model = nn.parallel.DistributedDataParallel(self.model)
        else:
            gpu_enabled = False
            device = torch.device("cpu")
            # dist.init_process_group(backend='gloo')
            # self.model = nn.parallel.DistributedDataParallel(self.model)
            # local_rank = int(os.environ["LOCAL_RANK"])
            # self.model = torch.nn.parallel.DistributedDataParallel(
            #     self.model,
            #     device_ids=[local_rank],
            #     output_device=local_rank,
            # )
            print("running on the CPU")
        print("Starting training...")
        for epoch in range(epochs):
            train_loss = 0.0
            eval_loss = 0.0
            self.model.train()
            for batch in train_loader:
                preds, true = [], []
                optimizer.zero_grad()
                for data in batch.to_data_list():
                    out = self.model(data, device)
                    preds.append(out.item())
                    true.append(data.y[0])
                    # print(f"pred {out.item()}, true {data.y[0]}")
                preds = torch.tensor(preds, requires_grad=True)
                true = torch.tensor(true, requires_grad=True)
                loss = criterion(preds, true)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            self.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    preds, true = [], []
                    for data in batch.to_data_list():
                        out = self.model(data, device)
                        preds.append(out.item())
                        true.append(data.y[0])
                    preds = torch.tensor(preds, requires_grad=True)
                    true = torch.tensor(true, requires_grad=True)
                    loss = criterion(preds, true)
                    eval_loss += loss.item()
            eval_loss /= len(test_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train: {train_loss}, Loss: {eval_loss}")
            if eval_loss < lowest_test_error:
                torch.save(self.model, "models/AffiNETy")
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
        return
