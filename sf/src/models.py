import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ViSNet, GraphSAGE
from torchmdnet.models.model import load_model

import torch.distributed as dist
import sys

def torch_gpu_memory():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Select your GPU device if multiple GPUs are available
        # device = torch.device('cuda:0')  # Default to first GPU if specific not specified
        # Alternatively, specify the device index directly in the memory functions

        # Get the total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory  # In bytes

        # Get the current GPU memory usage by tensors in bytes
        allocated_memory = torch.cuda.memory_allocated(0)

        # Get the current GPU cached memory in bytes
        cached_memory = torch.cuda.memory_reserved(0)

        # Convert bytes to gigabytes for easier interpretation
        total_memory_gb = total_memory / (1024**3)
        allocated_memory_gb = allocated_memory / (1024**3)
        cached_memory_gb = cached_memory / (1024**3)

        # Print memory usage
        print(f"  Total GPU Memory (GB): {total_memory_gb:.2f}")
        print(f"  Allocated Memory (GB): {allocated_memory_gb:.2f}")
        print(f"  Cached Memory (GB): {cached_memory_gb:.2f}")
    else:
        print("CUDA is not available")

def boltzmannFactorRescaling(ensemble_energies: torch.tensor):
    RT = 1.98720425864083 / 1000 * 298  # (kcal * K) / (mol * K)
    lowest_e = torch.min(ensemble_energies)
    boltzmann_factor = torch.exp((torch.sub(ensemble_energies, lowest_e)) / -RT)
    ensemble_energies = torch.mul(ensemble_energies.reshape(-1, 1), boltzmann_factor)
    return ensemble_energies


def boltzmannFactorQ(ensemble_energies: torch.tensor):
    RT = 1.98720425864083 / 1000 * 298  # (kcal * K) / (mol * K)
    Q = torch.sum(torch.exp(-RT * ensemble_energies))
    ensemble_energies = (
        1
        / Q
        * torch.sum(
            torch.mul(
                torch.exp(-RT * ensemble_energies).reshape(-1, 1), ensemble_energies
            )
        )
    )
    return ensemble_energies


class AffiNETy_PL_P_L(nn.Module):
    def __init__(
        self,
        # could try to reduce cutoff to 3.0, hidden channels 128->64,
        visnet_pl=ViSNet(cutoff=5.0, num_heads=8, num_layers=6, ),
        visnet_p=ViSNet( cutoff=5.0, num_heads=8, num_layers=6, ),
        visnet_l=ViSNet( cutoff=5.0, num_heads=8, num_layers=6, ),
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


class AffiNETy_graphSage_MLP(nn.Module):
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
        super(AffiNETy_graphSage_MLP, self).__init__()
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
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            out = self.p_model(
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)

        pl_e_avg = self.pl_relu3(
            self.pl_n3(self.pl_relu2(self.pl_n2(self.pl_relu1(self.pl_n1(pl_es)))))
        )
        p_e_avg = self.p_relu3(
            self.p_n3(self.p_relu2(self.p_n2(self.p_relu1(self.p_n1(p_es)))))
        )
        l_e_avg = self.l_relu3(
            self.l_n3(self.l_relu2(self.l_n2(self.l_relu1(self.l_n1(l_es)))))
        )

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        result = pl_e_avg - p_e_avg - l_e_avg
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

    def model_output(self):
        return "AffiNETy_graphSage"

    def forward(self, data, device):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            out = self.p_model(
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)
        pl_e_avg = torch.mean(pl_es)
        p_e_avg = torch.mean(p_es)
        l_e_avg = torch.mean(l_es)

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        val = (pl_e_avg - p_e_avg - l_e_avg) / -RT
        if val > 0:
            result = torch.log(val)
        return result


class AffiNETy_graphSage_boltzmann_mlp(nn.Module):
    def __init__(
        self,
        pl_model=GraphSAGE(-1, 5, 3),
        p_model=GraphSAGE(-1, 5, 3),
        l_model=GraphSAGE(-1, 5, 3),
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
        super(AffiNETy_graphSage_boltzmann_mlp, self).__init__()
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

    def model_output(self):
        return "AffiNETy_graphSage_boltzmann_mlp"

    def forward(self, data, device):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            out = self.p_model(
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)

        pl_e_avg = self.pl_relu3(
            self.pl_n3(self.pl_relu2(self.pl_n2(self.pl_relu1(self.pl_n1(pl_es)))))
        )
        p_e_avg = self.p_relu3(
            self.p_n3(self.p_relu2(self.p_n2(self.p_relu1(self.p_n1(p_es)))))
        )
        l_e_avg = self.l_relu3(
            self.l_n3(self.l_relu2(self.l_n2(self.l_relu1(self.l_n1(l_es)))))
        )

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        val = (pl_e_avg - p_e_avg - l_e_avg) / -RT
        # if val > 0:
        #     val = torch.log(val)
        return val

model_path = "/storage/ice1/7/3/awallace43/torchmd_data/epoch=2139-val_loss=0.2543-test_loss=0.2317.ckpt"

def detach_except_min(tensor_list):
    # Check if the list is not empty and elements are indeed tensors
    min_value = float('inf')
    min_index = -1
    # Loop to find the minimum value tensor
    for i, tensor in enumerate(tensor_list):
        current_min = tensor.min().item()  # Get the minimum value in the current tensor
        if current_min < min_value:
            min_value = current_min
            min_index = i

    # Detach all tensors except the one with the minimum value
    for i, tensor in enumerate(tensor_list):
        if i != min_index:
            tensor_list[i] = tensor.detach()  # Detach the tensor

    return tensor_list

class AffiNETy_ViSNet_boltzmann_mlp(nn.Module):
    def __init__(
        self,
        pl_model=load_model(model_path),
        p_model= load_model(model_path),
        l_model= load_model(model_path),
        temperature=298.0,
        pl_in=8,
        p_in=8,
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
        super(AffiNETy_ViSNet_boltzmann_mlp, self).__init__()
        self.pl_model = pl_model
        # self.p_model = p_model
        # self.l_model = l_model
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

    def model_output(self):
        return "AffiNETy_ViSNet_boltzmann_mlp"

    def forward(self, data, device, verbose=False):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        # pl_es = torch.zeros(len(data.pl_z), dtype=torch.float, device=device)
        # p_es = torch.zeros(len(data.p_z), dtype=torch.float, device=device)
        # l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        # pl_es = torch.zeros(len(data.pl_z), dtype=torch.float, device='cpu')
        # p_es = torch.zeros(len(data.p_z), dtype=torch.float, device='cpu')
        # l_es = torch.zeros(len(data.l_z), dtype=torch.float, device='cpu')
        pl_es, p_es, l_es = [],[],[]
        # with torch.autograd.graph.save_on_cpu():
        print("\nsize:", len(data.pl_z[0]))
        # torch_gpu_memory()
        for i in range(len(data.l_z)):
            z = data.l_z[i].clone().to(device)
            pos = data.l_pos[i].clone().to(device)
            batch = torch.zeros(len(data.l_z[i]), dtype=torch.int64, device=device)
            energies, forces = self.pl_model(
                z,
                pos,
                batch,
            )
            # energies = energies.cpu()
            # l_es[i] = energies
            l_es.append(energies.view(-1))
            del energies, forces
            del z
            del pos
            # torch.cuda.empty_cache()  # Clears cached memory
        if verbose:
            print(f"{l_es = }")
            torch_gpu_memory()
        for i in range(len(data.pl_z)):
            if verbose:
                print(f"Start {i = }")
                torch_gpu_memory()
            z = data.pl_z[i].to(device)
            pos = data.pl_pos[i].to(device)
            batch = torch.zeros(len(data.pl_z[i]), dtype=torch.int64, device=device)
            energies, forces = self.pl_model(
                z,
                pos,
                batch,
            )
            if verbose:
                print("pre-delete")
                torch_gpu_memory()
            # energies = energies.cpu()
            # cannot store grads in cpu memory or gpu completely... Using detach removes the model graphs, effectively not allowing backprop to update these submodels.
            # energies = energies.detach().cpu()
            # energies = energies.cpu()
            # pl_es[i] = energies
            pl_es.append(energies.view(-1))
            pl_es = detach_except_min(pl_es)
            del energies, forces
            del z
            del pos
            # torch.cuda.empty_cache()  # Clears cached memory
        if verbose:
            print(f"{pl_es = }")
            torch_gpu_memory()
        for i in range(len(data.p_z)):
            if verbose:
                print(f"Start {i = }")
                torch_gpu_memory()
            batch = torch.zeros(len(data.p_z[i]), dtype=torch.int64, device=device)
            z = data.p_z[i].to(device)
            pos = data.p_pos[i].to(device)
            energies, forces  = self.pl_model(
                z,
                pos,
                batch,
            )
            # energies = energies.cpu()
            # p_es[i] = energies
            p_es.append(energies.view(-1))
            p_es = detach_except_min(p_es)
            if verbose:
                print("pre-delete")
                torch_gpu_memory()
            del energies, forces
            del z
            del pos
            # torch.cuda.empty_cache()  # Clears cached memory
        if verbose:
            torch_gpu_memory()
            print(f"{p_es = }")

        pl_es, _ = torch.sort(torch.cat(pl_es))
        p_es, _ = torch.sort( torch.cat(p_es))
        l_es, _ = torch.sort( torch.cat(l_es))
        print(pl_es)
        print(p_es)
        print(l_es)

        pl_e_avg = self.pl_relu3(
            self.pl_n3(self.pl_relu2(self.pl_n2(self.pl_relu1(self.pl_n1(pl_es)))))
        )
        p_e_avg = self.p_relu3(
            self.p_n3(self.p_relu2(self.p_n2(self.p_relu1(self.p_n1(p_es)))))
        )
        l_e_avg = self.l_relu3(
            self.l_n3(self.l_relu2(self.l_n2(self.l_relu1(self.l_n1(l_es)))))
        )

        print(f"{pl_e_avg.item()} - ({p_e_avg.item()} + {l_e_avg.item()})")

        R  =  1.98720425864083 / 1000
        RT = R * self.temperature  # (kcal * K) / (mol * K)
        # k_BT = 1.38e-23 * self.temperature
        dS = -R * torch.log(data.l_num_rot_bonds.to(device))
        # print(f"{dS = }")
        val = (pl_e_avg - p_e_avg - l_e_avg + dS) / -RT
        print(f'{val.item() = }')
        if val > 0:
            val = torch.log(val)
        return val

class AffiNETy_equivariant_torchmdnet_boltzmann_mlp(nn.Module):
    def __init__(
        self,
        pl_model=load_model(model_path),
        p_model= load_model(model_path),
        l_model= load_model(model_path),
        temperature=298.0,
        pl_in=8,
        p_in=8,
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
        super(AffiNETy_equivariant_torchmdnet_boltzmann_mlp, self).__init__()
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

    def model_output(self):
        return "AffiNETy_equivariant_torchmdnet_boltzmann_mlp"

    def forward(self, data, device, verbose=False):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        # pl_es = torch.zeros(len(data.pl_z), dtype=torch.float, device=device)
        # p_es = torch.zeros(len(data.p_z), dtype=torch.float, device=device)
        # l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        # pl_es = torch.zeros(len(data.pl_z), dtype=torch.float, device='cpu')
        # p_es = torch.zeros(len(data.p_z), dtype=torch.float, device='cpu')
        # l_es = torch.zeros(len(data.l_z), dtype=torch.float, device='cpu')
        pl_es, p_es, l_es = [],[],[]
        # with torch.autograd.graph.save_on_cpu():
        print("\nsize:", len(data.pl_z[0]))
        # torch_gpu_memory()
        for i in range(len(data.l_z)):
            z = data.l_z[i].clone().to(device)
            pos = data.l_pos[i].clone().to(device)
            energies, forces = self.l_model(
                z,
                pos,
            )
            # energies = energies.cpu()
            # l_es[i] = energies
            l_es.append(energies.view(-1))
            del energies, forces
            del z
            del pos
            torch.cuda.empty_cache()  # Clears cached memory
        if verbose:
            print(f"{l_es = }")
            torch_gpu_memory()
        for i in range(len(data.pl_z)):
            if verbose:
                print(f"Start {i = }")
                torch_gpu_memory()
            z = data.pl_z[i].to(device)
            pos = data.pl_pos[i].to(device)
            energies, forces = self.pl_model(
                z,
                pos,
            )
            if verbose:
                print("pre-delete")
                torch_gpu_memory()
            # energies = energies.cpu()
            # cannot store grads in cpu memory or gpu completely... Using detach removes the model graphs, effectively not allowing backprop to update these submodels.
            # energies = energies.detach().cpu()
            # energies = energies.cpu()
            # pl_es[i] = energies
            pl_es.append(energies.view(-1))
            pl_es = detach_except_min(pl_es)
            del energies, forces
            del z
            del pos
            torch.cuda.empty_cache()  # Clears cached memory
        if verbose:
            print(f"{pl_es = }")
            torch_gpu_memory()
        for i in range(len(data.p_z)):
            if verbose:
                print(f"Start {i = }")
                torch_gpu_memory()
            z = data.p_z[i].to(device)
            pos = data.p_pos[i].to(device)
            energies, forces  = self.p_model(
                z,
                pos,
            )
            # energies = energies.cpu()
            # p_es[i] = energies
            p_es.append(energies.view(-1))
            p_es = detach_except_min(p_es)
            if verbose:
                print("pre-delete")
                torch_gpu_memory()
            del energies, forces
            del z
            del pos
            torch.cuda.empty_cache()  # Clears cached memory
        if verbose:
            torch_gpu_memory()
            print(f"{p_es = }")

        pl_es, _ = torch.sort(torch.cat(pl_es))
        p_es, _ = torch.sort( torch.cat(p_es))
        l_es, _ = torch.sort( torch.cat(l_es))

        pl_e_avg = self.pl_relu3(
            self.pl_n3(self.pl_relu2(self.pl_n2(self.pl_relu1(self.pl_n1(pl_es)))))
        )
        p_e_avg = self.p_relu3(
            self.p_n3(self.p_relu2(self.p_n2(self.p_relu1(self.p_n1(p_es)))))
        )
        l_e_avg = self.l_relu3(
            self.l_n3(self.l_relu2(self.l_n2(self.l_relu1(self.l_n1(l_es)))))
        )

        # print(f"{pl_e_avg} - ({p_e_avg + l_e_avg})")

        R  =  1.98720425864083 / 1000
        RT = R * self.temperature  # (kcal * K) / (mol * K)
        # k_BT = 1.38e-23 * self.temperature
        # dS = -R * torch.log(data.l_num_rot_bonds.to(device))
        print(f"{dS = }")
        val = (pl_e_avg - p_e_avg - l_e_avg) / -RT
        print(f'{val = }')
        if val > 0:
            val = torch.log(val)
        return val


class AffiNETy_graphSage_boltzmann_avg_Q(nn.Module):
    def __init__(
        self,
        pl_model=GraphSAGE(-1, 5, 3),
        p_model=GraphSAGE(-1, 5, 3),
        l_model=GraphSAGE(-1, 5, 3),
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
        super(AffiNETy_graphSage_boltzmann_avg_Q, self).__init__()
        self.pl_model = pl_model
        self.p_model = p_model
        self.l_model = l_model
        self.temperature = temperature

    def forward(self, data, device):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            out = self.p_model(
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)

        pl_e_avg = torch.mean(boltzmannFactorRescaling(pl_es))
        p_e_avg = torch.mean(boltzmannFactorRescaling(p_es))
        l_e_avg = torch.mean(boltzmannFactorRescaling(l_es))

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        val = (pl_e_avg - p_e_avg - l_e_avg) / RT
        if val > 0:
            val = torch.log(val)
        return val

    def model_output(self):
        return "AffiNETy_graphSage_boltzmann_avg_Q"


class AffiNETy_graphSage_boltzmann_avg(nn.Module):
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
        super(AffiNETy_graphSage_boltzmann_avg, self).__init__()
        self.pl_model = pl_model
        self.p_model = p_model
        self.l_model = l_model
        self.temperature = temperature

    def model_output(self):
        return "AffiNETy_graphSage_boltzmann_avg"

    def forward(self, data, device):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            out = self.p_model(
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)

        pl_ens = boltzmannFactorRescaling(pl_es)
        pl_e_avg = torch.mean(pl_ens)
        p_ens = boltzmannFactorRescaling(p_es)
        p_e_avg = torch.mean(p_es)
        l_ens = boltzmannFactorRescaling(l_es)
        l_e_avg = torch.mean(l_es)

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        val = (pl_e_avg - p_e_avg - l_e_avg) / -RT
        if val > 0:
            val = torch.log(val)
        return val


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

        # self.pl_n1 = nn.Linear(pl_in, 32)
        # self.pl_relu1 = nn.ReLU()
        # self.pl_n2 = nn.Linear(32, 32)
        # self.pl_relu2 = nn.ReLU()
        # self.pl_n3 = nn.Linear(32, 1)
        # self.pl_relu3 = nn.ReLU()
        #
        # self.p_n1 = nn.Linear(p_in, 32)
        # self.p_relu1 = nn.ReLU()
        # self.p_n2 = nn.Linear(32, 32)
        # self.p_relu2 = nn.ReLU()
        # self.p_n3 = nn.Linear(32, 1)
        # self.p_relu3 = nn.ReLU()
        #
        # self.l_n1 = nn.Linear(l_in, 32)
        # self.l_relu1 = nn.ReLU()
        # self.l_n2 = nn.Linear(32, 32)
        # self.l_relu2 = nn.ReLU()
        # self.l_n3 = nn.Linear(32, 1)
        # self.l_relu3 = nn.ReLU()

    def forward(self, data, device):
        """
        Forward pass through the model.

        Parameters:
        - data_list: A list of Data objects, where each object contains fields for PL and L graphs.

        Returns:
        - torch.Tensor: The predicted output values.
        """
        pl_es = torch.zeros(len(data.pl_x), dtype=torch.float, device=device)
        p_es = torch.zeros(len(data.p_x), dtype=torch.float, device=device)
        l_es = torch.zeros(len(data.l_z), dtype=torch.float, device=device)
        for i in range(len(data.l_z)):
            batch = torch.zeros(len(data.l_x[i]), dtype=torch.int64, device=device)
            out = self.l_model(
                x=data.l_x[i].to(device),
                edge_index=data.l_edge_index[i].to(device),
                edge_attr=data.l_edge_attr[i].to(device),
                batch=batch,
            )
            l_es[i] = torch.sum(out)
        for i in range(len(data.pl_x)):
            batch = torch.zeros(len(data.pl_x[i]), dtype=torch.int64, device=device)
            out = self.pl_model(
                x=data.pl_x[i].to(device),
                edge_index=data.pl_edge_index[i].to(device),
                edge_attr=data.pl_edge_attr[i].to(device),
                batch=batch,
            )
            pl_es[i] = torch.sum(out)
        for i in range(len(data.p_x)):
            batch = torch.zeros(len(data.p_x[i]), dtype=torch.int64, device=device)
            out = self.p_model(
                x=data.p_x[i].to(device),
                edge_index=data.p_edge_index[i].to(device),
                edge_attr=data.p_edge_attr[i].to(device),
                batch=batch,
            )
            p_es[i] = torch.sum(out)

        pl_es, _ = torch.sort(pl_es)
        p_es, _ = torch.sort(p_es)
        l_es, _ = torch.sort(l_es)
        pl_e_avg = torch.mean(pl_es)
        p_e_avg = torch.mean(p_es)
        l_e_avg = torch.mean(l_es)

        # pl_e_avg =  self.pl_relu3(self.pl_n3(self.pl_relu2(self.pl_n2(self.pl_relu1(self.pl_n1(pl_es))))))
        # p_e_avg =  self.p_relu3(self.p_n3(self.p_relu2(self.p_n2(self.p_relu1(self.p_n1(p_es))))))
        # l_e_avg =  self.l_relu3(self.l_n3(self.l_relu2(self.l_n2(self.l_relu1(self.l_n1(l_es))))))

        RT = 1.98720425864083 / 1000 * self.temperature  # (kcal * K) / (mol * K)
        result = (pl_e_avg - p_e_avg - l_e_avg) / -RT
        return result


from torch.distributed.elastic.multiprocessing.errors import record
import os


class AffiNETy:
    def __init__(
        self,
        dataset=None,
        model=AffiNETy_graphSage,
        pl_model=GraphSAGE(
            in_channels=-1, hidden_channels=6, num_layers=4, out_channels=1, jk="max"
        ),
        p_model=GraphSAGE(
            in_channels=-1, hidden_channels=6, num_layers=4, out_channels=1, jk="max"
        ),
        l_model=GraphSAGE(
            in_channels=-1, hidden_channels=6, num_layers=4, out_channels=1, jk="max"
        ),
        lr=1e-4,
        use_GPU=False,
        num_workers=1,
        pl_in=2,
        p_in=2,
        l_in=20,
    ):
        self.model = model(
            pl_model, p_model, l_model, p_in=p_in, pl_in=pl_in, l_in=l_in
        )
        self.dataset = dataset
        self.use_GPU = use_GPU
        self.num_workers = num_workers
        self.lr = lr
        return

    @record
    def train(
        self,
        epochs=100,
        batch_size=2,
        split_percent=0.8,
        verbose=False,
        pre_trained_model=None,
    ):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided!")
        if pre_trained_model:
            if pre_trained_model == 'prev':
                pre_trained_model = f"models/{self.model.model_output()}.pt"
            if os.path.exists(pre_trained_model):
                self.model.load_state_dict(torch.load(pre_trained_model))
            else:
                print("No pre-trained model found... Random start")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print(self.model.parameters())
        if verbose:
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
            for n, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = []
                true = []
                for data in batch.to_data_list():
                    # data = data.to(device)
                    out = self.model(data, device)
                    preds.append(out.view(-1))
                    true.append(data.y.view(-1))
                preds = torch.cat(preds)
                true = torch.cat(true)
                if verbose or n ==2:
                    print(preds)
                    print(true)
                if verbose:
                    torch_gpu_memory()
                loss = criterion(preds, true.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # if n == 0:
                #     sys.exit()
            train_loss /= len(train_loader)

            self.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    preds = []
                    true = []
                    for data in batch.to_data_list():
                        # data = data.to(device)
                        out = self.model(data, device)
                        preds.append(out.view(-1))
                        true.append(data.y.view(-1))
                    preds = torch.cat(preds)
                    true = torch.cat(true)
                    loss = criterion(preds, true)
                    eval_loss += loss.item()
            eval_loss /= len(test_loader)

            print(f"Epoch {epoch+1}/{epochs}, Train: {train_loss}, Eval: {eval_loss}")
            if eval_loss < lowest_test_error:
                torch.save(
                    self.model.state_dict(), f"models/{self.model.model_output()}.pt"
                )
        return

    def eval_casf(
        self,
        batch_size=2,
        pre_trained_model=None,
    ):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided!")
        if pre_trained_model is None:
            raise ValueError("Need to specify a trained model! It can be a path to a model or 'prev' to usethe most recently trained model")
        if pre_trained_model == 'prev':
            pre_trained_model = f"models/{self.model.model_output()}.pt"
        if os.path.exists(pre_trained_model):
            self.model.load_state_dict(torch.load(pre_trained_model))
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        if torch.cuda.is_available() and self.use_GPU:
            gpu_enabled = True
            device = torch.device("cuda:0")
            print("running on the GPU")
            self.model = self.model.cuda()
            self.model = self.model.to(device)
        else:
            gpu_enabled = False
            device = torch.device("cpu")
            print("running on the CPU")
        self.model.eval()
        with torch.no_grad():
            preds = []
            true = []
            for batch in data_loader :
                for data in batch.to_data_list():
                    data = data.to(device)
                    out = self.model(data, device)
                    preds.append(out.view(-1))
                    true.append(data.y.view(-1))
        preds = torch.cat(preds)
        true = torch.cat(true)
        results = result = torch.cat((preds.unsqueeze(0), true.unsqueeze(0)), dim=0)

        if results.is_cuda:
            results = results.cpu()
        results = results.numpy()
        return results

    def predict(
        self,
        batch_size=2,
        pre_trained_model=None,
    ):
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided!")
        if pre_trained_model is None:
            raise ValueError("Need to specify a trained model! It can be a path to a model or 'prev' to usethe most recently trained model")
        if pre_trained_model == 'prev':
            pre_trained_model = f"models/{self.model.model_output()}.pt"
        if os.path.exists(pre_trained_model):
            self.model.load_state_dict(torch.load(pre_trained_model))
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        if torch.cuda.is_available() and self.use_GPU:
            gpu_enabled = True
            device = torch.device("cuda:0")
            print("running on the GPU")
            self.model = self.model.cuda()
            self.model = self.model.to(device)
        else:
            gpu_enabled = False
            device = torch.device("cpu")
            print("running on the CPU")
        self.model.eval()
        with torch.no_grad():
            preds = []
            true = []
            for batch in data_loader :
                for data in batch.to_data_list():
                    data = data.to(device)
                    out = self.model(data, device)
                    preds.append(out.view(-1))
                    true.append(data.y.view(-1))
        preds = torch.cat(preds)
        true = torch.cat(true)
        results = result = torch.cat((preds.unsqueeze(0), true.unsqueeze(0)), dim=0)

        if results.is_cuda:
            results = results.cpu()
        results = results.numpy()
        return results
