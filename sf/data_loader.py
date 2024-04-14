import os
import ase
import subprocess
from ase import neighborlist

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn.models import ViSNet


dirs = glob("*")

cutoff = 8.0
edge_dim = 50
node_dim = 100
data_list = []
print('starting')
for d in dirs:
    print(f'directory: {d}')
    for i in range(10):
        print('converting to xyz')
        path_pro = f'{d}/prot_{i}.pdb'
        path_lig = f'{d}/lig_{i}.sdf'
        cmd = f'obabel -ipdb {path_pro} -oxyz protein.xyz'
        subprocess.run(cmd, shell = True, check = True)
        cmd = f'obabel -isdf {path_lig} -oxyz ligand.xyz'
        subprocess.run(cmd, shell = True, check = True)

        print('reading in ase')
        #read it in
        ligand_atoms = ase.io.read('protein.xyz')
        protein_atoms = ase.io.read('ligand.xyz')
        #make a data point
        combined_atoms = ligand_atoms + protein_atoms

        print('putting it in torch')
        z = torch.tensor(combined_atoms.get_atomic_numbers(), dtype=torch.int) # Required for ViSNet
        pos = torch.tensor(combined_atoms.get_positions(), dtype=torch.float) # Required for ViSNet
        # create edge index and edge weights
        source, target, distances = neighborlist.neighbor_list(
            "ijd", combined_atoms, cutoff=cutoff, self_interaction=False
        )
        # one hot encode the atomic numbers
        x = F.one_hot(torch.tensor(combined_atoms.get_atomic_numbers() - 1, num_classes=node_dim).float())
        # create edge index
        edge_index = torch.tensor(np.array([source, target]), dtype=torch.float)
        edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)

    # create data object
    _d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z, pos=pos)
    data_list.append(_d)
    print('removing files')
    os.remove('protein.xyz')
    os.remove('ligand.xyz')
    break

print(data_list[0])
