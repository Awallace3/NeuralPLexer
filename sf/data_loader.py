import os
import ase
import argparse
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from ase import neighborlist
from ase.io import read

import os.path as osp
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


# from torch_geometric.nn.models import ViSNet
from rdkit.Chem import AllChem
from rdkit import Chem

parser = argparse.ArgumentParser(description="Process the PDBBIND setting")
parser.add_argument(
    "--pdbbind",
    dest="PDBBIND",
    action="store_true",
    help="Disable PDBBIND (default: enabled)",
)
args = parser.parse_args()
print(f"PDBBIND setting: {args.PDBBIND}")

PDBBIND = args.PDBBIND
if PDBBIND:
    pl_dir = "/storage/ice1/7/3/awallace43/pdb_gen/pl"
    p_dir = "/storage/ice1/7/3/awallace43/pdb_gen/p"
    l_pkl = "/storage/ice1/7/3/awallace43/pdb_gen/l/pdbbind_final.pkl"
    v = "pdbbind"
else:
    pl_dir = "/storage/ice1/7/3/awallace43/casf2016/pl"
    p_dir = "/storage/ice1/7/3/awallace43/casf2016/p"
    l_pkl = "/storage/ice1/7/3/awallace43/casf2016/l/casf_final.pkl"
    v = "casf"


def mol_to_arrays(mol: Chem.Mol):
    conf = mol.GetConformer()  # Get the 3D conformation of the molecule
    atom_positions = [
        conf.GetAtomPosition(atom_idx) for atom_idx in range(mol.GetNumAtoms())
    ]
    coords_array = torch.tensor([[pos.x, pos.y, pos.z] for pos in atom_positions])
    atomic_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    return coords_array, atomic_numbers

def rdkit_mol_to_ase_atoms(rdkit_mol):
    """
    Convert an RDKit Mol object to an ASE Atoms object.

    Parameters:
    rdkit_mol (rdkit.Chem.Mol): RDKit Mol object

    Returns:
    ase.Atoms: ASE Atoms object
    """
    # Add Hs and compute 3D coordinates if needed
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDG())

    # Extract atomic numbers and positions
    atomic_numbers = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
    positions = [rdkit_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in rdkit_mol.GetAtoms()]
    ase_positions = [(pos.x, pos.y, pos.z) for pos in positions]

    # Create an ASE Atoms object
    ase_atoms = ase.Atoms(numbers=atomic_numbers, positions=ase_positions)

    return ase_atoms


def read_sdf_get_coordinates(sdf_file_path):
    # Create a molecule supplier from the SDF file
    supplier = SDMolSupplier(sdf_file_path)
    molecules_data = []

    # Iterate over each molecule in the SDF file
    for mol in supplier:
        if mol is not None:
            # Assuming each molecule has a single conformation
            conf = mol.GetConformer()
            # Extract atomic coordinates for the conformation
            atom_positions = np.array(
                [list(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
            )
            molecules_data.append(atom_positions)
        else:
            molecules_data.append(None)
            print(f"{sdf_file_path} not processed")
    return molecules_data


def ase_to_ViSNet_data(
    ase_mol,
    cutoff=8.0,  # electrostatics should be nearly 0 by 8 angstroms
    node_dim=100,
):
    z = torch.tensor(ase_mol.get_atomic_numbers(), dtype=torch.int)
    pos = torch.tensor(ase_mol.get_positions(), dtype=torch.float)
    source, target, distances = neighborlist.neighbor_list(
        "ijd", ase_mol, cutoff=cutoff, self_interaction=False
    )
    x = F.one_hot(
        torch.tensor(ase_mol.get_atomic_numbers()) - 1, num_classes=node_dim
    ).float()
    edge_index = torch.tensor(np.array([source, target]), dtype=torch.float)
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)
    return x, edge_index, edge_attr, z, pos


class AffiNETy_dataset(Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None, r_cut=5.0, dataset="casf2016"
    ):
        self.dataset = dataset
        self.df_lig = pd.read_pickle(l_pkl)
        self.pdb_ids = self.df_lig["pdb_id"].to_list()
        self.num_systems = len(self.pdb_ids)
        self.num_confs = range(self.df_lig.iloc[0]["num_conformers"])
        super(AffiNETy_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [
            f"{self.dataset}.pkl",
        ]

    @property
    def processed_file_names(self):
        return [f"{self.dataset}_{i}.pt" for i in range(self.num_systems)]

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process_single_pdb(pdb_id):
        return
    """
    TODO: implement parallelism...
    def process_single_pdb(pdb_id, ...):
        # ... implementation of process for single pdb_id ...

    def process(self):
        with Pool(processes=<number_of_processes>, initializer=initializer) as pool:
            # Passing the task to multiprocessing Pool
            pool.map(process_single_pdb, [(pdb_id, ...) for pdb_id in self.pdb_ids])
    """

    def process(self):
        idx = 0
        print(f"Creating {self.dataset}...")
        for i in self.pdb_ids:
            try:
                pls = {
                    "x": [],
                    "edge_index": [],
                    "edge_attr": [],
                    "z": [],
                    "pos": [],
                }
                ps = pls.copy()
                ls = pls.copy()
                print(f"pdb_id: {i}")
                if os.path.exists(osp.join(self.processed_dir, f"{self.dataset}_{i}.pt")):
                    print('    already processed')
                    continue
                lig = self.df_lig[self.df_lig['pdb_id'] == i]['conformers'].to_list()[0]
                for j in self.num_confs:
                    # PL
                    if (
                        not os.path.exists(f"{pl_dir}/{i}/prot_{j}.pdb")
                        or not os.path.exists(f"{pl_dir}/{i}/lig_{j}.sdf")
                        or not os.path.exists(f"{p_dir}/{i}/prot_{j}.pdb")
                        or not os.path.exists(f"{p_dir}/{i}/prot_{j}.pdb")
                    ):
                        continue
                    pl_pro = read(f"{pl_dir}/{i}/prot_{j}.pdb")
                    pl_lig = read(f"{pl_dir}/{i}/lig_{j}.sdf")
                    pl = pl_pro + pl_lig
                    x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(pl)
                    pls["x"].append(x)
                    pls["edge_index"].append(edge_index)
                    pls["edge_attr"].append(edge_attr)
                    pls["z"].append(z)
                    pls["pos"].append(pos)
                    # P
                    p = read(f"{p_dir}/{i}/prot_{j}.pdb")
                    x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(p)
                    ps["x"].append(x)
                    ps["edge_index"].append(edge_index)
                    ps["edge_attr"].append(edge_attr)
                    ps["z"].append(z)
                    ps["pos"].append(pos)
                    # L
                    l = rdkit_mol_to_ase_atoms(lig[j])
                    x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(l)
                    ls["x"].append(x)
                    ls["edge_index"].append(edge_index)
                    ls["edge_attr"].append(edge_attr)
                    ls["z"].append(z)
                    ls["pos"].append(pos)
                if len(pls['x']) == 0 or len(ps['x']) == 0 or len(ls['x'])==0:
                    continue
                _d = Data(
                    # pl
                    pl_x=pls["x"],
                    pl_edge_index=pls["edge_index"],
                    pl_edge_attr=pls["edge_attr"],
                    pl_z=pls["z"],
                    pl_pos=pls["pos"],
                    # p
                    p_x=ps["x"],
                    p_edge_index=ps["edge_index"],
                    p_edge_attr=ps["edge_attr"],
                    p_z=ps["z"],
                    p_pos=ps["pos"],
                    # l
                    l_x=ls["x"],
                    l_edge_index=ls["edge_index"],
                    l_edge_attr=ls["edge_attr"],
                    l_z=ls["z"],
                    l_pos=ls["pos"],
                )
                if idx == 0:
                    print(_d)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    _d = self.pre_transform(_d)

                torch.save(_d, osp.join(self.processed_dir, f"{self.dataset}_{i}.pt"))
                idx += 1
            except Exception:
                print(f'{i} failed...')
                continue
        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"{self.dataset}_{idx}.pt"))
        return data


def main():
    AffiNETy_dataset(root=f"data_{v}", dataset=v)
    return


if __name__ == "__main__":
    main()
