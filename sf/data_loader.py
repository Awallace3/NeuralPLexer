import os
import ase
import subprocess
from ase import neighborlist
from ase.io import read

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn.models import ViSNet
from rdkit.Chem import AllChem
from rdkit import Chem
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Process the PDBBIND setting")
parser.add_argument('--pdbbind', dest='PDBBIND', action='store_true',
                    help='Disable PDBBIND (default: enabled)')
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
    v = 'casf'



def mol_to_arrays(mol: Chem.Mol):
    conf = mol.GetConformer()  # Get the 3D conformation of the molecule
    atom_positions = [conf.GetAtomPosition(atom_idx) for atom_idx in range(mol.GetNumAtoms())]
    coords_array = np.array([[pos.x, pos.y, pos.z] for pos in atom_positions])
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    return coords_array, atomic_numbers

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
            atom_positions = np.array([list(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())])
            molecules_data.append(atom_positions)
        else:
            molecules_data.append(None)
            print(f"{sdf_file_path} not processed")
    return molecules_data

def ase_to_ViSNet_data(ase_mol,
    cutoff = 8.0, # electrostatics should be nearly 0 by 8 angstroms
    node_dim = 100,
                       ):
    z = torch.tensor(ase_mol.get_atomic_numbers(), dtype=torch.int)
    pos = torch.tensor(ase_mol.get_positions(), dtype=torch.float)
    source, target, distances = neighborlist.neighbor_list(
        "ijd", ase_mol, cutoff=cutoff, self_interaction=False
    )
    x = F.one_hot(torch.tensor(ase_mol["atomic_numbers"]) - 1, num_classes=node_dim).float()
    edge_index = torch.tensor(np.array([source, target]), dtype=torch.float)
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)
    return x, edge_index, edge_attr, z, pos


def gather_data():
    # Since not all ligands converted to RDKit for Torsional Diffusion, we will
    # use their pdb_id's to dictate which datapoints to create

    data_list = []
    print('starting')
    df_lig = pd.read_pickle(l_pkl)
    print(df_lig.columns.values)
    pdb_ids = df_lig['pdb_id'].to_list()
    num_confs = range(df_lig.iloc[0]['num_conformers'])
    for i in pdb_ids:
        pls = {
            "x": [],
            "edge_index": [],
            "edge_attr": [],
            "z": [],
            "pos": [],
        }
        ps = pls.copy()
        ls = pls.copy()
        print(f'pdb_id: {i}')
        for j in num_confs:
            # PL
            pl_pro = read(f"{pl_dir}/{i}/prot_{j}.pdb")
            pl_lig = read(f"{pl_dir}/{i}/lig_{j}.sdf")
            pl = pl_pro + pl_lig
            x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(pl)
            pls['x'] = x
            pls['edge_index'] = edge_index
            pls['edge_attr'] = edge_attr
            pls['z'] = z
            pls['pos'] = pos
            # P
            p = read(f"{p_dir}/{i}/prot_{j}.pdb")
            x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(p)
            ps['x'] = x
            ps['edge_index'] = edge_index
            ps['edge_attr'] = edge_attr
            ps['z'] = z
            ps['pos'] = pos
            # P
            l = read(f"{p_dir}/{i}/prot_{j}.pdb")
            x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(l)
            ls['x'] = x
            ls['edge_index'] = edge_index
            ls['edge_attr'] = edge_attr
            ls['z'] = z
            ls['pos'] = pos

        # create data object
        _d = Data(
           # pl
           pl_x=pl['x'],
           pl_edge_index=pl['edge_index'],
           pl_edge_attr=pl['edge_attr'],
           pl_z=pl['z'],
           pl_pos=pl['pos'],
            # p
           p_x=p['x'],
           p_edge_index=p['edge_index'],
           p_edge_attr=p['edge_attr'],
           p_z=p['z'],
           p_pos=p['pos'],
            # l
           l_x=l['x'],
           l_edge_index=l['edge_index'],
           l_edge_attr=l['edge_attr'],
           l_z=l['z'],
           l_pos=l['pos'],
        )
        print(_d)
        break
    return

class AffiNETy_dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, r_cut=5.0, dataset='casf2016'):
        super(atomic_module_dataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

    @property
    def raw_file_names(self):
        return [
            f"{self.dataset}.pkl",
        ]

    @property
    def processed_file_names(self):
        # return ["data_46628.pt", "data_46629.pt"]
        return [f"{self.dataset}_{i}.pt" for i in range(MAX_SIZE)]

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    # TODO update process with gather_data() function call...
    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            print(f"raw_path: {raw_path}")
            # Read data from `raw_path`.
            monomers, cartesian_multipoles, total_charge = util.load_monomer_dataset(raw_path, MAX_SIZE)
            for i in range(len(monomers)):
                if i % 1000 == 0:
                    print(f"{i}/{len(monomers)}")
                mol = monomers[i]
                R = mol.geometry
                Z = mol.atomic_numbers
                node_features = np.array(Z)
                # node_features = np.zeros((len(Z), MAX_Z))
                # for n, z in enumerate(Z):
                #     node_features[n, z] = 1.0
                node_features = torch.tensor(node_features)
                edge_index, edge_feature_vector = edge_function_system(R, 5.0)
                edge_index = torch.tensor(edge_index).t()
                edge_feature_vector = torch.tensor(edge_feature_vector).view(-1, 8)
                cartesian_mult = torch.tensor(cartesian_multipoles[i])

                R = torch.tensor(R)
                if idx == 0:
                    print('edge_index', edge_index.size())
                    print('edge_feature_vector', edge_feature_vector.size())
                    print('cartesian_mult', cartesian_mult.size())

                data = Data(
                    x=node_features,
                    edge_attr=edge_feature_vector,
                    edge_index=edge_index,
                    y=cartesian_mult,
                    R=R,
                    molecule_ind=len(R),
                    total_charge=total_charge[i],
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
                idx += 1
                if idx > MAX_SIZE:
                    break

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

def main():
    gather_data()
    return


if __name__ == "__main__":
    main()

