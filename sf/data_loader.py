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
from rdkit.Chem import AllChem
from rdkit import Chem

parser = argparse.ArgumentParser(description="Process the PDBBIND setting")
parser.add_argument('--no-pdbbind', dest='PDBBIND', action='store_false',
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

def ase_to_ViSNet_data(ase_mol):
    z = torch.tensor(ase_mol.get_atomic_numbers(), dtype=torch.int)
    pos = torch.tensor(ase_mol.get_positions(), dtype=torch.float)
    source, target, distances = neighborlist.neighbor_list(
        "ijd", ase_mol, cutoff=cutoff, self_interaction=False
    )
    x = F.one_hot(torch.tensor(ase_mol.get_atomic_numbers() - 1, num_classes=node_dim).float())
    edge_index = torch.tensor(np.array([source, target]), dtype=torch.float)
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)
    return x, edge_index, edge_attr, z, pos


def gather_data():
    # Since not all ligands converted to RDKit for Torsional Diffusion, we will
    # use their pdb_id's to dictate which datapoints to create

    cutoff = 8.0 # electrostatics should be nearly 0 by 8 angstroms
    node_dim = 100
    data_list = []
    print('starting')
    df_lig = pd.read_pickle(l_pkl)
    print(df_lig.columns.values)
    pdb_ids = df_lig['pdb_id'].to_list()
    num_confs = range(df_lig.iloc[0]['num_confs'])
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
            pl_pro = ase.io.read(f"{pl_dir}/{i}/prot_{j}.pdb")
            pl_lig = ase.io.read(f"{pl_dir}/{i}/lig_{j}.sdf")
            pl = ligand_atoms + protein_atoms
            x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(pl)
            pls['x'] = x
            pls['edge_index'] = edge_index
            pls['edge_attr'] = edge_attr
            pls['z'] = z
            pls['pos'] = pos
            # P
            p = ase.io.read(f"{p_dir}/{i}/prot_{j}.pdb")
            x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(p)
            ps['x'] = x
            ps['edge_index'] = edge_index
            ps['edge_attr'] = edge_attr
            ps['z'] = z
            ps['pos'] = pos
            # P
            l = ase.io.read(f"{p_dir}/{i}/prot_{j}.pdb")
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


def main():
    gather_data()
    return


if __name__ == "__main__":
    main()

