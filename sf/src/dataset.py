import os
import ase
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from ase import neighborlist
from ase.io import read
from itertools import islice
import pickle

import os.path as osp
from multiprocessing import Pool

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
    positions = [
        rdkit_mol.GetConformer().GetAtomPosition(atom.GetIdx())
        for atom in rdkit_mol.GetAtoms()
    ]
    ase_positions = [(pos.x, pos.y, pos.z) for pos in positions]

    # Create an ASE Atoms object
    ase_atoms = ase.Atoms(numbers=atomic_numbers, positions=ase_positions)

    return ase_atoms


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


def chunks(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    iterator = iter(iterable)
    for first in iterator:  # take first item from iterator
        yield list(islice(iterator, size-1, None))  # then take size - 1 more

class AffiNETy_PL_P_L_dataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        dataset="casf2016",
        NUM_THREADS=1,
        pl_dir="/storage/ice1/7/3/awallace43/casf2016/pl",
        p_dir="/storage/ice1/7/3/awallace43/casf2016/p",
        l_pkl="/storage/ice1/7/3/awallace43/casf2016/l/casf_final.pkl",
        power_ranking_file = "/storage/ice1/7/3/awallace43/CASF-2016/power_ranking/CoreSet.dat",

        chunk_size=None,
        num_confs_protein=None,
        num_confs_ligand=None,
        ensure_processed=True,
    ):

        self.dataset = dataset
        self.df_lig = pd.read_pickle(l_pkl)
        self.pdb_ids = self.df_lig["pdb_id"].to_list()
        self.num_systems = len(self.pdb_ids)
        if num_confs_protein:
            self.num_confs_protein = range(num_confs_protein)
        else:
            self.num_confs_protein = range(self.df_lig.iloc[0]["num_conformers"])
        if num_confs_ligand:
            self.num_confs_ligand = range(num_confs_ligand)
        else:
            self.num_confs_ligand = range(self.df_lig.iloc[0]["num_conformers"])
        self.NUM_THREADS = NUM_THREADS
        print(f"Setting {NUM_THREADS = }")
        self.pl_dir=pl_dir
        self.p_dir=p_dir
        self.power_ranking_file = power_ranking_file
        self.ensure_processed=ensure_processed
        if chunk_size:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = NUM_THREADS * 10

        with open(power_ranking_file, 'rb') as handle:
            self.power_ranking_dict = pickle.load(handle)
        super(AffiNETy_PL_P_L_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [
            # f"{self.dataset}.pkl",
        ]

    @property
    def processed_file_names(self):
        if self.ensure_processed:
            return [f"{self.dataset}_{i}.pt" for i in self.pdb_ids]
        else:
            vals = [os.path.basename(i) for i in glob(f"{self.processed_dir}/*")]
            print(f"Only using subset: {len(vals)} / {len(self.pdb_ids)}")
            return vals

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process_single_pdb(self, datapoint):
        if not self.ensure_processed:
            return
        n, i, lig_confs = datapoint
        # try:
        pls = {
            # "x": [],
            # "edge_index": [],
            # "edge_attr": [],
            "z": [],
            "pos": [],
        }
        ps = {
            "z": [],
            "pos": [],
        }
        ls = {
            "z": [],
            "pos": [],
        }
        print(f"pdb_id : {i} : {n}")
        pre_processed_path = f"{self.processed_dir}/../pre_processed/{self.dataset}_{i}.pt"
        if os.path.exists(pre_processed_path):
            print("    already processed")
            return
        failed = False
        for j in self.num_confs_protein:
            # PL
            if (
                not os.path.exists(f"{self.pl_dir}/{i}/prot_{j}.pdb")
                or not os.path.exists(f"{self.pl_dir}/{i}/lig_{j}.sdf")
            ):
                print(f"Failed: cannot find files...", f"  {self.pl_dir}/{i}/prot_{j}.pdb",f"  {self.pl_dir}/{i}/lig_{j}.sdf", sep='\n')
                return
            try:
                pl_pro = read(f"{self.pl_dir}/{i}/prot_{j}.pdb")
                pl_lig = read(f"{self.pl_dir}/{i}/lig_{j}.sdf")
                pl = pl_pro + pl_lig
                z = torch.tensor(pl.get_atomic_numbers(), dtype=torch.int64) # Required for ViSNet
                pos = torch.tensor(pl.get_positions(), dtype=torch.float, requires_grad=True) # Required for ViSNet
                # x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(pl)
                pls["z"].append(z)
                pls["pos"].append(pos)
            except Exception as e:
                print(e)
                print("Failed on PL conversion(s)")
                failed = True
                break
            # P
            if (
                not os.path.exists(f"{self.pl_dir}/{i}/prot_{j}.pdb")
                or not os.path.exists(f"{self.pl_dir}/{i}/lig_{j}.sdf")
            ):
                print(f"Failed: cannot find files...", f"  {self.pl_dir}/{i}/prot_{j}.pdb",f"  {self.pl_dir}/{i}/lig_{j}.sdf", sep='\n')
                return
            try:
                p = read(f"{self.p_dir}/{i}/prot_{j}.pdb")
                z = torch.tensor(p.get_atomic_numbers(), dtype=torch.int64) # Required for ViSNet
                pos = torch.tensor(p.get_positions(), dtype=torch.float, requires_grad=True) # Required for ViSNet
                ps["z"].append(z)
                ps["pos"].append(pos)
            except Exception as e:
                print(e)
                print("Failed on PL conversion(s)")
                failed = True
                break
        # L
        print(f"  {len(lig_confs) = }")
        for j in lig_confs:
            try:
                l = rdkit_mol_to_ase_atoms(j)
                z = torch.tensor(l.get_atomic_numbers(), dtype=torch.int64) # Required for ViSNet
                pos = torch.tensor(l.get_positions(), dtype=torch.float, requires_grad=True) # Required for ViSNet
                ls["z"].append(z)
                ls["pos"].append(pos)
            except Exception as e:
                print(e)
                print("Failed on L conversion(s)")
                failed = True
                break
        if len(pls["z"]) == 0 or len(ls["z"]) == 0 or failed:
            print("Failed to update:", len(pls['z']), len(ps['z']), len(ls['z']), failed)
            return
        # pls = {k: torch.tensor(v) for k, v in pls.items()}
        # ls = {k: torch.tensor(v) for k, v in ls.items()}
        _d = Data(
            # pl
            # pl_x=pls["x"],
            # pl_edge_index=pls["edge_index"],
            # pl_edge_attr=pls["edge_attr"],
            pl_z=pls["z"],
            pl_pos=pls["pos"],
            p_z=ps["z"],
            p_pos=ps["pos"],
            # l
            # l_x=ls["x"],
            # l_edge_index=ls["edge_index"],
            # l_edge_attr=ls["edge_attr"],
            l_z=ls["z"],
            l_pos=ls["pos"],
            y=self.power_ranking_dict[i],
            batch=torch.tensor([0 for _ in range(len(pls['z'][0]))], dtype=torch.int64),
        )
        print(_d)
        print(f"  {pre_processed_path = }")
        torch.save(_d, pre_processed_path)
        return

    def generate_data(self):
        for n, pdb_id in enumerate(self.pdb_ids):
            conformers = self.df_lig[self.df_lig["pdb_id"] == pdb_id]["conformers"].to_list()[0]
            yield (n, pdb_id, conformers)

    def chunks(self, data, chunk_size):
        """Yield successive chunk_size chunks from data."""
        chunk = []
        for item in data:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk  # yield the last chunk even if it's smaller

    def process(self):
        print(f"Creating {self.dataset}...")
        if not os.path.exists(f"{self.processed_dir}/../pre_processed"):
            os.mkdir(f"{self.processed_dir}/../pre_processed")
        if self.NUM_THREADS > 1:
            with Pool(processes=self.NUM_THREADS) as pool:
                for data_chunk in self.chunks(self.generate_data(), self.chunk_size):
                    pool.imap(self.process_single_pdb, data_chunk)
                pool.close()
                pool.join()
        else:
            for d in self.generate_data():
                self.process_single_pdb(d)

        # Need to convert all available files to f"{self.dataset}_{idx}.pt" format
        pre_processed_files = glob(f"{self.processed_dir}/../pre_processed/*")
        # print(pre_processed_files)
        print("Copying files...")
        for n, i in enumerate(pre_processed_files):
            cmd = f"mv {i} {self.processed_dir}/{self.dataset}_{n}.pt"
            # print(cmd)
            os.system(cmd)
        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"{self.dataset}_{idx}.pt"))
        return data


class AffiNETy_dataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        dataset="casf2016",
        NUM_THREADS=1,
        pl_dir="/storage/ice1/7/3/awallace43/casf2016/pl",
        p_dir="/storage/ice1/7/3/awallace43/casf2016/p",
        l_pkl="/storage/ice1/7/3/awallace43/casf2016/l/casf_final.pkl",
        power_ranking_file = "/storage/ice1/7/3/awallace43/CASF-2016/power_ranking/CoreSet.dat",

        chunk_size=None,
        num_confs=None,
        ensure_processed=True,
    ):

        self.dataset = dataset
        self.df_lig = pd.read_pickle(l_pkl)
        self.pdb_ids = self.df_lig["pdb_id"].to_list()
        self.num_systems = len(self.pdb_ids)
        if num_confs:
            self.num_confs = range(num_confs)
        else:
            self.num_confs = range(self.df_lig.iloc[0]["num_conformers"])
        self.NUM_THREADS = NUM_THREADS
        print(f"Setting {NUM_THREADS = }")
        self.pl_dir=pl_dir
        self.p_dir=p_dir
        self.power_ranking_file = power_ranking_file
        self.ensure_processed=ensure_processed
        if chunk_size:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = NUM_THREADS * 10

        with open(power_ranking_file, 'rb') as handle:
            self.power_ranking_dict = pickle.load(handle)
        super(AffiNETy_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [
            f"{self.dataset}.pkl",
        ]

    @property
    def processed_file_names(self):
        if self.ensure_processed:
            return [f"{self.dataset}_{i}.pt" for i in self.pdb_ids]
        else:
            vals = [os.path.basename(i) for i in glob(f"{self.processed_dir}/*")]
            print(vals)
            return vals

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process_single_pdb(self, datapoint):
        n, i, lig_confs = datapoint
        # try:
        pls = {
            "x": [],
            "edge_index": [],
            "edge_attr": [],
            "z": [],
            "pos": [],
        }
        ps = pls.copy()
        ls = pls.copy()
        print(f"pdb_id: {i} : {n}")
        if os.path.exists(osp.join(self.processed_dir, f"{self.dataset}_{i}.pt")):
            print("    already processed")
            return
        failed = False
        for j in self.num_confs:
            # PL
            if (
                not os.path.exists(f"{self.pl_dir}/{i}/prot_{j}.pdb")
                or not os.path.exists(f"{self.pl_dir}/{i}/lig_{j}.sdf")
                or not os.path.exists(f"{self.p_dir}/{i}/prot_{j}.pdb")
                or not os.path.exists(f"{self.p_dir}/{i}/prot_{j}.pdb")
            ):
                return
            try:
                pl_pro = read(f"{self.pl_dir}/{i}/prot_{j}.pdb")
                pl_lig = read(f"{self.pl_dir}/{i}/lig_{j}.sdf")
                pl = pl_pro + pl_lig
                x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(pl)
                pls["x"].append(x)
                pls["edge_index"].append(edge_index)
                pls["edge_attr"].append(edge_attr)
                pls["z"].append(z)
                pls["pos"].append(pos)
                # P
                p = read(f"{self.p_dir}/{i}/prot_{j}.pdb")
                x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(p)
                ps["x"].append(x)
                ps["edge_index"].append(edge_index)
                ps["edge_attr"].append(edge_attr)
                ps["z"].append(z)
                ps["pos"].append(pos)
                # L
                l = rdkit_mol_to_ase_atoms(lig_confs[j])
                x, edge_index, edge_attr, z, pos = ase_to_ViSNet_data(l)
                ls["x"].append(x)
                ls["edge_index"].append(edge_index)
                ls["edge_attr"].append(edge_attr)
                ls["z"].append(z)
                ls["pos"].append(pos)
            except Exception as e:
                print(e)
                print("Failed on conversion(s)")
                failed = True
                break
        if len(pls["x"]) == 0 or len(ps["x"]) == 0 or len(ls["x"]) == 0 or failed:
            print("Failed to update:", len(pls['x']), len(ps['x']), len(ls['x']), failed)
            return
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
        print(_d)
        # if pre_filter is not None and not pre_filter(data):
        #     return

        # if pre_transform is not None:
        #     _d = pre_transform(_d)
        d_path = osp.join(self.processed_dir, f"{self.dataset}_{i}.pt")
        print(f"  {d_path = }")
        torch.save(_d, d_path)
        # except Exception:
        #     print(f"{i} failed...")
        return

    def generate_data(self):
        for n, pdb_id in enumerate(self.pdb_ids):
            conformers = self.df_lig[self.df_lig["pdb_id"] == pdb_id]["conformers"].to_list()[0]
            yield (n, pdb_id, conformers)

    def chunks(self, data, chunk_size):
        """Yield successive chunk_size chunks from data."""
        chunk = []
        for item in data:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk  # yield the last chunk even if it's smaller

    def process(self):
        print(f"Creating {self.dataset}...")
        # data = [
        #     (
        #         n,
        #         i,
        #         self.df_lig[self.df_lig["pdb_id"] == i]["conformers"].to_list()[0],
        #     )
        #     for n, i in enumerate(self.pdb_ids)
        # ]
        if self.NUM_THREADS > 1:
            with Pool(processes=self.NUM_THREADS) as pool:
                for data_chunk in self.chunks(self.generate_data(), self.chunk_size):
                    pool.imap(self.process_single_pdb, data_chunk)
                pool.close()
                pool.join()
        else:
            for d in self.generate_data():
                self.process_single_pdb(d)
        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"{self.dataset}_{idx}.pt"))
        return data


def main():
    import argparse

    print("imports done!\n")

    NUM_THREADS = int(os.getenv("OMP_NUM_THREADS"))
    print(f"Setting {NUM_THREADS = }")

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
    AffiNETy_dataset(
        root=f"data_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
    )
    return


if __name__ == "__main__":
    main()
