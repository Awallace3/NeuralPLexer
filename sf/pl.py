from dataclasses import dataclass
import os
from pprint import pprint as pp
import subprocess
import pandas as pd
import argparse
import glob
import math
import multiprocessing as mp
import os
import re
import subprocess
import warnings
from subprocess import check_output
import numpy as np
import pandas as pd
import torch
import tqdm
print('RDKit')
from rdkit import Chem
from rdkit.Chem import AllChem

print("AF")
from af_common.residue_constants import restype_1to3
print("NPL")
from neuralplexer.data.indexers import collate_numpy
from neuralplexer.data.physical import calc_heavy_atom_LJ_clash_fraction
from neuralplexer.data.pipeline import (featurize_protein_and_ligands,
                                        inplace_to_cuda, inplace_to_torch,
                                        process_mol_file, write_conformer_sdf,
                                        write_pdb_models, write_pdb_single)
from neuralplexer.model.config import (_attach_binding_task_config,
                                       get_base_config)
from neuralplexer.model.wrappers import NeuralPlexer
from neuralplexer.util.pdb3d import (compute_ligand_rmsd, compute_tm_rmsd,
                                     get_lddt_bs)
torch.set_grad_enabled(False)
print("Start")

PDBBIND = True
if PDBBIND:
    pdbbind_dir = "/storage/ice1/7/3/awallace43/PDBBind_processed/"
    pdbbind_output = "/storage/ice1/7/3/awallace43/pdb_gen/pl"
else:
    pdbbind_dir = "/storage/ice1/7/3/awallace43/CASF-2016/coreset"
    pdbbind_output = "/storage/ice1/7/3/awallace43/casf2016/pl"

@dataclass
class Args:
    task: str = "batched_structure_sampling"
    sample_id: int = 0
    template_id: int = 0
    cuda: bool = False  # actions like 'store_true' are represented by a boolean
    model_checkpoint: str = None
    input_ligand: str = None
    input_receptor: str = None
    input_template: str = None
    out_path: str = None
    n_samples: int = 64
    chunk_size: int = 8
    num_steps: int = 100
    latent_model: str = None
    sampler: str = "langevin_simulated_annealing"
    start_time: str = "1.0"
    max_chain_encoding_k: int = -1
    exact_prior: bool = False
    discard_ligand: bool = False
    discard_sdf_coords: bool = False
    detect_covalent: bool = False
    use_template: bool = False
    separate_pdb: bool = False
    rank_outputs_by_confidence: bool = False
    csv_path: str = None





# 1a30_ligand.mol2  1a30_ligand_opt.mol2  1a30_ligand.sdf  1a30_pocket.pdb  1a30_protein.mol2  1a30_protein.pdb

def pdbbind_csv_creation(pdbbind_dir=pdbbind_dir):
    if PDBBIND:
        v = "pdbbind"
    else:
        v = 'casf'
    csv_path = f"{v}_processed_pl.csv"
    pdb_dirs = glob.glob(pdbbind_dir + "/*")
    data_dict = {
        "sample_id": [],
        "pdb_id": [],
        # "chain_id": [],
        "protein_path": [],
        "ligand_path": [],
        # "reference_path": [],
    }
    for i in pdb_dirs:
        pdb_id = i.split("/")[-1]
        if PDBBIND:
            protein_path = i + "/" + pdb_id + "_protein_processed.pdb"
            ligand_path = i + "/" + pdb_id + "_ligand.sdf"
        else:
            protein_path = i + "/" + pdb_id + "_protein.pdb"
            ligand_path = i + "/" + pdb_id + "_ligand.sdf"
        # reference_path = i + '/' + pdb_id + '_reference.sdf'
        # protein_path = pdb_id + "_protein_processed.pdb"
        # ligand_path = pdb_id + "_ligand.sdf"
        data_dict["sample_id"].append(pdb_id)
        data_dict["pdb_id"].append(pdb_id)
        # data_dict["chain_id"].append(chain_id)
        data_dict["protein_path"].append(protein_path)
        data_dict["ligand_path"].append(ligand_path)
        # data_dict["reference_path"].append(reference_path)
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_path)
    return csv_path, df

def multi_pose_sampling(
    ligand_path,
    receptor_path,
    args,
    model,
    out_path,
    save_pdb=True,
    separate_pdb=True,
    chain_id=None,
    template_path=None,
    confidence=True,
    **kwargs,
):
    struct_res_all, lig_res_all = [], []
    plddt_all, plddt_lig_all = [], []
    chunk_size = args.chunk_size
    for _ in range(args.n_samples // chunk_size):
        # Resample anchor node frames
        np_sample, mol = featurize_protein_and_ligands(
            ligand_path,
            receptor_path,
            n_lig_patches=model.config.mol_encoder.n_patches,
            chain_id=chain_id,
            template_path=template_path,
            discard_sdf_coords=args.discard_sdf_coords,
            **kwargs,
        )
        np_sample_batched = collate_numpy([np_sample for _ in range(chunk_size)])
        sample = inplace_to_torch(np_sample_batched)
        if args.cuda:
            sample = inplace_to_cuda(sample)
        output_struct = model.sample_pl_complex_structures(
            sample,
            sampler=args.sampler,
            num_steps=args.num_steps,
            return_all_states=False,
            start_time=args.start_time,
            exact_prior=args.exact_prior,
        )
        if mol is not None:
            ref_mol = AllChem.Mol(mol)
            out_x1 = np.split(output_struct["ligands"].cpu().numpy(), args.chunk_size)
        out_x2 = np.split(
            output_struct["receptor_padded"].cpu().numpy(), args.chunk_size
        )
        if confidence:
            plddt, plddt_lig = model.run_confidence_estimation(
                sample, output_struct, return_avg_stats=True
            )

        for struct_idx in range(args.chunk_size):
            struct_res = {
                "features": {
                    "asym_id": np_sample["features"]["res_chain_id"],
                    "residue_index": np.arange(len(np_sample["features"]["res_type"]))
                    + 1,
                    "aatype": np_sample["features"]["res_type"],
                },
                "structure_module": {
                    "final_atom_positions": out_x2[struct_idx],
                    "final_atom_mask": sample["features"]["res_atom_mask"]
                    .bool()
                    .cpu()
                    .numpy(),
                },
            }
            struct_res_all.append(struct_res)
            if mol is not None:
                lig_res_all.append(out_x1[struct_idx])
            if confidence:
                plddt_all.append(plddt[struct_idx].item())
                if plddt_lig is None:
                    plddt_lig_all.append(None)
                else:
                    plddt_lig_all.append(plddt_lig[struct_idx].item())
    if confidence and args.rank_outputs_by_confidence:
        struct_plddts = np.array(plddt_lig_all if all(plddt_lig_all) else plddt_all)  # rank outputs using ligand plDDT if available
        struct_plddts = np.argsort(-struct_plddts).argsort()  # ensure that higher plDDTs have a higher rank (e.g., `rank1`)
    if save_pdb:
        if separate_pdb:
            for struct_id, struct_res in enumerate(struct_res_all):
                if confidence and args.rank_outputs_by_confidence:
                    write_pdb_single(
                        struct_res, out_path=os.path.join(out_path, f"prot_rank{struct_plddts[struct_id] + 1}.pdb")
                    )
                else:
                    write_pdb_single(
                        struct_res, out_path=os.path.join(out_path, f"prot_{struct_id}.pdb")
                    )
        write_pdb_models(
            struct_res_all, out_path=os.path.join(out_path, f"prot_all.pdb")
        )
    if mol is not None:
        write_conformer_sdf(
            ref_mol, None, out_path=os.path.join(out_path, f"lig_ref.sdf")
        )
        lig_res_all = np.array(lig_res_all)
        write_conformer_sdf(
            mol, lig_res_all, out_path=os.path.join(out_path, f"lig_all.sdf")
        )
        for struct_id in range(len(lig_res_all)):
            if confidence and args.rank_outputs_by_confidence:
                write_conformer_sdf(
                    mol,
                    lig_res_all[struct_id : struct_id + 1],
                    out_path=os.path.join(out_path, f"lig_rank{struct_plddts[struct_id] + 1}.sdf"),
                )
            else:
                write_conformer_sdf(
                    mol,
                    lig_res_all[struct_id : struct_id + 1],
                    out_path=os.path.join(out_path, f"lig_{struct_id}.sdf"),
                )
    else:
        ref_mol = None
    if confidence:
        return ref_mol, plddt_all, plddt_lig_all
    return ref_mol

def structure_prediction(args, df):
    torch.set_grad_enabled(False)
    config = get_base_config()

    if args.model_checkpoint is not None:
        # No need to specify this when loading the entire model
        model = NeuralPlexer.load_from_checkpoint(
            checkpoint_path=args.model_checkpoint, strict=False
        )
        config = model.config
        if args.latent_model is not None:
            config.latent_model = args.latent_model
        if args.task == "pdbbind_benchmarking":
            config.task.use_template = True
        elif args.task == "binding_site_recovery_benchmarking":
            config.task.use_template = True
        else:
            config.task.use_template = args.use_template
        config.task.detect_covalent = args.detect_covalent

    model = NeuralPlexer.load_from_checkpoint(
        config=config, checkpoint_path=args.model_checkpoint, strict=False
    )
    model.eval()
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        model.cuda()

    for n, r in df.iterrows():
        print(r['pdb_id'])
        print(r)
        args.input_ligand = r["ligand_path"]
        args.input_receptor = r["protein_path"]
        args.sample_id = n
        args.out_path = pdbbind_output + "/" + r['pdb_id']
        if os.path.exists(args.out_path):
            continue
        os.mkdir(args.out_path)
        try:
            if args.start_time != "auto":
                args.start_time = float(args.start_time)
            if args.task == "single_sample_trajectory":
                single_sample_sampling(args, model)
            elif args.task == "batched_structure_sampling":
                # Handle no ligand input
                if args.input_ligand is not None:
                    ligand_paths = list(args.input_ligand.split("|"))
                else:
                    ligand_paths = None
                if not args.input_receptor.endswith(".pdb"):
                    warnings.warn("Assuming the provided receptor input is a protein sequence")
                    create_full_pdb_with_zero_coordinates(
                        args.input_receptor, args.out_path + "/input.pdb"
                    )
                    args.input_receptor = args.out_path + "/input.pdb"
                multi_pose_sampling(
                    ligand_paths,
                    args.input_receptor,
                    args,
                    model,
                    args.out_path,
                    template_path=args.input_template,
                    separate_pdb=args.separate_pdb,
                )
        except Exception as e:
            print(f"Error: {e}")
            continue
"""

    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--sample-id", default=0, type=int)
    parser.add_argument("--template-id", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--model-checkpoint", type=str)
    parser.add_argument("--input-ligand", type=str)
    parser.add_argument("--input-receptor", type=str)
    parser.add_argument("--input-template", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--n-samples", default=64, type=int)
    parser.add_argument("--chunk-size", default=8, type=int)
    parser.add_argument("--num-steps", default=100, type=int)
    parser.add_argument("--latent-model", type=str)
    parser.add_argument("--sampler", required=True, type=str)
    parser.add_argument("--start-time", default="1.0", type=str)
    parser.add_argument("--max-chain-encoding-k", default=-1, type=int)
    parser.add_argument("--exact-prior", action="store_true")
    parser.add_argument("--discard-ligand", action="store_true")
    parser.add_argument("--discard-sdf-coords", action="store_true")
    parser.add_argument("--detect-covalent", action="store_true")
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--separate-pdb", action="store_true")
    parser.add_argument("--rank-outputs-by-confidence", action="store_true")
    parser.add_argument("--csv-path", type=str)
"""


def main():
    csv_path, df = pdbbind_csv_creation()
    args = Args()
    args.task = "batched_structure_sampling"
    # args.input_receptor = pdbbind_dir
    # args.input_ligand = pdbbind_dir
    args.out_path = pdbbind_output
    args.n_samples = 10
    args.chunk_size = 4
    args.num_steps = 20
    args.cuda = True
    args.sampler = "langevin_simulated_annealing"
    args.model_checkpoint = "../neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt"
    args.out_path = pdbbind_output
    args.separate_pdb = True
    # args.csv_path = csv_path
    pp(args)
    structure_prediction(args, df)
    return


if __name__ == "__main__":
    main()
