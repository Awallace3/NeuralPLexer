import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, MolToSmiles
import glob
import os
import argparse

parser = argparse.ArgumentParser(description="Process the PDBBIND setting")
parser.add_argument('--no-pdbbind', dest='PDBBIND', action='store_false', 
                    help='Disable PDBBIND (default: enabled)')
args = parser.parse_args()
print(f"PDBBIND setting: {args.PDBBIND}")

PDBBIND = args.PDBBIND
if PDBBIND:
    pdbbind_dir = "/storage/ice1/7/3/awallace43/pdb_gen/pl"
    pdbbind_output = "/storage/ice1/7/3/awallace43/pdb_gen/l"
    v = "pdbbind"
    out_pkl = "pdbbind_out.pkl"
else:
    pdbbind_dir = "/storage/ice1/7/3/awallace43/CASF-2016/coreset"
    pdbbind_output = "/storage/ice1/7/3/awallace43/casf2016/l"
    v = 'casf'
    out_pkl = "casf_out.pkl"
csv_path = f"{v}_lig.csv"

print(f"{pdbbind_dir = }")

def sdf_to_smiles(sdf_file_path, smiles_file_path=None):
    supplier = SDMolSupplier(sdf_file_path)
    smiles_list = []
    # Iterate through the molecules in the SDF
    for mol in supplier:
        if mol is not None:
            # Get the canonical SMILES string of the current molecule
            smiles = MolToSmiles(mol, canonical=True)
            smiles_list.append(smiles)
        else:
            print("A molecule in the SDF could not be parsed.")
    return smiles_list

def create_csv(pdbbind_dir=pdbbind_dir):
    pdb_dirs = glob.glob(pdbbind_dir + "/*")
    data_dict = {
        "pdb_id": [],
        "num_conformers": [],
        "smile_str": [],
    }
    for n, i in enumerate(pdb_dirs):
        pdb_id = i.split("/")[-1]
        if PDBBIND:
            ligand_path = i + "/" +  "lig_0.sdf"
        else:
            ligand_path = i + "/" + pdb_id + "_ligand.sdf"
        if not os.path.exists(ligand_path):
            print(f"{ligand_path} does not exist!")
            continue
        smiles = sdf_to_smiles(ligand_path)
        if len(smiles) == 0:
            continue
        data_dict['smile_str'].append(smiles[0])
        data_dict["pdb_id"].append(pdb_id)
        data_dict['num_conformers'].append(10)
    df = pd.DataFrame(data_dict)
    print(df)
    df.to_csv(csv_path, index=False)
    return csv_path, df

def run_generate():
    os.system(f"python generate_confs.py --test_csv {csv_path} --inference_steps 20 --model_dir workdir/drugs_default --out {out_pkl} --tqdm --batch_size 128 --no_energy")
    return
    

def read_output():
    data = pd.read_pickle(out_pkl)
    df = pd.read_csv(csv_path)
    df['conformers'] = df.apply(lambda r: data.get(r['smile_str'], None), axis=1)
    print(df)
    df.to_pickle(f"{pdbbind_output}/{v}_final.pkl")
    return

def main():
    print('Creating csv...')
    create_csv()
    print("Generation...")
    run_generate()
    print('read_output...')
    read_output()
    return


if __name__ == "__main__":
    main()
