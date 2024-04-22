import pandas as pd

PDBBIND = True
if PDBBIND:
    pdbbind_dir = "/storage/ice1/7/3/awallace43/PDBBind_processed/"
    pdbbind_output = "/storage/ice1/7/3/awallace43/pdb_gen/p"
else:
    pdbbind_dir = "/storage/ice1/7/3/awallace43/CASF-2016/coreset"
    pdbbind_output = "/storage/ice1/7/3/awallace43/casf2016/p"

def ligand_csv(pdbbind_dir=pdbbind_dir):
    if PDBBIND:
        v = "pdbbind"
    else:
        v = 'casf'
    csv_path = f"{v}_processed_p.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return csv_path, df
    pdb_dirs = glob.glob(pdbbind_dir + "/*")
    data_dict = {
        "sample_id": [],
        "pdb_id": [],
        "protein_path": [],
        "ligand_path": [],
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


def main():
    ligand_csv(pdbbind_dir)
    return


if __name__ == "__main__":
    main()
