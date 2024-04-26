print("imports...\n")
import os
import argparse
import pandas as pd
import math

from src.dataset import AffiNETy_dataset, AffiNETy_PL_P_L_dataset, AffiNETy_torchmd_dataset

print("imports done!\n")

NUM_THREADS = os.getenv("OMP_NUM_THREADS")
if NUM_THREADS != "":
    NUM_THREADS = 12
else:
    NUM_THREADS = int(NUM_THREADS)
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
    power_ranking_file = "/storage/ice1/7/3/awallace43/index/INDEX_general_PL.2020.csv"
    # power_ranking_file_pkl = power_ranking_file.replace("csv", "pkl")
    power_ranking_file_pkl = power_ranking_file.replace(".csv", "_full.pkl")
else:
    pl_dir = "/storage/ice1/7/3/awallace43/casf2016/pl"
    p_dir = "/storage/ice1/7/3/awallace43/casf2016/p"
    l_pkl = "/storage/ice1/7/3/awallace43/casf2016/l/casf_final.pkl"
    power_ranking_file = "/storage/ice1/7/3/awallace43/CASF-2016/power_ranking_coreset.csv"
    # power_ranking_file_pkl = power_ranking_file.replace("csv", "pkl")
    power_ranking_file_pkl = power_ranking_file.replace(".csv", "_full.pkl")
    v = "casf"


def unit_conversion(unit):
    unit_to_factor = {
        "M": 1,  # Molar (exact)
        "mM": 1e-3,  # Millimolar
        "uM": 1e-6,  # Micromolar
        "nM": 1e-9,  # Nanomolar
        "pM": 1e-12,  # Picomolar
        "fM": 1e-15,  # Femtomolar
    }
    conversion_factor = unit_to_factor.get(unit, None)
    return conversion_factor


def process_K_label(l: str):
    """
    We want to have K_d, assume all binding is inhibitory
    to get K_d = K_i to expand dataset
    """
    l = l.strip()
    conv = unit_conversion(l[-2:])
    if "Ki" == l[:2] or "Kd" == l[:2]:
        l = l.replace("Ki", "").replace("Kd", "").replace("<", "").replace(">", "").replace("=", "").replace("~", "")
        val = float(l[:-2])
        return -math.log10(val * conv)
    elif "Ka=" == l[:3]:
        l = l.replace("Ki", "").replace("Ki", "").replace("<", "").replace(">", "").replace("=", "").replace("~", "")
        val = float(l[:-2])
        return math.log10(val * conv)
    elif "IC50" in l:
        l = l.replace("IC50", "").replace("Ki", "").replace("<", "").replace(">", "").replace("=", "").replace("~", "")
        print(l)
        val = float(l[:-2])
        return -math.log10(val * conv)
    else:
        return None

def cleanup_input_labels():
    if os.path.exists(power_ranking_file_pkl):
        return
    import pickle
    df = pd.read_csv(power_ranking_file, delimiter=",")
    # df["pK"] = df["Ka"].apply(lambda x: process_K_label(x))
    # print(df.columns.values)
    # for n, i in df.iterrows():
    #     pK = process_K_label(i['Ka'])
    #     print(i[''])

    df["pK"] = df["Ka"].apply(lambda x: process_K_label(x))
    print(df["pK"].max())
    print(len(df))
    df.dropna(subset=['pK'], inplace=True)
    print(len(df))
    print(df)
    if 'code' in df.columns:
        col = 'code'
    else:
        col = 'pdb_id'
    mapping = pd.Series(df.pK.values, index=df[col]).to_dict()
    print(mapping)
    with open(power_ranking_file_pkl, 'wb') as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def main():
    cleanup_input_labels()
    # return
    AffiNETy_torchmd_dataset(
        root=f"data_n_8_full_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=8,
        ensure_processed=True,
        chunk_size=100000,
    )
    return
    AffiNETy_dataset(
        root=f"data_n_8_full_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=8,
        ensure_processed=True,
    )
    return
    AffiNETy_PL_P_L_dataset(
        root=f"data_PL_P_L_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=2,
        ensure_processed=True,
    )
    return


if __name__ == "__main__":
    main()
