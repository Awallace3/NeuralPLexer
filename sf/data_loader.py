print("imports...\n")
import os
import argparse
import pandas as pd
import math

from src.dataset import AffiNETy_dataset, AffiNETy_PL_L_dataset

print("imports done!\n")

NUM_THREADS = int(os.getenv("OMP_NUM_THREADS"))
# NUM_THREADS = 1
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
    power_ranking_file = (
        "/storage/ice1/7/3/awallace43/CASF-2016/power_ranking_coreset.csv"
    )
    power_ranking_file_pkl = power_ranking_file.replace("csv", "pkl")
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


def process_K_label(l):
    """
    We want to have K_d, assume all binding is inhibitory
    to get K_d = K_i to expand dataset
    """
    conv = unit_conversion(l[-2:])
    v = float(l[3:-2])
    if "Ki=" == l[:3] or "Kd=" == l[:3]:
        return -math.log(v * conv)
    elif "Ka=" == l[:3]:
        return math.log(v * conv)
    else:
        return None


def cleanup_input_labels():
    if os.path.exists(power_ranking_file_pkl):
        return
    import pickle
    df = pd.read_csv(power_ranking_file, delimiter=",")
    df["pK"] = df["Ka"].apply(lambda x: process_K_label(x))
    print(df)
    mapping = pd.Series(df.pK.values, index=df.code).to_dict()
    print(mapping)
    with open(power_ranking_file_pkl, 'wb') as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def main():
    cleanup_input_labels()
    # AffiNETy_dataset(
    #     root=f"data_{v}",
    #     dataset=v,
    #     NUM_THREADS=NUM_THREADS,
    #     pl_dir=pl_dir,
    #     p_dir=p_dir,
    #     l_pkl=l_pkl,
    #     power_ranking_file=power_ranking_file_pkl,
    #     num_confs=8,
    # )

    AffiNETy_PL_L_dataset(
        root=f"data_PL_L_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=2,
        ensure_processed=True,
    )
    return


if __name__ == "__main__":
    main()
