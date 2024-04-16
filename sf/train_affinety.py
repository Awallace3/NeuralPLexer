from src.dataset import AffiNETy_dataset, AffiNETy_PL_P_L_dataset
from src.models import AffiNETy, AffiNETy_PL_P_L, AffiNETy_graphSage
import os
import argparse

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


def main():
    # ds = AffiNETy_PL_P_L_dataset(
    #     root=f"data_PL_P_L_{v}",
    #     dataset=v,
    #     NUM_THREADS=NUM_THREADS,
    #     pl_dir=pl_dir,
    #     l_pkl=l_pkl,
    #     power_ranking_file=power_ranking_file_pkl,
    #     num_confs_protein=2,
    #     ensure_processed=False,
    # )
    ds = AffiNETy_dataset(
        root=f"data_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=2,
        ensure_processed=False,
    )
    m = AffiNETy(
        dataset=ds,
        model=AffiNETy_graphSage,
        num_workers=NUM_THREADS,
        use_GPU=True,
    )
    m.train()
    return


if __name__ == "__main__":
    main()
