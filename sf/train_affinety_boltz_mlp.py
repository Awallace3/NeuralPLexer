from src.dataset import AffiNETy_dataset, AffiNETy_PL_P_L_dataset
from src import models
import os
import argparse

print("imports done!\n")

NUM_THREADS = os.getenv("OMP_NUM_THREADS")
if NUM_THREADS != "":
    NUM_THREADS = 14
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
    power_ranking_file_pkl = power_ranking_file.replace("csv", "pkl")
else:
    pl_dir = "/storage/ice1/7/3/awallace43/casf2016/pl"
    p_dir = "/storage/ice1/7/3/awallace43/casf2016/p"
    l_pkl = "/storage/ice1/7/3/awallace43/casf2016/l/casf_final.pkl"
    power_ranking_file = "/storage/ice1/7/3/awallace43/CASF-2016/power_ranking_coreset.csv"
    power_ranking_file_pkl = power_ranking_file.replace("csv", "pkl")
    v = "casf"


def main():
    num_confs_protein=8
    ds = AffiNETy_dataset(
        root=f"data_n_{num_confs_protein}_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=num_confs_protein,
        ensure_processed=False,
    )
    m = models.AffiNETy(
        dataset=ds,
        # model=models.AffiNETy_graphSage_boltzmann_avg,
        # model=models.AffiNETy_graphSage_boltzmann_avg_Q,
        model=models.AffiNETy_graphSage_boltzmann_mlp,
        pl_in=num_confs_protein,
        p_in=num_confs_protein,
        num_workers=NUM_THREADS,
        use_GPU=True,
        lr=1e-4,
    )
    m.train(
        batch_size=16,
        pre_trained_model='prev',
        # pre_trained_model="./models/AffiNETy_mean.pt"
    )
    return


if __name__ == "__main__":
    main()
