import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Only validate on CASF
PDBBIND = False
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
    trained_models = [
            # models.AffiNETy_graphSage_boltzmann_avg,
            models.AffiNETy_graphSage_boltzmann_avg_Q,
            models.AffiNETy_graphSage_boltzmann_mlp,
    ]
    results = {}
    for n, i in enumerate(trained_models):
        m = models.AffiNETy(
            dataset=ds,
            model=i,
            pl_in=num_confs_protein,
            p_in=num_confs_protein,
            num_workers=NUM_THREADS,
            use_GPU=True,
            lr=1e-4,
        )
        b_mlp_results = m.eval_casf(
            batch_size=16,
            pre_trained_model='prev',
        )
        print(b_mlp_results)
        if n == 0:
            results['CASF2016'] = b_mlp_results[1, :]
        results[m.model.model_output()] = b_mlp_results[0, :]
    df = pd.DataFrame(results)
    df.to_csv("./outs/eval_casf2.csv")
    return


if __name__ == "__main__":
    main()
