from src.dataset import AffiNETy_dataset, AffiNETy_PL_P_L_dataset, AffiNETy_torchmd_dataset
from src import models
import os
import argparse
from torch_geometric.nn.models import ViSNet

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

PDBBIND = not args.PDBBIND
# PDBBIND = args.PDBBIND
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

def train_graphSage_AtomicEnergy_models(num_confs_protein):
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
        # model=models.AffiNETy_graphSage_boltzmann_mlp,
        # model=models.AffiNETy_graphSage_boltzmann_mlp,
        model=models.AffiNETy_GCN_AtomicEnergy_boltzmann_mlp,
        # model=models.AffiNETy_graphSage_AtomicEnergy_boltzmann_mlp,
        pl_model=None,
        pl_in=num_confs_protein,
        p_in=num_confs_protein,
        num_workers=NUM_THREADS,
        use_GPU=True,
        lr=1e-4,
    )
    m.train(
        batch_size=4,
        pre_trained_model='prev',
        verbose=1,
        # pre_trained_model="./models/AffiNETy_mean.pt"
    )

def train_graphSage_models(num_confs_protein):
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
        # model=models.AffiNETy_graphSage_boltzmann_mlp,
        # model=models.AffiNETy_graphSage_boltzmann_mlp,
        model=models.AffiNETy_graphSage_GCN_boltzmann_mlp,
        pl_in=num_confs_protein,
        p_in=num_confs_protein,
        num_workers=NUM_THREADS,
        use_GPU=True,
        lr=1e-4,
    )
    m.train(
        batch_size=4,
        pre_trained_model='prev',
        # pre_trained_model="./models/AffiNETy_mean.pt"
    )


def train_torchMD_models(num_confs_protein):
    from torchmdnet.models.model import load_model
    ds = AffiNETy_torchmd_dataset(
        root=f"data_n_{num_confs_protein}_full_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=num_confs_protein,
        ensure_processed=False,
    )
    model_path = "/storage/ice1/7/3/awallace43/torchmd_data/epoch=2139-val_loss=0.2543-test_loss=0.2317.ckpt"
    # model_path = "/home/hice1/awallace43/scratch/torchmd_data/ani/ANI1-equivariant_transformer/epoch=359-val_loss=0.0004-test_loss=0.0120.ckpt"
    # model_path = "/storage/ice1/7/3/awallace43/torchmd_data/epoch=649-val_loss=0.0003-test_loss=0.0059.ckpt"
    m = models.AffiNETy(
        dataset=ds,
        model=models.AffiNETy_equivariant_torchmdnet_boltzmann_mlp,
        pl_model=load_model(model_path, derivative=False),
        p_model= load_model(model_path, derivative=False),
        l_model= load_model(model_path, derivative=False),
        pl_in=num_confs_protein,
        p_in=num_confs_protein,
        num_workers=NUM_THREADS,
        use_GPU=True,
        lr=1e-4,
    )
    m.train(
        batch_size=2,
        pre_trained_model='prev',
        # verbose=True,
        # pre_trained_model="./models/AffiNETy_mean.pt"
    )

def train_ViSNet_models(num_confs_protein):
    # from torchmdnet.models.model import load_model
    ds = AffiNETy_torchmd_dataset(
        root=f"data_n_{num_confs_protein}_full_{v}",
        dataset=v,
        NUM_THREADS=NUM_THREADS,
        pl_dir=pl_dir,
        p_dir=p_dir,
        l_pkl=l_pkl,
        power_ranking_file=power_ranking_file_pkl,
        num_confs_protein=num_confs_protein,
        ensure_processed=False,
    )
    # model_path = "/storage/ice1/7/3/awallace43/torchmd_data/epoch=2139-val_loss=0.2543-test_loss=0.2317.ckpt"
    # model_path = "/home/hice1/awallace43/scratch/torchmd_data/ani/ANI1-equivariant_transformer/epoch=359-val_loss=0.0004-test_loss=0.0120.ckpt"
    # model_path = "/storage/ice1/7/3/awallace43/torchmd_data/epoch=649-val_loss=0.0003-test_loss=0.0059.ckpt"
    m = models.AffiNETy(
        dataset=ds,
        # model=models.AffiNETy_ViSNet_boltzmann_mlp,
        model=models.AffiNETy_ViSNet_boltzmann_mlp2,
        # model=models.AffiNETy_ViSNet_boltzmann_avg_Q,
        pl_model=ViSNet(
                        lmax=1,
                        cutoff=3.0,
                        hidden_channels=12,
                        num_heads=4,
                        num_layers=6,
                        trainable_vecnorm=False,
                        num_rbf=16,
                        trainable_rbf=False,
                        # max_z=50,
                        # max_num_neighbors=42,
                        vertex=False,
                        ),
        # p_model=ViSNet( cutoff=3.0, hidden_channels=48, num_heads=4, num_layers=6, ),
        # l_model=ViSNet( cutoff=3.0, hidden_channels=48, num_heads=4, num_layers=6, ),
        # pl_model=load_model(model_path, derivative=False),
        # p_model= load_model(model_path, derivative=False),
        # l_model= load_model(model_path, derivative=False),
        pl_in=num_confs_protein,
        p_in=num_confs_protein,
        num_workers=NUM_THREADS,
        use_GPU=True,
        lr=1e-4,
    )
    m.train(
        batch_size=2,
        pre_trained_model='prev',
        # verbose=True,
        # pre_trained_model="./models/AffiNETy_mean.pt"
    )

def main():
    num_confs_protein=8
    # train_torchMD_models(num_confs_protein)
    train_ViSNet_models(num_confs_protein)
    # train_graphSage_models(num_confs_protein)
    # train_graphSage_GCN_models(num_confs_protein)
    # train_graphSage_AtomicEnergy_models(num_confs_protein)
    return


if __name__ == "__main__":
    main()
