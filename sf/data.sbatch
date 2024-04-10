#!/bin/bash
#SBATCH --job-name=protein_gen
#SBATCH --account=hive-cs207
#SBATCH -N1 -n6 #number of nodes and cores required per job
#SBATCH --mem-per-cpu=8G #memory per core
#SBATCH --time=03-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH -phive-gpu

cd $SLURM_SUBMIT_DIR
conda init
conda activate npl

export OMP_NUM_THREADS=12
python3 protein.py
