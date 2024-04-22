#!/bin/bash
#SBATCH --job-name=P-CASF
#SBATCH -N1 -n6 #number of nodes and cores required per job
#SBATCH --mem-per-cpu=8G #memory per core
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH -pice-gpu
#SBATCH --output=outs/pro_casf.out

cd $SLURM_SUBMIT_DIR
conda activate npl_td

export OMP_NUM_THREADS=6
python3 protein_casf.py
