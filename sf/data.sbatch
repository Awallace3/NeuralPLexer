#!/bin/bash
#SBATCH --job-name=charges
#SBATCH --account=hive-cs207
#SBATCH -N1 -n12 #number of nodes and cores required per job
#SBATCH --mem-per-cpu=8G #memory per core
#SBATCH --time=03-00:00:00
#SBATCH -phive-gpu

cd $SLURM_SUBMIT_DIR
conda activate npl
export OMP_NUM_THREADS=12
python3 data.py
