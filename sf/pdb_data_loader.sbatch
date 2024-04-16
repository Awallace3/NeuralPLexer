#!/bin/bash
#SBATCH --job-name=AF-DL-PDBBIND
#SBATCH -N1 -n6 #number of nodes and cores required per job
#SBATCH --mem-per-cpu=8G #memory per core
#SBATCH --time=16:00:00
#SBATCH -pice-cpu
#SBATCH --output=outs/dl_affinety_pdbbind.out

cd $SLURM_SUBMIT_DIR
__conda_setup="$('/home/hice1/awallace43/scratch/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/hice1/awallace43/scratch/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/hice1/awallace43/scratch/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/hice1/awallace43/scratch/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate /home/hice1/awallace43/scratch/miniconda3/envs/npl_td3

export OMP_NUM_THREADS=6
python3 data_loader.py --pdbbind
