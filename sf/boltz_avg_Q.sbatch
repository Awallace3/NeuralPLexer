#!/bin/bash
#SBATCH --job-name=AN-b-aQ
#SBATCH -N1 -n12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
#SBATCH -pice-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --output=outs/AN-boltz-avg-Q.out

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
conda activate /home/hice1/awallace43/scratch/miniconda3/envs/npl_td4

export OMP_NUM_THREADS=12
echo "Starting script..."
python3 -u ./train_affinety_boltz_avg_Q.py --pdbbind
