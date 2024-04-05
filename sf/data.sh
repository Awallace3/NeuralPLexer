#!/bin/bash
#SBATCH --job-name=charges
#SBATCH --account=hive-cs207
#SBATCH -N1 -n12 #number of nodes and cores required per job
#SBATCH --mem-per-cpu=8G #memory per core
#SBATCH --time=03-00:00:00
#SBATCH -phive-gpu

# cd $SLURM_SUBMIT_DIR

export pdbbind_dir=/storage/home/hhive1/awallace43/data/share/PDBBind_processed
export pdbbind_output=/storage/home/hhive1/awallace43/data/share/PDBBind_nlp

for i in $(ls -d $pdbbind_dir/*); do
    echo "Processing ${i}"
    pdbid=`basename "${i}"`
    echo "pdbid:" $pdbid
    if [ -d "${pdbbind_output}/${pdbid}" ]; then
        echo "Already processed ${pdbid}"
        continue
    fi
    mkdir -p ${pdbbind_output}/${pdbid}
    # pdb2fasta ${i}/${pdbid}_protein_processed.pdb > ${pdbbind_output}/${pdbid}/${pdbid}_protein.fasta
    input_protein=${pdbbind_output}/${pdbid}/${pdbid}_protein.pdb
    input_ligand=${pdbbind_output}/${pdbid}/${pdbid}_ligand.mol2
    cp ${i}/${pdbid}_protein_processed.pdb ${input_protein}
    cp ${i}/${pdbid}_ligand.mol2
    neuralplexer-inference --task=batched_structure_sampling --input-receptor ${pdbbind_output}/${pdbid}/${pdbid}.fasta --input-ligand ${pdbbind_output}/${pdbid}/${pdbid}_ligand.mol2 --out-path npl_out --n-samples 10 --chunk-size 4 --num-steps=20 --cuda --sampler=langevin_simulated_annealing --model-checkpoint ../neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt --out-path ${pdbbind_output}/${pdbid}/npl_out --separate-pdb
done
