#/usr/bin/bash

# neuralplexer-inference --task=batched_structure_sampling --input-receptor 1Y0L_pro.fasta --input-ligand lig.sdf --out-path npl_out --n-samples 2 --chunk-size 2 --num-steps=2 --cuda --sampler=langevin_simulated_annealing --model-checkpoint ../../neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt --separate-pdb
neuralplexer-inference --task=batched_structure_sampling --input-receptor 1Y0L_pro_1_hfixed_pqr.pdb --input-ligand lig.sdf --out-path npl_out --n-samples 2 --chunk-size 2 --num-steps=2 --cuda --sampler=langevin_simulated_annealing --model-checkpoint ../../neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt --separate-pdb

# ~/gits/NeuralPLexer/neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt

