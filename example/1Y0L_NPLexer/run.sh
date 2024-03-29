#/usr/bin/bash

neuralplexer-inference --task=batched_structure_sampling --input-receptor 1Y0L_pro.fasta --input-ligand lig.sdf --out-path npl_out --n-samples 16 --chunk-size 4 --num-steps=40 --cuda --sampler=langevin_simulated_annealing --model-checkpoint ../neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt

# ~/gits/NeuralPLexer/neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt

