#!/bin/bash

# Define an array of configuration paths
CONFIGS=(
        # "coraLapPE.yaml"
        # "coraSignNet.yaml"
        # "coraLinearNode.yaml"
        # "coraNoEnc.yaml"
        # "coraRWSE.yaml"
        # "coraLearnable.yaml"
        # "coraNode2VecLearnable.yaml"
        "coraNode2Vec.yaml"
        )

# NEW_CONFIGS=("coraNode2Vec.yaml"
#             #  "coraNode2VecLearnable.yaml")

# Define an array of seed values
SEEDS=(1 2 3)

# Loop through all configuration files
for cfg in "${CONFIGS[@]}"
do
    # Loop through all seed values
    for seed in "${SEEDS[@]}"
    do
        # Run the Python command with the current configuration file and seed
        python main.py --cfg "configs/Exphormer/$cfg" seed "$seed"
    done
done

PARAMS=(
        "posenc_Node2Vec.norm True"
        "posenc_Node2Vec.model linear"
        "posenc_Node2Vec.norm True posenc_Node2Vec.model linear"
        "posenc_Node2Vec.raw_norm_type LayerNorm"
        "True posenc_Node2Vec.raw_norm_type LayerNorm posenc_Node2Vec.model linear"
        )

for param in "${PARAMS[@]}"
do
    # Loop through all seed values
    for seed in "${SEEDS[@]}"
    do
        # Run the Python command with the current configuration file and seed
        python main.py --cfg "configs/Exphormer/coraNode2Vec.yaml" seed "$seed" name_tag "$param"
    done
done
