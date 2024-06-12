#!/bin/bash

# Define an array of configuration paths
CONFIGS=(
        "citeseerLapPE.yaml"
        "citeseerLinearNode.yaml"
        "citeseerNoEnc.yaml"
        "citeseerRWSE.yaml"
        "citeseerLearnable.yaml"
        "citeseerNode2VecLearnable.yaml"
        "citeseerNode2Vec.yaml"
        )

NEW_CONFIGS=("citeseerNode2Vec.yaml")
            #  "coraNode2VecLearnable.yaml")

# Define an array of seed values
SEEDS=(1 2 3)

#Loop through all configuration files
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
        "posenc_Node2Vec.norm True posenc_Node2Vec.is_directed False"
         "posenc_Node2Vec.raw_norm_type LayerNorm"
         "True posenc_Node2Vec.raw_norm_type LayerNorm posenc_Node2Vec.model linear"
        )

for param in "${PARAMS[@]}"
do
    # Loop through all seed values
    for seed in "${SEEDS[@]}"
    do
        # Run the Python command with the current configuration file and seed
        python main.py --cfg "configs/Exphormer/citeseerNode2Vec.yaml" seed "$seed" name_tag "$param"
    done
done
