#!/bin/bash

pip install --quiet orion

# TODO: change this to the maximum number of desired trials.
MAX_TRIALS=50
EXPERIMENT_NAME="batch_size_experiment"

mkdir logs

EPOCHS_PER_EXPERIMENT=5

orion -v --debug hunt --max-trials $MAX_TRIALS -n $EXPERIMENT_NAME ./ift6758.github.io/project/train.py \
        --experiment_name $EXPERIMENT_NAME \
        --epochs $EPOCHS_PER_EXPERIMENT \
        --batch_size~"choices(32,64,128,256)" \
        --num_layers~"randint(1,10)" \
        --activation~"choices('relu','tanh','linear')" \
        --learning_rate~"choices(0.05, 0.01, 0.1, 0.005)" \
        --num_like_pages~"choices(1000, 5000, 10_000)" \
        --use_dropout~"choices(True, False)" \
        --use_batchnorm~"choices(True, False)" \
        >> "logs/$EXPERIMENT_NAME.txt"
