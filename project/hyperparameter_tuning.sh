#!/bin/bash

pip install --quiet orion

# TODO: change this to the maximum number of desired trials.
MAX_TRIALS=5
EXPERIMENT_NAME="batch_size_experiment"

mkdir logs
rm logs/$EXPERIMENT_NAME.txt

EPOCHS_PER_EXPERIMENT=5

export ORION_RESULTS_PATH="logs/$EXPERIMENT_NAME-results.txt"

orion -v --debug hunt --max-trials $MAX_TRIALS -n $EXPERIMENT_NAME ./train.py \
        --epochs $EPOCHS_PER_EXPERIMENT \
        --batch_size~"choices(32,64,128,256)" \
        --num_layers~"randint(1,10)" \
        --activation~"choices('relu','tanh','linear')" \
        > "logs/$EXPERIMENT_NAME.txt"
