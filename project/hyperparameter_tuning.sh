#!/bin/bash

pip install --quiet orion

# TODO: change this to the maximum number of desired trials.
MAX_TRIALS=50
EXPERIMENT_NAME="SGD_with_regularization"

mkdir logs

MAX_EPOCHS_PER_EXPERIMENT=50

orion -v --debug hunt --max-trials $MAX_TRIALS -n $EXPERIMENT_NAME ./ift6758.github.io/project/train.py \
        --experiment_name $EXPERIMENT_NAME \
        --epochs $MAX_EPOCHS_PER_EXPERIMENT \
        --batch_size 256 \
        --num_layers~"randint(1,5)" \
        --dense_units~"choices(64, 128, 256)" \
        --activation tanh \
        --learning_rate~"choices(0.05, 0.01, 0.1, 0.005)" \
        --num_like_pages 5000 \
        --use_dropout~"choices(True, False)" \
        --use_batchnorm~"choices(True, False)" \
        --l1_reg~"choices(0, 0.05, 0.01, 0.005, 0.001)" \
        --l2_reg~"choices(0, 0.05, 0.01, 0.005, 0.001)" \
        >> "logs/$EXPERIMENT_NAME.txt"
