#!/bin/bash

pip install --quiet orion

# TODO: change this to the maximum number of desired trials.
MAX_TRIALS=20
EXPERIMENT_NAME="slim_models"

mkdir -f logs

MAX_EPOCHS_PER_EXPERIMENT=500

orion -v --debug hunt --max-trials $MAX_TRIALS -n $EXPERIMENT_NAME ./ift6758.github.io/project/train.py \
        --experiment_name $EXPERIMENT_NAME \
        --epochs $MAX_EPOCHS_PER_EXPERIMENT \
        --batch_size 64 \
        --num_layers~"randint(1,3)" \
        --dense_units~"choices(32, 64, 128)" \
        --activation~"choices(tanh, relu)" \
        --learning_rate~"choices(0.01, 0.005)" \
        --num_like_pages 5000 \
        --use_dropout True \
        --use_batchnorm False \
        --optimizer sgd \
        --l1_reg~"choices(0.01, 0.005)" \
        --l2_reg~"choices(0.01, 0.005)" \
        >> "logs/$EXPERIMENT_NAME.txt"
