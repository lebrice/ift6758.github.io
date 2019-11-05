#!/bin/bash

pip install --quiet orion

# TODO: change this to the maximum number of desired trials.
MAX_TRIALS=10
EXPERIMENT_NAME="likes_conv_condensing_2"

mkdir -p logs

MAX_EPOCHS_PER_EXPERIMENT=500

orion -v --debug hunt --max-trials $MAX_TRIALS -n $EXPERIMENT_NAME ./ift6758.github.io/project/train.py \
        --experiment_name $EXPERIMENT_NAME \
        --epochs $MAX_EPOCHS_PER_EXPERIMENT \
        --batch_size 128 \
        --num_layers~"randint(1, 10)" \
        --dense_units~"choices(32, 64, 128, 256)" \
        --learning_rate 0.005 \
        --num_like_pages~"choices(50000, 20000, 10000, 5000)" \
        --likes_condensing_factor~"choices(2,3,4,5,10)" \
        --likes_condensed_vector_max_size~"choices(128, 256, 512, 1024)" \
        >> "logs/$EXPERIMENT_NAME.txt"
