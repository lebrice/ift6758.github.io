#!/bin/bash

pip install --quiet orion

# TODO: change this to the maximum number of desired trials.
MAX_TRIALS=10
EXPERIMENT_NAME="better_likes_condensing"

latest_tag=`git --git-dir=ift6758.github.io/.git --work-tree=ift6758.github.io describe --tags`
echo "latest tag is '$latest_tag'"
echo "experiment name is '$EXPERIMENT_NAME'"

if [[ $latest_tag == $EXPERIMENT_NAME ]]; then
        echo "Starting experiment '$EXPERIMENT_NAME'."
else
        echo "WARNING: experiment name '$EXPERIMENT_NAME' is not the same as the latest tag '$latest_tag'!"
        echo "this will make it hard to reproduce the results, as the code used at train and test-time might be different!"
fi

mkdir -p logs

MAX_EPOCHS_PER_EXPERIMENT=500

orion -v hunt --max-trials $MAX_TRIALS -n $EXPERIMENT_NAME ./ift6758.github.io/project/train.py \
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
