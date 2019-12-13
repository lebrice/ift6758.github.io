#!/bin/bash
source /home/mila/teaching/user07/miniconda3/bin/activate
conda activate /home/mila/teaching/user07/miniconda3/envs/datascience

rm -rf ift6758.github.io
BRANCH_OR_TAG="best_model_02"

git clone https://github.com/lebrice/ift6758.github.io.git --recursive --branch $BRANCH_OR_TAG
pip install -e ift6758.github.io/project/SimpleParsing
pip install orion

#ift6758.github.io/project/train_best_model.sh
ift6758.github.io/project/hyperparameter_tuning.sh
