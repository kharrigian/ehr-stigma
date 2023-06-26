#!/bin/sh

############ 
## Config
############

## Variables NOTE: PUT THE PATH TO YOUR CLONE OF THE REPO HERE AND YOUR CONDA ENVIRONMENT NAME
SCRIPT_DIR='/Users/kharrigian/dev/research/johns-hopkins/ehr-stigma/'

## Dataset
DATASET="mimic-iv-discharge"; DATASET_NAME="mimic-iv/camera-ready"

## Configs
CONFIGS=(
    "./configs/model/train/baseline-majority.default.json"
    "./configs/model/train/baseline-statistical.default.json"
)
CONFIG_NAMES=(
    "majority"
    "statistical"
)

############ 
## Setup
############

echo "Going to project directory (full path)"
cd $SCRIPT_DIR

############ 
## Execute
############

for i in {0..1}; do

    # echo "Initializing Configuration Paths"
    cpath=${CONFIGS[$i]}
    cname=${CONFIG_NAMES[$i]}

    echo "~~~~ Starting Cross Validation ~~~~"
    time python -u scripts/model/train/baseline.py \
        --model_settings $cpath \
        --dataset $DATASET \
        --eval_random_state 42 \
        --eval_cv 5 \
        --eval_test \
        --phraser_passes 2 \
        --phraser_min_count 5 \
        --phraser_threshold 10 \
        --cache_errors \
        --output_dir "./data/results/model/$DATASET_NAME/final-cv/baseline-$cname/" \
        --model_cache_dir "./data/models/classifiers/$DATASET_NAME/final-cv/baseline-$cname/" \
        --rm_existing

    echo "~~~~ Starting No Cross Validation ~~~~"
    time python -u scripts/model/train/baseline.py \
        --model_settings $cpath \
        --dataset $DATASET \
        --eval_random_state 42 \
        --eval_test \
        --phraser_passes 2 \
        --phraser_min_count 5 \
        --phraser_threshold 10 \
        --cache_errors \
        --output_dir "./data/results/model/$DATASET_NAME/final-no_cv/baseline-$cname/" \
        --model_cache_dir "./data/models/classifiers/$DATASET_NAME/final-no_cv/baseline-$cname/" \
        --rm_existing 

done
