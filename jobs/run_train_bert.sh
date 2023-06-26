#!/bin/sh

############ 
## Config
############

## Variables NOTE: PUT THE PATH TO YOUR CLONE OF THE REPO HERE AND YOUR CONDA ENVIRONMENT NAME
SCRIPT_DIR='/Users/kharrigian/dev/research/johns-hopkins/ehr-stigma/'

## Device (cuda or cpu)
DEVICE="cpu"

## Dataset/Model Combinations
DATASET="mimic-iv-discharge"; DATASET_NAME="mimic-iv/camera-ready"; MODEL="emilyalsentzer/Bio_ClinicalBERT"; MODEL_NAME="clinical_bert"
# DATASET="mimic-iv-discharge"; DATASET_NAME="mimic-iv/camera-ready"; MODEL="bert-base-uncased"; MODEL_NAME="base_bert"

## Which Categories
KEYWORD_CATEGORIES="adamant compliance other"

## Configs
CONFIG="./configs/model/train/bert-full.default.json"
CONFIG_NAME="full"

############ 
## Setup
############

echo "Going to project directory (full path)"
cd $SCRIPT_DIR

############ 
## Execute
############

echo "~~~~ Starting Cross Validation ~~~~"
time python -u scripts/model/train/bert.py \
    --model_settings $CONFIG \
    --dataset $DATASET \
    --model $MODEL \
    --tokenizer $MODEL \
    --eval_random_state 42 \
    --eval_cv 5 \
    --eval_test \
    --cache_errors \
    --keyword_groups $KEYWORD_CATEGORIES \
    --output_dir "./data/results/model/$DATASET_NAME/final-cv/enhanced-$MODEL_NAME-base-separate-$CONFIG_NAME/" \
    --model_cache_dir "./data/models/classifiers/$DATASET_NAME/final-cv/enhanced-$MODEL_NAME-base-separate-$CONFIG_NAME/" \
    --device $DEVICE \
    --rm_existing

echo "~~~~ Starting No Cross Validation ~~~~"
time python -u scripts/model/train/bert.py \
    --model_settings $CONFIG \
    --dataset $DATASET \
    --model $MODEL \
    --tokenizer $MODEL \
    --eval_random_state 42 \
    --eval_test \
    --cache_errors \
    --keyword_groups $KEYWORD_CATEGORIES \
    --output_dir "./data/results/model/$DATASET_NAME/final-no_cv/enhanced-$MODEL_NAME-base-separate-$CONFIG_NAME/" \
    --model_cache_dir "./data/models/classifiers/$DATASET_NAME/final-no_cv/enhanced-$MODEL_NAME-base-separate-$CONFIG_NAME/" \
    --device $DEVICE \
    --rm_existing

echo "Job Task ID $SLURM_ARRAY_TASK_ID Complete"