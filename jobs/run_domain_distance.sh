#!/bin/sh

############ 
## Config
############

## Variables NOTE: PUT THE PATH TO YOUR CLONE OF THE REPO HERE AND YOUR CONDA ENVIRONMENT NAME
SCRIPT_DIR='/Users/kharrigian/dev/research/johns-hopkins/ehr-stigma/'

## Device (cuda or cpu)
DEVICE="cpu"

## Choose Dataset/Model
DATASET="mimic-iv-discharge"
DATASET_NAME="mimic-iv/camera-ready"
MODEL_NAME="mimic-iv-discharge_clinical-bert"
MODEL_ROOT="data/resources/models/mimic-iv-discharge_clinical-bert"

############ 
## Setup
############

echo "Going to project directory (full path)"
cd $SCRIPT_DIR

############ 
## Execution
############

echo "Running Program"
time python -u scripts/model/experiments/domain_distance.py \
    --dataset_id $DATASET \
    --output_dir "./data/results/domain-distance/$DATASET_NAME/$MODEL_NAME/" \
    --model_root $MODEL_ROOT \
    --target_domains "keyword" "label" "encounter_type" "patient_gender" "patient_race" \
    --race_ignore "Unknown" "Declined to Answer" "Patient Declined To Answer" "Unable To Obtain" \
    --sex_ignore "Unknown" \
    --race_other "Hispanic or Latino" "Asian" "American Indian or Native" "American Indian or Alaska Native" "Portuguese" "South American" "Mixed" \
    --training_only \
    --device $DEVICE \
    --verbose

