#!/bin/sh

#######
## Parameters
#######

## Variables NOTE: PUT THE PATH TO YOUR CLONE OF THE REPO HERE AND YOUR CONDA ENVIRONMENT NAME
SCRIPT_DIR='/Users/kharrigian/dev/research/johns-hopkins/ehr-stigma/'

## Source
NOTE_FILE="./data/resources/annotations/matches.csv"
METADATA_FILE="./data/resources/annotations/matches.metadata.csv"

## Model
ROOT_MODEL_DIR="./data/models/classifiers/mimic-iv/camera-ready/final-no_cv/baseline-statistical/"
MODEL_VARIATION="keyword_tokens_tfidf/linear"

## Output Directory
OUTPUT_DIR="./data/results/predictions/mimic-iv-discharge/mimic-iv-discharge_baseline-statistical/"

############ 
## Setup
############

echo "Going to project directory (full path)"
cd $SCRIPT_DIR

#######
## Execution
#######

echo "Starting Adamant Application"
python -u scripts/model/apply/baseline.py \
    --model $ROOT_MODEL_DIR/$MODEL_VARIATION/adamant_fold-0/ \
    --preprocessing $ROOT_MODEL_DIR/preprocessing.params.joblib \
    --note_file $NOTE_FILE \
    --keyword_category adamant \
    --output_dir $OUTPUT_DIR \
    --cache_note_text \
    --rm_existing

python -u scripts/model/apply/baseline.py \
    --model $ROOT_MODEL_DIR/$MODEL_VARIATION/compliance_fold-0/ \
    --preprocessing $ROOT_MODEL_DIR/preprocessing.params.joblib \
    --note_file $NOTE_FILE \
    --keyword_category compliance \
    --output_dir $OUTPUT_DIR \
    --cache_note_text \
    --rm_existing

python -u scripts/model/apply/baseline.py \
    --model $ROOT_MODEL_DIR/$MODEL_VARIATION/other_fold-0/ \
    --preprocessing $ROOT_MODEL_DIR/preprocessing.params.joblib \
    --note_file $NOTE_FILE \
    --keyword_category other \
    --output_dir $OUTPUT_DIR \
    --cache_note_text \
    --rm_existing

echo "Application Complete"