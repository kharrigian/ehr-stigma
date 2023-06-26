#!/bin/sh

#######
## Parameters
#######

## Variables NOTE: PUT THE PATH TO YOUR CLONE OF THE REPO HERE AND YOUR CONDA ENVIRONMENT NAME
SCRIPT_DIR='/Users/kharrigian/dev/research/johns-hopkins/ehr-stigma/'

## Model Device (cuda or cpu)
DEVICE="cpu"

## Source
NOTE_FILE="./data/resources/annotations/matches.csv"
METADATA_FILE="./data/resources/annotations/matches.metadata.csv"

## Model
ROOT_MODEL_DIR="./data/resources/models/mimic-iv-discharge_clinical-bert/"
ROOT_MODEL_TOKENIZER="emilyalsentzer/Bio_ClinicalBERT"

## Model Checkpoints
ADAMANT_CHECKPOINT=50
COMPLIANCE_CHECKPOINT=400
OTHER_CHECKPOINT=350

## Output Directory
OUTPUT_DIR="./data/results/predictions/mimic-iv-discharge/mimic-iv-discharge_clinical-bert/"

############ 
## Setup
############

echo "Going to project directory (full path)"
cd $SCRIPT_DIR

#######
## Execution
#######

echo "Starting Adamant Application"
python -u scripts/model/apply/bert.py \
    --keyword_category adamant \
    --model "$ROOT_MODEL_DIR/adamant_fold-0/checkpoint-$ADAMANT_CHECKPOINT/" \
    --tokenizer $ROOT_MODEL_TOKENIZER \
    --note_file $NOTE_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 16 \
    --cache_note_text \
    --device $DEVICE

echo "Starting Compliance Application"
python -u scripts/model/apply/bert.py \
    --keyword_category compliance \
    --model "$ROOT_MODEL_DIR/compliance_fold-0/checkpoint-$COMPLIANCE_CHECKPOINT/" \
    --tokenizer $ROOT_MODEL_TOKENIZER \
    --note_file $NOTE_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 16 \
    --cache_note_text \
    --device $DEVICE

echo "Starting Other Descriptors Application"
python -u scripts/model/apply/bert.py \
    --keyword_category other \
    --model "$ROOT_MODEL_DIR/other_fold-0/checkpoint-$OTHER_CHECKPOINT/" \
    --tokenizer $ROOT_MODEL_TOKENIZER \
    --note_file $NOTE_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 16 \
    --cache_note_text \
    --device $DEVICE

echo "Application Complete"