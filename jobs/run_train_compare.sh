#!/bin/sh

############ 
## Config
############

## Variables NOTE: PUT THE PATH TO YOUR CLONE OF THE REPO HERE AND YOUR CONDA ENVIRONMENT NAME
SCRIPT_DIR='/Users/kharrigian/dev/research/johns-hopkins/ehr-stigma/'

## NOTE: Configuration is done in the header of scripts/model/train/compare.py. This just serves
## as an executable

############ 
## Setup
############

echo "Going to project directory (full path)"
cd $SCRIPT_DIR

############ 
## Execution
############

echo "Running Program"
time python -u scripts/model/train/compare.py \
    --plot_splits train dev test
