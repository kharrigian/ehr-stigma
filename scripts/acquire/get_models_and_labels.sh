#!/bin/bash

## Collect Credentials
echo ">> Please provide your Physionet Credentials"
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD

## Make Resources Directory
echo ">> Initializing Resource Directory"
RESOURCE_DIR="./data/resources/"
mkdir -p $RESOURCE_DIR

## Paths (Can comment out certain paths below if you don't want to download everything)
PROJECT_BASE="https://physionet.org/files/stigmatizing-language/1.0.0/"
ANNOTATION_FILES=(
    "annotations/annotations.csv"
)
MODEL_FILES=(
    "models/mimic-iv-discharge_baseline-majority/"
    "models/mimic-iv-discharge_baseline-statistical/"
    "models/mimic-iv-discharge_base-bert/"
    "models/mimic-iv-discharge_clinical-bert/"
)

echo ">> Downloading Annotations"
for path in ${ANNOTATION_FILES[*]}; do
    echo "[* $path *]"
    wget -P $RESOURCE_DIR --cut-dirs=4 -r -N -c -np -nH --reject="index.html*" -e robots=off -q --show-progress --http-user="$USERNAME" --http-password="$PASSWORD" $PROJECT_BASE/$path
done

echo ">> Downloading Models"
for path in ${MODEL_FILES[*]}; do
    echo "[* $path *]"
    wget -P $RESOURCE_DIR --cut-dirs=4 -r -N -c -np -nH --reject="index.html*" -e robots=off -q --show-progress --http-user="$USERNAME" --http-password="$PASSWORD" $PROJECT_BASE/$path
done

echo ">> All Annotation and Model Resources Downloaded!"