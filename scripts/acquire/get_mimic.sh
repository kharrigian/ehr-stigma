#!/bin/bash

## Collect Credentials
echo ">> Please provide your Physionet Credentials"
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD

## Make Resource Directory
echo ">> Initializing MIMIC-IV Resource Directory"
RESOURCE_DIR="./data/resources/datasets/mimic-iv/"
mkdir -p $RESOURCE_DIR

## Root Paths
MIMIC_IV_BASE="https://physionet.org/files/mimiciv/2.2/"
MIMIC_IV_BASE_FILES="hosp/admissions.csv.gz hosp/patients.csv.gz hosp/diagnoses_icd.csv.gz hosp/services.csv.gz hosp/transfers.csv.gz"
MIMIC_IV_NOTES="https://physionet.org/files/mimic-iv-note/2.2/"
MIMIC_IV_NOTES_FILES="note/discharge.csv.gz"

echo ">> Downloading MIMIC-IV Base Dataset"
for path in $MIMIC_IV_BASE_FILES; do
    echo "[* $path *]"
    wget -P $RESOURCE_DIR -r -N -c -np -nd -q --show-progress --http-user="$USERNAME" --http-password="$PASSWORD" $MIMIC_IV_BASE/$path
done

echo ">> Downloading MIMIC-IV Notes Dataset"
for path in $MIMIC_IV_NOTES_FILES; do
    echo "[* $path *]"
    wget -P $RESOURCE_DIR -r -N -c -np -nd -q --show-progress --http-user="$USERNAME" --http-password="$PASSWORD" $MIMIC_IV_NOTES/$path
done

echo ">> All MIMIC-IV Resources Downloaded!"