# Characterization of Stigmatizing Language in Medical Records

## Preliminaries

1. Get MIMIC data `bash ./scripts/acquire/get_mimic.sh`. Will ask for username and password. Must have signed data-usage agreements at respective websites:

    - Base Dataset: https://physionet.org/content/mimiciv/2.2/
    - Notes Dataset: https://physionet.org/content/mimic-iv-note/2.2/

2. Download Our Pretrained Models and Annotations from Physionet. Like above, this will require a username and password.

    - Our Resources (annotations.csv, model directory): TBD

#### Expected Resource Structure

```
data/
    resources/
            anchors/
                anchors.json
            annotations/
                annotations.csv
            datasets/
                mimic-iv/
                    admissions.csv.gz
                    diagnoses_icd.csv.gz
                    discharge.csv.gz
                    patients.csv.gz
                    services.csv.gz
                    transfers.csv.gz
            models/
                mimic-iv-discharge_clinical-bert/
                        adamant_fold-0/
                        compliance_fold-0/
                        other_fold-0/
```

## Installation

```
pip install -r requirements.txt
pip install -e .
```