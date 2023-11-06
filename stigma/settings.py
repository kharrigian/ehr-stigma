
####################
### Imports
####################

## Standard Library
import os
import json

####################
### Paths
####################

## Helper Path (Root Repo Directory)
_ROOT_DIR = os.path.dirname(__file__) + "/../"

## Path to Raw MIMIC-IV 2.2 Dataset (Downloaded from Physionet - see get_mimic.sh)
MIMIC_SOURCE_DIR = f"{_ROOT_DIR}/data/resources/datasets/mimic-iv/"

## Default Keywords (Anchors)
DEFAULT_KEYWORDS = f"{_ROOT_DIR}/data/resources/keywords/keywords.json"

## Default Annotations
DEFAULT_ANNOTATIONS_DIR = f"{_ROOT_DIR}/data/resources/annotations/"

## Model Paths (Should align with Bert Models Should Align with BERT_TOKENIZERS)
MODELS = {
    "mimic-iv-discharge_majority_overall":{
        "model_type":"baseline",
        "preprocessing":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-majority/preprocessing.params.joblib"),
        "tasks":{
            "adamant":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-majority/keyword/majority/adamant_fold-0/"),
            "compliance":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-majority/keyword/majority/compliance_fold-0/"),
            "other":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-majority/keyword/majority/other_fold-0/"),
        }
    },
    "mimic-iv-discharge_majority_keyword":{
        "model_type":"baseline",
        "preprocessing":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/preprocessing.params.joblib"),
        "tasks":{
            "adamant":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/keyword/linear/adamant_fold-0/"),
            "compliance":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/keyword/linear/compliance_fold-0/"),
            "other":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/keyword/linear/other_fold-0/"),
        }
    },
    "mimic-iv-discharge_logistic-regression_context":{
        "model_type":"baseline",
        "preprocessing":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/preprocessing.params.joblib"),
        "tasks":{
            "adamant":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/tokens_tfidf/linear/adamant_fold-0/"),
            "compliance":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/tokens_tfidf/linear/compliance_fold-0/"),
            "other":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/tokens_tfidf/linear/other_fold-0/"),
        }
    },
    "mimic-iv-discharge_logistic-regression_keyword-context":{
        "model_type":"baseline",
        "preprocessing":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/preprocessing.params.joblib"),
        "tasks":{
            "adamant":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/keyword_tokens_tfidf/linear/adamant_fold-0/"),
            "compliance":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/keyword_tokens_tfidf/linear/compliance_fold-0/"),
            "other":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_baseline-statistical/keyword_tokens_tfidf/linear/other_fold-0/"),
        }
    },
    "mimic-iv-discharge_base-bert":{
        "model_type":"bert",
        "preprocessing":None,
        "tasks":{
            "adamant":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_base-bert/adamant_fold-0/checkpoint-250/"),
            "compliance":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_base-bert/compliance_fold-0/checkpoint-100/"),
            "other":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_base-bert/other_fold-0/checkpoint-250/")
        }
    },
    "mimic-iv-discharge_clinical-bert":{
        "model_type":"bert",
        "preprocessing":None,
        "tasks":{
            "adamant":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_clinical-bert/adamant_fold-0/checkpoint-50/"),
            "compliance":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_clinical-bert/compliance_fold-0/checkpoint-400/"),
            "other":os.path.abspath(f"{_ROOT_DIR}/data/resources/models/mimic-iv-discharge_clinical-bert/other_fold-0/checkpoint-350/")
        }
    }
}

## BERT Tokenizers (Should align with MODELS)
TOKENIZERS = {
    "mimic-iv-discharge_base-bert":"bert-base-uncased",
    "mimic-iv-discharge_clinical-bert":"emilyalsentzer/Bio_ClinicalBERT"
}

####################
### Verify Expected Paths
####################

## Warnings (Not Actually Required for Some Functionality)
if not os.path.exists(MIMIC_SOURCE_DIR):
    print(f"WARNING: Did not find MIMIC-IV source data: {MIMIC_SOURCE_DIR}")
if not os.path.exists(DEFAULT_ANNOTATIONS_DIR):
    print(f"WARNING: Annotations directory not found: {DEFAULT_ANNOTATIONS_DIR}")

## Required For Nearly All Functionality
if not os.path.exists(DEFAULT_KEYWORDS):
    raise FileNotFoundError(f"Did not find Keyword Map: {DEFAULT_KEYWORDS}")

####################
### Resources
####################

## Keywords
with open(DEFAULT_KEYWORDS,"r") as the_file:
    CAT2KEYS = json.load(the_file)

## Keyword - Category Map
KEY2CAT = {}
for keyword_category, keyword_list in CAT2KEYS.items():
    for keyword in keyword_list:
        if keyword in KEY2CAT:
            raise KeyError("Found duplicated keywords.")
        KEY2CAT[keyword] = keyword_category

####################
### Special Parameters
####################

MIMIC_IV_PERFORMANCE_GROUPS = {
    "keyword":list(KEY2CAT.keys()),
    "patient_gender":[
        "Male",
        "Female"
    ],
    "patient_race":[
        "White or Caucasian",
        "Black or African American",
        "Hispanic or Latino",
        "Asian",
    ],
    "patient_insurance_type":[
        "Other",
        "Medicare",
        "Medicaid",
    ],
}