"""
Generic utility functions, primarily used for loading data
"""

#####################
### Imports
#####################

## Standard Library
import os
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd

## Local
from . import settings
from .text_utils import (clean_excel_text,
                         normalize_excel_text)

#####################
### Functions
#####################

def flatten(l):
    """
    
    """
    return [x for s in l for x in s]

def chunks(l, n):
    """

    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _timeline_bounds(timeline):
    """
    
    """
    timeline = timeline + [(None, None)]
    timeline_bounds = []
    for s, e in zip(timeline[:-1], timeline[1:]):
        timeline_bounds.append((s[0], e[0], s[1]))
    return timeline_bounds

def _load_mimic_iv_patient_timeline(verbose=True):
    """
    
    """
    ## Services
    if verbose:
        print("[Loading Service Timeline]")
    services = pd.read_csv(f"{settings.MIMIC_SOURCE_DIR}/services.csv.gz")
    services["transfertime"] = pd.to_datetime(services["transfertime"])
    services["timeline"] = services[["transfertime","curr_service"]].apply(tuple,axis=1)
    services = services.groupby(["subject_id","hadm_id"]).agg({"timeline":lambda x: sorted(x, key=lambda i: i[0])})
    services["timeline"] = services["timeline"].map(_timeline_bounds)
    ## Transfers
    if verbose:
        print("[Loading Transfer Timeline]")
    transfers = pd.read_csv(f"{settings.MIMIC_SOURCE_DIR}/transfers.csv.gz")
    transfers["intime"] = pd.to_datetime(transfers["intime"])
    transfers["outtime"] = pd.to_datetime(transfers["outtime"])
    hadm_id_rep = []
    hadm_id_count = Counter()
    for subject, hadm_id in transfers[["subject_id","hadm_id"]].values:
        if not pd.isnull(hadm_id):
            hadm_id_rep.append(int(hadm_id))
        else:
            hadm_id_count[subject] += 1
            hadm_id_rep.append(-hadm_id_count[subject])
    transfers["hadm_id"] = hadm_id_rep
    transfers["timeline"] = transfers[["intime","outtime","eventtype","careunit"]].apply(tuple,axis=1)
    transfers = transfers.groupby(["subject_id","hadm_id"]).agg({"timeline":lambda x: sorted(x, key=lambda i: i[0])})
    ## Return
    return services, transfers

def build_mimic_iv_discharge_metadata(verbose=True):
    """
    
    """
    ## Load Metadata
    if verbose:
        print("[Loading Metadata]")
    patients = pd.read_csv(f"{settings.MIMIC_SOURCE_DIR}/patients.csv.gz")
    admissions = pd.read_csv(f"{settings.MIMIC_SOURCE_DIR}/admissions.csv.gz")
    diagnoses = pd.read_csv(f"{settings.MIMIC_SOURCE_DIR}/diagnoses_icd.csv.gz")
    ## Format Patients
    if verbose:
        print("[Formatting Patients]")
    patient_cols = {
        "subject_id":"enterprise_mrn",
        "gender":"patient_gender",
        "anchor_age":"patient_age",
        "anchor_year_group":"patient_time_period",
    }
    patients = patients.loc[:,list(patient_cols.keys())]
    patients = patients.rename(columns=patient_cols)
    patients["patient_gender"] = patients["patient_gender"].map(lambda i: {"M":"Male","F":"Female"}.get(i))
    ## Format Admissions
    if verbose:
        print("[Formatting Admissions]")
    admission_cols = {
        "hadm_id":"encounter_id",
        "subject_id":"enterprise_mrn",
        "admittime":"encounter_date_start",
        "dischtime":"encounter_date_end",
        "admission_type":"patient_admission_type",
        "race":"patient_race",
        "language":"patient_language",
        "marital_status":"patient_marital_status",
        "insurance":"patient_insurance_type",
    }
    admissions = admissions.loc[:,list(admission_cols.keys())]
    admissions = admissions.rename(columns=admission_cols)
    admissions["encounter_date_start"] = pd.to_datetime(admissions["encounter_date_start"])
    admissions["encounter_date_end"] = pd.to_datetime(admissions["encounter_date_end"])
    admissions["patient_marital_status"] = admissions["patient_marital_status"].map(lambda i: i if not isinstance(i, str) else "Unknown" if i.startswith("UNKNOWN") else i.title())
    admissions["patient_language"] = admissions["patient_language"].map(lambda i: i if not isinstance(i, str) else "Unknown" if i == "?" else i.title())
    admissions["patient_admission_type"] = admissions["patient_admission_type"].map(lambda i: i if not isinstance(i, str) else i.title())
    admissions["patient_race"] = admissions["patient_race"].map(lambda i: {'WHITE': 'White or Caucasian',
                'BLACK/AFRICAN AMERICAN': 'Black or African American',
                'BLACK/AFRICAN': 'Black or African American',
                'BLACK/CAPE VERDEAN': 'Black or African American',
                'BLACK/CARIBBEAN ISLAND':"Black or African American",
                'WHITE - RUSSIAN': 'White or Caucasian',
                "WHITE - EASTERN EUROPEAN":"White or Caucasian",
                'WHITE - BRAZILIAN': 'White or Caucasian',
                'WHITE - OTHER EUROPEAN': 'White or Caucasian',
                'PORTUGUESE': 'Portuguese',
                'AMERICAN INDIAN/ALASKA NATIVE':"American Indian or Native",
                "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER":"American Indian or Native",
                "MULTIPLE RACE/ETHNICITY":"Mixed",
                'HISPANIC OR LATINO': 'Hispanic or Latino',
                'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic or Latino',
                "HISPANIC/LATINO - COLUMBIAN":"Hispanic or Latino",
                "HISPANIC/LATINO - CUBAN":"Hispanic or Latino",
                "HISPANIC/LATINO - DOMINICAN":"Hispanic or Latino",
                "HISPANIC/LATINO - GUATEMALAN":"Hispanic or Latino",
                "HISPANIC/LATINO - HONDURAN":"Hispanic or Latino",
                "HISPANIC/LATINO - MEXICAN":"Hispanic or Latino",
                "HISPANIC/LATINO - SALVADORAN":"Hispanic or Latino",
                "HISPANIC/LATINO - CENTRAL AMERICAN":"Hispanic or Latino",
                "SOUTH AMERICAN":"South American",
                'ASIAN': 'Asian',
                'ASIAN - SOUTH EAST ASIAN':"Asian",
                'ASIAN - KOREAN': 'Asian',
                'ASIAN - ASIAN INDIAN': 'Asian',
                'ASIAN - CHINESE': 'Asian',
                'OTHER': 'Other',
                'UNKNOWN': 'Unknown',
                'UNABLE TO OBTAIN': 'Unknown',
                'PATIENT DECLINED TO ANSWER': 'Unknown'}.get(i,i))
    ## Format ICD Codes
    if verbose:
        print("[Formatting ICD Codes]")
    diagnoses_ = []
    for code in [9,10]:
        code_subset = diagnoses.loc[diagnoses["icd_version"]==code].copy()
        code_col = f"patient_icd_{code}_codes"
        code_subset = code_subset.rename(columns={"icd_code":code_col,"hadm_id":"encounter_id"})
        code_subset = code_subset[["encounter_id",code_col]]
        code_subset = code_subset.dropna(subset=[code_col])
        code_subset = code_subset.groupby("encounter_id").agg({code_col:lambda x: "; ".join(x)})
        diagnoses_.append(code_subset)
    diagnoses = pd.merge(diagnoses_[0], diagnoses_[1], left_index=True, right_index=True, how="outer").reset_index()
    ## Merge
    if verbose:
        print("[Merging DataFrames]")
    merged = pd.merge(admissions,
                      patients,
                      on=["enterprise_mrn"],
                      how="left")
    merged = pd.merge(merged,
                      diagnoses,
                      on=["encounter_id"],
                      how="left")
    ## Return
    return merged

def _assign_mimic_iv_unit_service(row, services, transfers):
    """
    
    """
    ## Initialize Default Service + Unit
    row_service = None
    row_unit = None
    ## Get Row Information
    service_timeline = services.loc[(row["enterprise_mrn"],row["encounter_id"])]["timeline"]
    transfer_timeline = transfers.loc[(row["enterprise_mrn"],row["encounter_id"])]["timeline"]
    ## Service/Transer Information
    row_service = service_timeline[-1][-1]
    if transfer_timeline[-1][-2] != "discharge":
        raise ValueError("Unexpected transfer timeline for discharge notes.")
    else:
        row_unit = transfer_timeline[-2][-1]
    ## Return
    return row_service, row_unit

def _load_mimic_iv_discharge(clean_text=False,
                             normalize_text=False,
                             sample_rate_chunk=None,
                             sample_rate_note=None,
                             chunksize=1000,
                             random_state=42,
                             verbose=True):
    """
    
    """
    ## Align Matches with Patient Timeline
    services, transfers = _load_mimic_iv_patient_timeline(verbose=verbose)
    ## Initialize
    df = pd.read_csv(f"{settings.MIMIC_SOURCE_DIR}/discharge.csv.gz",
                     low_memory=False,
                     iterator=True,
                     chunksize=chunksize,
                     encoding="latin-1")
    ## Seeds
    chunk_sampler = np.random.RandomState(random_state)
    ## Run Search
    for n, n_df in enumerate(df):
        if verbose:
            print(f"[Loading Chunk {n}]")
        ## Sampling
        if sample_rate_chunk is not None and chunk_sampler.random() > sample_rate_chunk:
            continue
        if sample_rate_note is not None:
            n_df = n_df.sample(frac=sample_rate_note, random_state=chunk_sampler, axis=0, replace=False)
            n_df = n_df.sort_index()
        ## Text Normalization
        if clean_text:
            n_df["text"] = n_df["text"].map(clean_excel_text)
        if normalize_text:
            n_df["text"] = n_df["text"].map(normalize_excel_text)
        ## Update Column Names
        n_df = n_df.rename(columns={
            "note_id":"encounter_note_id",
            "subject_id":"enterprise_mrn",
            "hadm_id":"encounter_id",
            "note_type":"encounter_type",
            "charttime":"encounter_date",
            "storetime":"encounter_date_aux",
            "text":"note_text",
        })
        ## Transfers/Services
        service_units = n_df.apply(lambda row: _assign_mimic_iv_unit_service(row, services, transfers), axis=1)
        n_df["encounter_note_service"] = [i[0] for i in service_units]
        n_df["encounter_note_unit"] = [i[1] for i in service_units]
        ## Yield
        yield n_df

def load_mimic_iv_discharge(clean_text=False,
                            normalize_text=False,
                            as_iterator=False,
                            sample_rate_chunk=None,
                            sample_rate_note=None,
                            chunksize=1000,
                            random_state=42,
                            verbose=False):
    """
    
    """
    ## Iterate Through Sources
    out =_load_mimic_iv_discharge(clean_text=clean_text,
                                  normalize_text=normalize_text,
                                  sample_rate_chunk=sample_rate_chunk,
                                  sample_rate_note=sample_rate_note,
                                  chunksize=chunksize,
                                  random_state=random_state,
                                  verbose=verbose)
    if not as_iterator:
        out = pd.concat(list(out), axis=0).reset_index(drop=True)
    return out

def load_annotations_mimic_iv_discharge(annotation_file=f"{settings.DEFAULT_ANNOTATIONS_DIR}/annotations.augmented.csv"):
    """
    
    """
    ## Establish and Check for file
    if not os.path.exists(annotation_file):
        raise FileNotFoundError("Please run scripts/acquire/build_mimic.py to create augmented annotation file.")
    ## Load Annotations
    annotations = pd.read_csv(annotation_file)
    ## Verify Everything Exists as Expected
    required_cols = [
        "enterprise_mrn",
        "encounter_id",
        "encounter_note_id",
        "encounter_type",
        "encounter_date",
        "encounter_note_service",
        "encounter_note_unit",
        "keyword_category",
        "keyword",
        "start",
        "end",
        "note_text",
        "label"
    ]
    missing_cols = [i for i in required_cols if i not in annotations.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing the following columns for the annotation_file: {missing_cols}")
    ## Return
    return annotations

def load_annotations_metadata_mimic_iv_discharge(annotations,
                                                 clean_text=False,
                                                 normalize_text=True,
                                                 load_all=False,
                                                 load_source=False,
                                                 random_state=42,
                                                 verbose=True,
                                                 **kwargs):
    """
    
    """
    ## Load Metadata
    metadata = build_mimic_iv_discharge_metadata(verbose=verbose)
    ## Split Assignment
    patients = sorted(metadata["enterprise_mrn"].unique())
    split_seed = np.random.RandomState(random_state)
    split_assign = split_seed.choice(["train","dev","test"], p=[0.7,0.2,0.1], size=len(patients), replace=True)
    patient2split = dict(zip(patients, split_assign))
    metadata["split"] = metadata["enterprise_mrn"].map(patient2split.get)
    assert not metadata["split"].isnull().any()
    ## Load Sources
    if load_source:
        if verbose:
            print("[Loading Source Data]")
        sources = load_mimic_iv_discharge(clean_text=False, normalize_text=False, as_iterator=False, verbose=verbose)
        ## Subset
        if not load_all and annotations is not None:
            sources = sources.loc[sources["encounter_note_id"].isin(annotations["encounter_note_id"]),:].reset_index(drop=True)
        ## Cleaning
        if clean_text:
            sources["note_text"] = sources["note_text"].map(clean_excel_text)
        if normalize_text:
            sources["note_text"] = sources["note_text"].map(normalize_excel_text)
        ## Merging
        merged_source_metadata = pd.merge(sources,
                                          metadata,
                                          on=["encounter_id","enterprise_mrn"],
                                          how="left")
        ## Column Name Update
        merged_source_metadata = merged_source_metadata.rename(columns={"note_text":"note_text_full"})
    else:
        ## Isolate
        merged_source_metadata = metadata
        ## Subset
        if not load_all and annotations is not None:
            merged_source_metadata = merged_source_metadata.loc[merged_source_metadata["encounter_id"].isin(annotations["encounter_id"]),:].reset_index(drop=True)
    ## Return Values
    ret_values = [merged_source_metadata, None] if not load_all else [None, merged_source_metadata]
    ## Return
    return ret_values