
"""
Search for keywords (anchors) in the MIMIC Discharge Dataset. Merge text into annotations
"""

########################
### Imports
########################

## Standard Library
import os
import sys
import json
import argparse
from collections import Counter

## External Libraries
import pandas as pd
from tqdm import tqdm

## Private
from stigma import (util,
                    text_utils,
                    settings)

########################
### Functions
########################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--annotations_dir", type=str, default=settings.DEFAULT_ANNOTATIONS_DIR)
    _ = parser.add_argument("--keywords", type=str, default=settings.DEFAULT_KEYWORDS)
    _ = parser.add_argument("--load_chunksize", type=int, default=1000)
    _ = parser.add_argument("--load_chunk_sample_rate", type=float, default=None, help="If desired, only sample this proportion of chunks.")
    _ = parser.add_argument("--load_note_sample_rate", type=float, default=None, help="If desired, only sample this proportion of notes from each chunk.")
    _ = parser.add_argument("--load_skip_cleaning", action="store_true", default=False)
    _ = parser.add_argument("--load_random_state", type=int, default=42)
    _ = parser.add_argument("--load_window_size", type=int, default=10)
    _ = parser.add_argument("--load_annotations_only", action="store_true", default=False, help="If desired, only look for matches amongst notes in annotated label set.")
    args = parser.parse_args()
    return args

def _verify_mimic_paths():
    """
    
    """
    ## Check for Directory
    if not os.path.exists(settings.MIMIC_SOURCE_DIR):
        raise FileNotFoundError(f"Unable to locate MIMIC data directory: '{settings.MIMIC_SOURCE_DIR}'")
    ## Iterate Through Files
    mimic_files = ["admissions.csv.gz","diagnoses_icd.csv.gz","discharge.csv.gz","patients.csv.gz","services.csv.gz","transfers.csv.gz"]
    missing_files = False
    for mf in mimic_files:
        if not os.path.exists(f"{settings.MIMIC_SOURCE_DIR}/{mf}"):
            missing_files = True
            print(f">> WARNING - Missing expected MIMIC data file: '{mf}'")
    ## Raise Error if Necessary
    if missing_files:
        raise FileNotFoundError("Unable to locate all required MIMIC dataset files.")
    return True

def _verify_keywords(keyword_file):
    """
    
    """
    ## Check for File
    if not os.path.exists(keyword_file):
        raise FileNotFoundError(f"Unable to locate keyword file: '{keyword_file}'")
    return True

def run_search(args):
    """
    
    """
    ## Check For Files
    print("[Verifying Data Paths]")
    _ = _verify_mimic_paths()
    _ = _verify_keywords(keyword_file=args.keywords)
    ## Initialize Output Directory
    if not os.path.exists(args.annotations_dir):
        print(f"[Initializing Annotation Output Directory: '{args.annotations_dir}']")
        _ = os.makedirs(args.annotations_dir)
    ## Load Keywords
    print("[Loading Keywords]")
    with open(f"{args.keywords}","r") as the_file:
        task2keyword = json.load(the_file)
    ## Load Labels
    labels = None
    if os.path.exists(f"{args.annotations_dir}/annotations.csv"):
        print("[Loading Known Labels]")
        labels = pd.read_csv(f"{args.annotations_dir}/annotations.csv")
    if labels is None and args.load_annotations_only:
        raise FileNotFoundError("Must have downloaded annotations.csv file to load matches for annotations only.")
    ## Initialize Search Module
    print("[Initializing Keyword Search Tool]")
    searcher = text_utils.KeywordSearch(task2keyword=task2keyword)
    ## Initialize Match Cache
    matches = []
    source = []
    n_documents = 0
    n_documents_matched = 0
    ## Iterate Through Chunks
    for n_df in util.load_mimic_iv_discharge(clean_text=not args.load_skip_cleaning,
                                             normalize_text=not args.load_skip_cleaning,
                                             as_iterator=True,
                                             sample_rate_chunk=args.load_chunk_sample_rate,
                                             sample_rate_note=args.load_note_sample_rate,
                                             chunksize=args.load_chunksize,
                                             random_state=args.load_random_state,
                                             verbose=True,
                                             encounter_note_ids=set(labels["encounter_note_id"].values) if (args.load_annotations_only and labels is not None) else None):
        ## Uodate Processed Documents
        n_documents += n_df.shape[0]
        ## Cache Source
        source.append(n_df.drop(["note_text"],axis=1))
        ## Run Search
        n_df["matches"] = n_df["note_text"].map(searcher.search)
        ## Filter
        n_df = n_df.loc[n_df["matches"].map(len) > 0, :]
        ## Update Matched Document Document
        n_documents_matched += n_df.shape[0]
        ## Cache Matches
        if n_df.shape[0] > 0:
            matches.append(n_df)
    print("[Pattern Search Complete. Searched {:,d} Documents. {:,d} Contained a Match.]".format(n_documents, n_documents_matched))
    ## Format Matches
    print("[Formatting Source and Matches]")
    matches = pd.concat(matches,axis=0).reset_index(drop=True)    
    source = pd.concat(source,axis=0).reset_index(drop=True)
    ## Aggregate Statistics
    print("[Gathering Match Statistics]")
    match_statistics = {task:Counter() for task in task2keyword.keys()}
    for match in matches["matches"].values:
        for span in match:
            match_statistics[span["task"]][span["keyword"]] += 1
    match_statistics = pd.concat({x:pd.Series(y).sort_values(ascending=False) for x, y in match_statistics.items()})
    ## Show Statistics
    print("[Match Statistics]")
    print(match_statistics.to_string())
    ## Format Into Context Windos
    match_windows = []
    for _, row in tqdm(matches.iterrows(), total=matches.shape[0], file=sys.stdout, desc="[Extracting Match Windows]"):
        ## Iterate Through Specific Matches
        for span in row["matches"]:
            ## Extract
            span_window = text_utils.get_context_window(text=row["note_text"],
                                                  start=span["start"],
                                                  end=span["end"],
                                                  window_size=args.load_window_size,
                                                  clean_normalize=args.load_skip_cleaning,
                                                  strip_operators=True)
            ## Cache
            match_windows.append({
                "enterprise_mrn":row["enterprise_mrn"],
                "encounter_id":row["encounter_id"],
                "encounter_note_id":row["encounter_note_id"],
                "encounter_type":row["encounter_type"],
                "encounter_date":row["encounter_date"],
                "encounter_note_service":row["encounter_note_service"],
                "encounter_note_unit":row["encounter_note_unit"],
                "keyword_category":span["task"],
                "keyword":span["keyword"],
                "start":span["start"],
                "end":span["end"],
                "note_text":span_window,
            })
    ## Concatenate
    match_windows = pd.DataFrame(match_windows)
    ## Build
    print("[Loading Additional Metadata]")
    metadata = util.build_mimic_iv_discharge_metadata(verbose=True)
    ## Add Note-level Metadata
    metadata = metadata.loc[metadata["encounter_id"].isin(source["encounter_id"]),:].reset_index(drop=True)
    if source["encounter_id"].value_counts().max() == 1:
        metadata = pd.merge(metadata, source[["encounter_id","encounter_note_service","encounter_note_unit"]], on=["encounter_id"], how="left")
    else:
        raise ValueError("Found multiple encounters for a single discharge note.")
    ## Annotation Augmentation
    if labels is not None:
        ## Augment Labels with Match Text + Selected Metadata
        print("[Augmenting Known Labels]")
        labels = pd.merge(match_windows,
                          labels.drop([a for a in labels.columns if a.startswith("annotator_")], axis=1),
                          on=["encounter_note_id","keyword_category","keyword","start","end"],
                          how="right")
    ## Cache
    print("[Caching Artifacts]")
    _ = match_windows.to_csv(f"{args.annotations_dir}/matches.csv",index=False,encoding="utf-8")
    _ = source.to_csv(f"{args.annotations_dir}/matches.source.csv",index=False,encoding="utf-8")
    _ = metadata.to_csv(f"{args.annotations_dir}/matches.metadata.csv", index=False)
    if labels is not None:
        _ = labels.to_csv(f"{args.annotations_dir}/annotations.augmented.csv",index=False)

def main():
    """
    
    """
    ## Get Arguments
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Run Desired Process
    _ = run_search(args)
    ## Done
    print("[Script Complete]")

#####################
### Execute
#####################

if __name__ == "__main__":
    _ = main()
    
