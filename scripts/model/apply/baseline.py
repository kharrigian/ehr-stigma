
"""
Apply a trained baseline classifier to a set of raw notes
"""

#########################
### Imports
#########################

## Standard Library
import os
import json
import argparse

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import dummy

## Private
from stigma import settings
from stigma import util
from stigma.model import util as model_utils

##########################
### Functions
##########################

def parse_command_line():
    """
    
    """
    ## Initialize Parser and Execute
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--model", type=str, default=None, help="Path to trained baseline classifier directory (includes classifier, preprocessor, and targets)")
    _ = parser.add_argument("--preprocessing", type=str, default=None, help="Path to preprocessing.params.joblib file")
    _ = parser.add_argument("--note_file", type=str, default=None, help="Path to processed note file to use as an input to the model.")
    _ = parser.add_argument("--annotations", type=str, choices={"mimic-iv-discharge"}, help="Load annotations instead of using a note file.")
    _ = parser.add_argument("--keyword_category", type=str, default=None, choices={"adamant","compliance","other"})
    _ = parser.add_argument("--output_dir", type=str, default=None, help="Name of the directory to use for storing predictions.")
    _ = parser.add_argument("--rm_existing", action="store_true", default=False, help="If included, will overwrite any existing prediction file.")
    _ = parser.add_argument("--n_samples", type=int, default=None, help="If desired, downsample the note data.")
    _ = parser.add_argument("--random_state", type=int, default=42, help="If downsampling, use this random seed.")
    _ = parser.add_argument("--batch_size", type=int, default=None, help="If desired, make predictions in incremental chunks of this size.")
    _ = parser.add_argument("--cache_note_text", action="store_true", default=False, help="If desired, store context used for prediction.")
    args = parser.parse_args()
    ## Check Parameters
    if args.model is None or not os.path.exists(args.model):
        raise FileNotFoundError("Must provide a valid model directory (--model)")
    if args.preprocessing is None or not os.path.exists(args.preprocessing):
        raise FileNotFoundError("Must provide a valid preprocessing cache (--preprocessing)")
    if args.annotations is None and (args.note_file is None or not os.path.exists(args.note_file)):
        raise FileNotFoundError("Must provide a valid note file (--note_file)")
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    if os.path.exists(f"{args.output_dir}/{args.keyword_category}.predictions.csv") and not args.rm_existing:
        raise FileExistsError("Must include --rm_existing to overwrite existing predictions for this keyword.")
    return args

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Cache Command Line Args
    with open(f"{args.output_dir}/{args.keyword_category}.cfg.json","w") as the_file:
        json.dump(vars(args), the_file, indent=2)
    ## Load Data
    if args.note_file is not None:
        print("[Loading Note File]")
        note_df = pd.read_csv(args.note_file)
    elif args.annotations is not None:
        print(f"[Loading Annotations: {args.annotations}]")
        if args.annotations == "mimic-iv-discharge":
            note_df = util.load_annotations_mimic_iv_discharge()
    ## Additional Columns
    print("[Appending Relevant Metadata to Notes]")
    note_df["keyword_category"] = note_df["keyword"].map(lambda k: settings.KEY2CAT.get(k))
    ## Isolate Keyword Category
    print("[Isolating Target Keyword Category]")
    note_df = note_df.loc[note_df["keyword"].map(lambda i: settings.KEY2CAT[i] == args.keyword_category)].reset_index(drop=True)
    ## Downsample
    if args.n_samples is not None:
        print("[Downsampling Notes]")
        note_df = note_df.sample(n=min(args.n_samples, note_df.shape[0]), random_state=args.random_state, replace=False).sort_index().reset_index(drop=True)
    ## Load Preprocessing Information
    print("[Loading Preprocessing Resources]")
    prepare_params = joblib.load(args.preprocessing); _ = prepare_params["params"].pop("tokenize", None)
    ## Format Note DataFrame
    print("[Preparing Notes for Modeling]")
    note_df = model_utils.prepare_model_data(dataset=note_df,
                                             tokenize=True,
                                             tokenizer=prepare_params["tokenizer"],
                                             phrasers=prepare_params["phrasers"],
                                             return_cache=False,
                                             **prepare_params["params"])
    ## Load Tokenizer and Classification Model
    print("[Loading Feature Extractor]")
    preprocessor = joblib.load(f"{args.model}/preprocessor.joblib")
    ## Load Model and Targets
    print("[Loading Classifier and Targets]")
    model = joblib.load(f"{args.model}/classifier.joblib")
    with open(f"{args.model}/targets.txt","r") as the_file:
        targets = the_file.read().split("\n")
    ## Note Groups
    print("[Establishing Note Processing Batches]")
    note_index_chunks = list(util.chunks(note_df.index.tolist(), args.batch_size)) if args.batch_size is not None else [note_df.index.tolist()]
    ## Iterate Through Chunks
    predictions = []
    for note_indices in tqdm(note_index_chunks, desc="[Making Predictions]", total=len(note_index_chunks)):
        ## Get Subset
        note_df_subset = note_df.loc[note_indices].copy()
        ## Build Base Feature Set
        X, features = preprocessor.build_base_feature_set_independent(annotations=note_df_subset)
        ## Transform the Base Feature Set
        if not isinstance(model, dummy.DummyClassifier):
            X, features = preprocessor.transform_base_feature_set_independent(X_out=X, features=features)
        ## Make Predictions
        predictions.append(model.predict_proba(X))
    ## Format All Predictions
    print("[Formatting Predictions]")
    predictions = pd.DataFrame(np.vstack(predictions), columns=targets)
    predictions["predicted_label"] = predictions.idxmax(axis=1)
    ## Merge Predictions and Metadata
    print("[Merging Results]")
    note_df = pd.concat([note_df, predictions], axis=1)
    ## Output
    print("[Caching Results]")
    if not args.cache_note_text:
        note_df = note_df.drop("note_text",axis=1)
    _ = note_df.to_csv(f"{args.output_dir}/{args.keyword_category}.predictions.csv",index=False)
    print("[Script Complete]")

####################
### Execution
####################

if __name__ == "__main__":
    _ = main()