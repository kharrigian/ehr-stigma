
"""
Apply a trained BERT classifier to a set of raw notes
"""

#########################
### Imports
#########################

## Standard Library
import os
import json
import argparse

## External Libraries
import torch
import pandas as pd
from transformers import AutoTokenizer

## Private
from stigma import settings
from stigma import util
from stigma.model import bert as model_bert

#########################
### Functions
#########################

def parse_command_line():
    """
    
    """
    ## Initialize Parser and Execute
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--model", type=str, default=None, help="Path to trained BERT classifier")
    _ = parser.add_argument("--tokenizer", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Path/name of relevant BERT tokenizer.")
    _ = parser.add_argument("--note_file", type=str, default=None, help="Path to processed note file to use as an input to the model.")
    _ = parser.add_argument("--annotations", type=str, choices={"mimic-iv-discharge"}, help="Load annotations instead of using a note file.")
    _ = parser.add_argument("--keyword_category", type=str, default=None, choices={"adamant","compliance","other"})
    _ = parser.add_argument("--output_dir", type=str, default=None, help="Name of the directory to use for storing predictions.")
    _ = parser.add_argument("--rm_existing", action="store_true", default=False, help="If included, will overwrite any existing prediction file.")
    _ = parser.add_argument("--batch_size", type=int, default=16, help="Size of batch to use for prediction.")
    _ = parser.add_argument("--n_samples", type=int, default=None, help="If desired, downsample the note data.")
    _ = parser.add_argument("--n_samples_per_keyword", type=int, default=None, help="If desired, downsample the note data to a maximum number of samples per keyword.")
    _ = parser.add_argument("--random_state", type=int, default=42, help="If downsampling, use this random seed.")
    _ = parser.add_argument("--cache_note_text", action="store_true", default=False)
    _ = parser.add_argument("--device", default="cpu", type=str, help="Choose the Device of CPU/CUDA") 
    args = parser.parse_args()
    ## Check Arguments
    if args.model is None:
        raise ValueError("Must provide a model path (--model)")
    if args.tokenizer is None:
        raise ValueError("Must provide a tokenizer (--tokenizer)")
    if args.annotations is None and (args.note_file is None or not os.path.exists(args.note_file)):
        raise FileNotFoundError("Must provide a valid note file (--note_file)")
    if args.output_dir is None:
        raise ValueError("Must provide an output directory (--output_dir)")
    if args.n_samples is not None and args.n_samples_per_keyword is not None:
        raise ValueError("Cannot specify both --n_samples and --n_samples_per_keyword.")
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    if os.path.exists(f"{args.output_dir}/{args.keyword_category}.predictions.csv") and not args.rm_existing:
        raise ValueError(f"Must include --rm_existing flag to overwrite existing set of predictions for category: {args.keyword_category}")
    return args

def _predict(note_df,
             args):
    """
    
    """
    ## Load Initialization Parameters
    init_param_file = f"{args.model}/init.pth"
    init_params = torch.load(init_param_file)
    ## Task ID Info
    task_id = {y:x for x, y in enumerate(init_params["task_targets"])}.get(args.keyword_category)
    targets = sorted(init_params["task_targets"][args.keyword_category], key=lambda x: init_params["task_targets"][args.keyword_category][x])
    ## Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ## Tokenize Data
    tokens, token_masks = model_bert.tokenize_and_mask(text=note_df["note_text"].tolist(),
                                                       keywords=note_df["keyword"].tolist(),
                                                       tokenizer=tokenizer,
                                                       mask_type=init_params["mask_type"] if "mask_type" in init_params else "keyword_all" if init_params["include_all"] else "keyword")
    ## Initialize Token Dataset
    dataset = model_bert.ClassificationTokenDataset(tokens=tokens,
                                                    token_masks=token_masks,
                                                    labels=[-1 for _ in tokens],
                                                    task_ids=[task_id for _ in tokens],
                                                    device=args.device)
    ## Dataset Encoding
    if init_params["classifier_only"]:
        encoder = model_bert.BERTEncoder(checkpoint=init_params["checkpoint"],
                                         pool=True,
                                         use_bert_pooler=False if "use_bert_pooler" not in init_params else init_params["use_bert_pooler"],
                                         random_state=init_params["random_state"]).to(args.device)
        dataset = model_bert.encode_dataset(dataset=dataset,
                                            bert=encoder,
                                            batch_size=128,
                                            device=args.device)
    ## Initialize Model and Weights
    if init_params["classifier_only"]:
        model = model_bert.MultitaskClassifier(task_targets=init_params["task_targets"],
                                               in_dim=768,
                                               p_dropout=init_params["p_dropout"],
                                               random_state=init_params["random_state"]).to(args.device)
    else:
        model = model_bert.BERTMultitaskClassifier(task_targets=init_params["task_targets"],
                                                   checkpoint=init_params["checkpoint"],
                                                   p_dropout=init_params["p_dropout"],
                                                   use_bert_pooler=False if "use_bert_pooler" not in init_params else init_params["use_bert_pooler"],
                                                   random_state=init_params["random_state"]).to(args.device)
    _ = model.load_state_dict(torch.load(f"{args.model}/model.pth", map_location=torch.device('cpu')))
    ## 
    _, predictions = model_bert.classification_evaluate_model(model=model,
                                                              dataset=dataset,
                                                              n_tasks=len(init_params["task_targets"]),
                                                              loss_fcn=None,
                                                              is_token=not init_params["classifier_only"],
                                                              batch_size=init_params["settings"]["eval_batch_size"],
                                                              verbose=not init_params["classifier_only"],
                                                              eval_id="Application Set",
                                                              score=False,
                                                              device=args.device)
    ## Which Prediction Set
    predictions = predictions[task_id].to("cpu").numpy()
    return predictions, targets

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
            note_df = util.load_annotations_mimic_iv_discharge(annotation_file=f"{settings.DEFAULT_ANNOTATIONS_DIR}/annotations.augmented.csv")
        else:
            raise ValueError(f"Dataset not recognized (--dataset {args.annotations})")
    ## Isolate Keyword Category
    print("[Isolating Keyword Category]")
    note_df = note_df.loc[note_df["keyword"].map(lambda i: settings.KEY2CAT[i] == args.keyword_category)].reset_index(drop=True)
    ## Downsample
    if args.n_samples is not None:
        print("[Downsampling Notes]")
        note_df = note_df.sample(n=min(args.n_samples, note_df.shape[0]), random_state=args.random_state, replace=False).sort_index().reset_index(drop=True)
    if args.n_samples_per_keyword is not None:
        print("[Downsample Notes (Keyword Specific Limits)]")
        keywords = note_df["keyword"].unique()
        note_df_sampled = []
        for word in keywords:
            word_note_df = note_df.loc[note_df["keyword"]==word]
            word_note_df = word_note_df.sample(n=min(args.n_samples_per_keyword, word_note_df.shape[0]), random_state=args.random_state, replace=False)
            note_df_sampled.append(word_note_df)
        note_df_sampled = pd.concat(note_df_sampled).sort_index().reset_index(drop=True)
        note_df = note_df_sampled
    ## Make Predictions
    predictions, targets = _predict(note_df=note_df, args=args)
    ## Format Predictions
    print("[Formatting Predictions]")
    predictions = pd.DataFrame(predictions, columns=targets)
    predictions["predicted_label"] = predictions.idxmax(axis=1)
    ## Merge Predictions and Metadata
    print("[Merging Results]")
    note_df = pd.concat([note_df, predictions], axis=1)
    ## Drop Note Text
    if not args.cache_note_text:
        print("[Dropping Note Text From Output]")
        note_df = note_df.drop("note_text", axis=1)
    ## Output
    print("[Caching Results]")
    _ = note_df.to_csv(f"{args.output_dir}/{args.keyword_category}.predictions.csv",index=False)
    print("[Script Complete]")

############################
### Execute
############################

if __name__ == "__main__":
    _ = main()