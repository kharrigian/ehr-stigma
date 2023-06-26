
"""
Fine-tune a BERT model using Classification Labels
"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import argparse
from uuid import uuid4

## External Libraries
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from transformers import AutoTokenizer

## Private
from stigma import util, settings
from stigma.model import bert as model_bert
from stigma.model import util as model_utils

## Local Files
sys.path.append(os.path.dirname(__file__))
import util as train_utils

#######################
### Functions
#######################

def parse_command_line():
    """
    
    """
    ## Create Parser
    parser = argparse.ArgumentParser()
    ## Add Arguments
    _ = parser.add_argument("--dataset",
                            type=str,
                            default="mimic-iv-discharge",
                            choices={"mimic-iv-discharge"})
    _ = parser.add_argument("--model_settings",
                            type=str,
                            default="./configs/model/train/bert-full.default.json",
                            help="Model training / architecture parameters.")
    _ = parser.add_argument("--model",
                            type=str,
                            default="emilyalsentzer/Bio_ClinicalBERT",
                            help="Path to model/checkpoint.")
    _ = parser.add_argument("--tokenizer",
                            type=str,
                            default="emilyalsentzer/Bio_ClinicalBERT",
                            help="Path to tokenizer for the model.")
    _ = parser.add_argument("--eval_rm_keywords",
                            type=lambda s: [i.strip() for i in s.split(',')],
                            default=None,
                            help="If included, remove examples for keywords included in this list. Whole argument quoted. Comma delimited.")
    _ = parser.add_argument("--eval_original_context",
                            type=int,
                            default=10,
                            help="Original context size (number of tokens left and right). Shouldn't change unless major change to underlying annotation data.")
    _ = parser.add_argument("--eval_target_context",
                            type=int,
                            default=10,
                            help="Number of tokens to include to the left of the keyword.")
    _ = parser.add_argument("--eval_cv",
                            default=None,
                            nargs="*",
                            type=float,
                            help="Cross validation setup. Either 1 parameter (int) or 2 (int, float).")
    _ = parser.add_argument("--eval_random_state",
                            default=42,
                            type=int,
                            help="Evaluation random seed.")
    _ = parser.add_argument("--eval_test",
                            action="store_true",
                            default=False,
                            help="If included, evaluate baselines on the test split.")
    _ = parser.add_argument("--cache_errors",
                            action="store_true",
                            default=False,
                            help="If included, cache model errors to disk.")
    _ = parser.add_argument("--output_dir",
                            default="./data/raw/kharrigian/results/model-training/bert/",
                            type=str,
                            help="Where to store plots.")
    _ = parser.add_argument("--model_cache_dir",
                            default="./data/raw/kharrigian/models/classifiers/bert/")
    _ = parser.add_argument("--rm_existing",
                            action="store_true",
                            default=False,
                            help="If included and output directory exists, remove it.")
    _ = parser.add_argument("--rm_models",
                            action="store_true",
                            default=False,
                            help="If included, remove models during cross validation.")
    _ = parser.add_argument("--keyword_groups",
                            type=str,
                            default=None,
                            nargs="+",
                            help="Which keyword categories to train together, For example, 'adamant compliance-other' or 'adamant-compliance-other'")
    _ = parser.add_argument("--device",
                            default="cpu",
                            type=str,
                            help="Choose the Device of CPU/CUDA")    
    ## Parse
    args = parser.parse_args()
    ## Format Cross Validation Arguments
    if args.eval_cv is not None:
        if len(args.eval_cv) == 1:
            args.eval_cv = int(args.eval_cv[0])
        elif len(args.eval_cv) == 2:
            args.eval_cv = (int(args.eval_cv[0]), args.eval_cv[1])
        else:
            raise ValueError("Number of --eval_cv arguments not recognized.")
    ## Format Context Arguments
    if args.eval_target_context == -1:
        args.eval_target_context = None
    if args.eval_original_context == -1:
        args.eval_original_context = None
    ## Check Model Settings
    if args.model_settings is None or not os.path.exists(args.model_settings):
        raise FileNotFoundError("Could not find --model_settings.")
    return args

def load_dataset(dataset_id,
                 original_context=10,
                 target_context=10,
                 rm_keywords=None):
    """
    Load the clinical dataset
    """
    ## Load Model Data
    annotations = model_utils.prepare_model_data(dataset=dataset_id,
                                                 eval_original_context=original_context,
                                                 eval_target_context=target_context,
                                                 eval_rm_keywords=rm_keywords,
                                                 return_cache=False)
    ## Isolate Relevant Keyword Categories
    print("[Isolating Appropriate Keyword Categories]")
    unique_keycats = list(settings.CAT2KEYS.keys())
    ## Get Raw Data
    print("[Grouping Data By Keyword Category]")
    dataset = {}
    targets = {}
    if dataset_id.startswith("mimic-iv"):
        meta_cols = list(settings.MIMIC_IV_PERFORMANCE_GROUPS.keys())
    else:
        meta_cols = None
    if meta_cols is None:
        raise ValueError("Dataset not recognized.")
    missing_meta = [m for m in meta_cols if m not in annotations.columns]
    meta_cols = [m for m in meta_cols if m in annotations.columns]
    if len(missing_meta) > 0:
        print(">> WARNING: Some of the expected group metadata attributes are missing. This may be expected.")
        print(">> WARNING: The following metadata columns are missing: {}".format(missing_meta))
    for keycat in unique_keycats:
        ## Subset
        k_subset = annotations.loc[(annotations["keyword_category"]==keycat)]
        k_subset = k_subset.dropna(subset=["label"])
        ## Label Map
        lbls = k_subset["label"].value_counts().sort_values().index.tolist()
        targets[keycat] = {"labels":lbls,"name":"label"}
        ## Get Subsets
        dataset[keycat] = {}
        for split in ["train","dev","test"]:
            ## Get DF
            ks_subset = k_subset.loc[(k_subset["split"]==split)]
            ## Extract Relevant Data
            ks_text = ks_subset["note_text"].tolist()
            ks_keywords = ks_subset["keyword"].tolist()
            ks_labels = list(model_utils.encode_targets(ks_subset["label"], lbls))
            ks_meta = ks_subset[meta_cols].reset_index(drop=True)
            source_cols = ["enterprise_mrn","encounter_id","encounter_note_id","encounter_type","keyword_category","keyword","note_text"]
            if dataset_id.startswith("mimic"):
                source_cols = source_cols + ["start","end"]
            ks_source = ks_subset.loc[:, source_cols].copy()
            ## Cache
            dataset[keycat][split] = {
                "text":ks_text,
                "labels":np.array(ks_labels),
                "keywords":ks_keywords,
                "metadata":ks_meta,
                "source":ks_source
            }
    ## Return
    return dataset, targets, annotations

def get_keyword_groups(annotations,
                       keyword_groups):
    """
    
    """
    ## Case 0: Default is No Groupings (E.g., every keyword category is a separate task)
    if keyword_groups is None:
        ## Format Groups
        keyword_groups = {i:[i] for i in annotations["keyword_category"].unique()}
    ## Case 1: Groupings
    else:
        ## Format Groups
        keyword_groups = {i:i.split("-") for i in keyword_groups}
        ## Validate Groups
        for x, y in keyword_groups.items():
            if any(i not in annotations["keyword_category"].values for i in y):
                raise KeyError(f"One of the task_group ({x}) attributes was not found.")
    ## Ensure Unique and Mututally Exclusive
    seen = set()
    for x, y in keyword_groups.items():
        for i in y:
            if i in seen:
                raise ValueError("Non-unique keyword groups")
            seen.add(i)
    ## Return
    return keyword_groups

def _run_bert(dataset_id,
              model_path,
              fold_data,
              model_settings,
              eval_test=False,
              random_state=42,
              run_id=None,
              model_cache_dir=None,
              rm_final_model=False,
              device="cpu"):
    """
    
    """
    ## Model Caching
    run_id = str(uuid4()) if run_id is None else run_id
    model_cache_dir = "./data/models/classifiers/" if model_cache_dir is None else model_cache_dir
    model_dir = f"{model_cache_dir}/{run_id}/"
    ## Initalize Model Directory
    if not os.path.exists(model_dir):
        _ = os.makedirs(model_dir)
    ## Develop A Task/Label Encoding
    unique_task_ids = sorted(set(fold_data["train"]["task_ids"]))
    task_2_id = {task:i for i, task in enumerate(unique_task_ids)}
    task_class_2_id = {t:{} for t in unique_task_ids}
    for tid, lbl in zip(fold_data["train"]["task_ids"], fold_data["train"]["labels"]):
        if lbl not in task_class_2_id[tid]:
            task_class_2_id[tid][lbl] = len(task_class_2_id[tid])
    id_2_task = sorted(task_2_id.keys(), key=lambda i: task_2_id.get(i))
    id_2_task_class = [sorted(task_class_2_id[tid], key=lambda x: task_class_2_id[tid].get(x)) for tid in id_2_task]
    ## Apply Encoding
    for split, split_data in fold_data.items():
        if split == "test" and not eval_test:
            split_data["task_ids_encoded"] = None
            split_data["labels_encoded"] = None
        else:
            split_data["task_ids_encoded"] = [task_2_id[tid] for tid in split_data["task_ids"]]
            split_data["labels_encoded"] = [task_class_2_id[tid][lbl] for tid, lbl in zip(split_data["task_ids"],split_data["labels"])]
    ## Save Target Ordering
    with open(f"{model_dir}/targets.json","w") as the_file:
        json.dump({"task_2_id":task_2_id, "task_class_2_id":task_class_2_id}, the_file, indent=2)
    ## Create Datasets
    train_dataset = model_bert.ClassificationTokenDataset(tokens=fold_data["train"]["tokens"],
                                                          token_masks=fold_data["train"]["token_masks"],
                                                          labels=fold_data["train"]["labels_encoded"],
                                                          task_ids=fold_data["train"]["task_ids_encoded"],
                                                          device=device)
    dev_dataset = model_bert.ClassificationTokenDataset(tokens=fold_data["dev"]["tokens"],
                                                        token_masks=fold_data["dev"]["token_masks"],
                                                        labels=fold_data["dev"]["labels_encoded"],
                                                        task_ids=fold_data["dev"]["task_ids_encoded"],
                                                        device=device)
    test_dataset = model_bert.ClassificationTokenDataset(tokens=fold_data["test"]["tokens"] if eval_test else None, 
                                                         token_masks=fold_data["test"]["token_masks"] if eval_test else None,
                                                         labels=fold_data["test"]["labels_encoded"] if eval_test else None,
                                                         task_ids=fold_data["test"]["task_ids_encoded"] if eval_test else None,
                                                         device=device)
    ## Initialze Model (And Encode if Necessary)
    if model_settings["classifier_only"]:
        ## Initialize Encoder
        bert = model_bert.BERTEncoder(checkpoint=model_path,
                                      random_state=random_state,
                                      pool=True,
                                      use_bert_pooler=model_settings["use_bert_pooler"]).to(device)
        ## Run Encoder to Build New Classification Datasets
        train_dataset = model_bert.encode_dataset(dataset=train_dataset,
                                                  bert=bert,
                                                  batch_size=128,
                                                  device=device)
        dev_dataset = model_bert.encode_dataset(dataset=dev_dataset,
                                                bert=bert,
                                                batch_size=128,
                                                device=device)
        if eval_test:
            test_dataset = model_bert.encode_dataset(dataset=test_dataset,
                                                     bert=bert,
                                                     batch_size=128,
                                                     device=device)
        ## Initialize Classifier
        model = model_bert.MultitaskClassifier(task_targets=task_class_2_id,
                                               in_dim=768,
                                               p_dropout=model_settings["p_dropout"],
                                               random_state=random_state).to(device)
    else:
        ## Initialize Combined Encoder and Classifier
        model = model_bert.BERTMultitaskClassifier(task_targets=task_class_2_id,
                                                   checkpoint=model_path,
                                                   p_dropout=model_settings["p_dropout"],
                                                   use_bert_pooler=model_settings["use_bert_pooler"],
                                                   random_state=random_state).to(device)
    ## Initialize Task-Specific Loss Functions
    task_weights = model_bert.get_loss_class_weights(train_dataset,
                                                     task_targets=task_class_2_id,
                                                     balance=model_settings["class_balance"],
                                                     device=device)
    loss_fcn = {}
    for task_id, weights in task_weights.items():
        loss_fcn[task_id] = torch.nn.CrossEntropyLoss(weight=weights,
                                                      reduction="mean") 
    ## Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=model_settings["learning_rate"],
                                  weight_decay=model_settings["weight_decay"])
    ## Random Initialization
    _ = torch.manual_seed(random_state)
    if torch.cuda.is_available():
        rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
    else:
        rng = torch.get_rng_state()
        torch.set_rng_state(rng)
    shuffle_seed = np.random.RandomState(random_state)
    ## Initialize Training Batch Index
    training_batch_index = list(range(len(train_dataset)))
    ## Task Weights
    tids, tcounts = torch.unique(torch.stack([i["task_id"] for i in train_dataset]).squeeze(), sorted=True, return_counts=True)
    if model_settings["balance_task_weights"]:
        task_weights = (tcounts / tcounts.sum()) ** -1
    else:
        task_weights = torch.ones(len(tids))
    ## Helpful Parameters
    n_tasks = len(task_weights)
    n_epochs = model_settings["num_train_epochs"]
    max_n_updates = model_settings["max_steps"]
    ## Performance Cache
    loss_cache = []
    performance_cache = []
    running_losses = torch.zeros_like(task_weights)
    running_instances = torch.zeros_like(task_weights)
    best_model = {t:{"epoch":None, "batch":None, "steps":None, "score":None, "metric":model_settings["metric_for_best_model"]} for t in range(n_tasks)}
    best_model_static = {t:0 for t in range(n_tasks)}
    ## Training Loop
    n_updates = 0
    is_last_update = False
    model_plateau = False
    print(">> Beginning Training Loop")
    for epoch in range(n_epochs):
        ## Check Steps
        if max_n_updates != -1 and n_updates >= max_n_updates:
            print(">> Maximum steps reached.")
            break
        if model_plateau:
            print(">> Early stopping criteria reached.")
            break
        ## Shuffle Training Indices and Get Batches
        _ = shuffle_seed.shuffle(training_batch_index)
        if model_settings["train_batch_size"] is None:
            training_batches = [training_batch_index]
        else:
            training_batches = list(util.chunks(training_batch_index, model_settings["train_batch_size"]))
        n_batches = len(training_batches)
        ## Iterate Over Mini-Batches
        for b, batch_ind in enumerate(training_batches):
            ## Check Steps
            if max_n_updates != -1 and n_updates >= max_n_updates:
                break
            ## Check Early Stopping
            if model_plateau:
                break
            ## Collate
            batch_train_data = model_bert.classification_collate_batch([train_dataset[ind] for ind in batch_ind],
                                                                       is_token=not model_settings["classifier_only"],
                                                                       device=device)
            ## Reset Gradients
            _ = optimizer.zero_grad()
            ## Forward Pass
            batch_out = model(batch_train_data)
            ## Compute Loss For Each Task
            batch_losses = torch.zeros_like(task_weights)
            batch_instances = torch.zeros_like(task_weights)
            for tid, (tlogits, tmask) in enumerate(zip(*batch_out)):
                ## No Task Examples in This Batch
                tn = tmask.sum()
                if tn == 0:
                    continue
                ## Task Instance
                batch_instances[tid] = tn
                ## Compute and Cache Task Loss
                tloss = loss_fcn[tid](tlogits[tmask], batch_train_data["label"][tmask])
                batch_losses[tid] = tloss
            ## Aggregate Task Loss
            total_batch_loss = batch_losses @ task_weights
            ## Backprop and Update
            total_batch_loss.backward()
            optimizer.step()
            ## Increase Updates
            n_updates += 1
            ## Check Update
            if max_n_updates != -1 and n_updates == max_n_updates:
                is_last_update = True
            if epoch == n_epochs - 1 and b == len(training_batches) - 1:
                is_last_update = True
            ## Track Cumulative Loss for the Epoch
            batch_losses = batch_losses.detach()
            batch_instances = batch_instances.detach()
            total_batch_loss_agg = (batch_losses * batch_instances)
            running_losses += total_batch_loss_agg
            running_instances += batch_instances
            ## Train Logging
            avg_train_loss = (running_losses / running_instances).to("cpu")
            if not model_settings["classifier_only"]:
                print(f"Epoch {epoch+1}/{n_epochs} | Batch {b+1}/{n_batches} | Average Training Loss:", avg_train_loss)
            ## Model Evaluation
            if n_updates == 1 or n_updates % model_settings["cache_frequency"] == 0 or is_last_update:
                ## Update User
                print(f">> Running Evaluation. {n_updates} Updates Completed.")
                ## Put Model in Evaluation Mode
                model.eval()
                ## Run
                train_performance = {}
                if model_settings["cache_training"]:
                    train_performance, _ = model_bert.classification_evaluate_model(model=model,
                                                                                    dataset=train_dataset,
                                                                                    n_tasks=n_tasks,
                                                                                    loss_fcn=loss_fcn,
                                                                                    batch_size=model_settings["eval_batch_size"],
                                                                                    is_token=not model_settings["classifier_only"],
                                                                                    verbose=not model_settings["classifier_only"],
                                                                                    eval_id="Training",
                                                                                    score=True,
                                                                                    device=device)
                validation_performance, _ = model_bert.classification_evaluate_model(model=model,
                                                                                     dataset=dev_dataset,
                                                                                     n_tasks=n_tasks,
                                                                                     loss_fcn=loss_fcn,
                                                                                     is_token=not model_settings["classifier_only"],
                                                                                     batch_size=model_settings["eval_batch_size"],
                                                                                     verbose=not model_settings["classifier_only"],
                                                                                     eval_id="Validation",
                                                                                     score=True,
                                                                                     device=device)
                ## Print and Cache Outcomes
                for spl_p, spl, spl_perf in zip(["Training","Validation"],
                                                ["train","dev"],
                                                [train_performance,validation_performance]):
                    ## Skip Unevaluated
                    if spl_perf is None or len(spl_perf) == 0:
                        continue
                    ## Print
                    for task, task_perf in spl_perf.items():
                        print(f">> [{spl_p}] Task ID {task}: {task_perf}")
                    ## Metadata
                    spl_perf["epoch"] = epoch
                    spl_perf["step"] = n_updates
                    spl_perf["split"] = spl
                    ## Cache
                    performance_cache.append(spl_perf)
                ## Cache Losses (Average Since Last Evaluation or Performance Result)
                loss_cache.append({
                    "split":"train",
                    "loss":[float(i) for i in avg_train_loss.numpy()] if not model_settings["cache_training"] else [float(train_performance[i]["loss"]) for i in range(n_tasks)],
                    "epoch":epoch + (b + 1) / n_batches,
                    "steps":n_updates
                })
                loss_cache.append({
                    "split":"dev",
                    "loss":[float(validation_performance[i]["loss"]) for i in range(n_tasks)],
                    "epoch":epoch + (b+1) / n_batches,
                    "steps":n_updates
                })
                ## Reset Loss Accumulation
                running_losses = torch.zeros_like(task_weights)
                running_instances = torch.zeros_like(task_weights)
                ## Model Performance Check (Caching Best Model)
                cur_best_models = {x:y["steps"] for x, y in best_model.items()}
                eval_best_models = {}
                found_new_best = set()
                for task_id in range(n_tasks):
                    task_perf = validation_performance[task_id]
                    if best_model[task_id]["score"] is None:
                        found_new_best.add(task_id)
                        eval_best_models[task_id] = n_updates
                        best_model[task_id]["score"] = task_perf[model_settings["metric_for_best_model"]]
                        best_model[task_id]["epoch"] = epoch
                        best_model[task_id]["batch"] = b
                        best_model[task_id]["steps"] = n_updates
                        best_model_static[task_id] = 0
                    else:
                        improvement = False
                        if model_settings["greater_is_better"] and task_perf[model_settings["metric_for_best_model"]] > best_model[task_id]["score"] * (1 + model_settings["early_stop_tol"]):
                            improvement = True
                        elif not model_settings["greater_is_better"] and task_perf[model_settings["metric_for_best_model"]] < best_model[task_id]["score"] * (1 - model_settings["early_stop_tol"]):
                            improvement = True
                        if improvement:
                            found_new_best.add(task_id)
                            eval_best_models[task_id] = n_updates
                            best_model[task_id]["score"] = task_perf[model_settings["metric_for_best_model"]]
                            best_model[task_id]["epoch"] = epoch
                            best_model[task_id]["batch"] = b
                            best_model[task_id]["steps"] = n_updates
                            best_model_static[task_id] = 0
                        else:
                            eval_best_models[task_id] = cur_best_models[task_id]
                            best_model_static[task_id] += 1
                ## Cache New Best
                if len(found_new_best) > 0:
                    print(">> Found a new best model for tasks: {}".format(sorted(found_new_best)))
                    if not os.path.exists(f"{model_dir}/checkpoint-{n_updates}/"):
                        _ = os.makedirs(f"{model_dir}/checkpoint-{n_updates}/")
                    _ = torch.save(model.state_dict(), f"{model_dir}/checkpoint-{n_updates}/model.pth")
                    _ = torch.save({
                        "classifier_only":model_settings["classifier_only"],
                        "mask_type":model_settings["mask_type"],
                        "use_bert_pooler":model_settings["use_bert_pooler"],
                        "settings":model_settings,
                        "task_targets":model._task_targets,
                        "checkpoint":model_path,
                        "p_dropout":model._p_dropout,
                        "random_state":model._random_state,
                        "loss_history":loss_cache,
                        "performance_history":performance_cache
                    }, f"{model_dir}/checkpoint-{n_updates}/init.pth")
                else:
                    print(">> No significant improvement for any task.")
                ## Remove Outdated (Only Relevant If New Best Models Found)
                if len(found_new_best) > 0:
                    to_remove = [v for v in cur_best_models.values() if v not in eval_best_models.values()]
                    for tr in to_remove:
                        if tr is None:
                            continue
                        print(f">> Removing previous model checkpoint (checkpoint-{tr})")
                        _ = os.system(f"rm -rf {model_dir}/checkpoint-{tr}/")
                ## Check Early Stopping
                if all(v >= model_settings["early_stop_patience"] for v in best_model_static.values()):
                    model_plateau = True
                ## Return Model To Training Mode
                model.train()
    ## Training Complete
    print(">> Training Loop Complete")
    ## Free Model From Memory (Best Models Are Cached on Disk)
    del model
    ## Ensure model was actually updated
    if n_updates == 0:
        raise ValueError("No updates were run.")
    ## Save Information About Best Models (e.g., Which Checkpoint To Use For Each Task, How they were trained)
    with open(f"{model_dir}/best_model.json","w") as the_file:
        json.dump(best_model, the_file, indent=2)
    with open(f"{model_dir}/summary.json","w") as the_file:
        json.dump({"loss":loss_cache, "performance":performance_cache, "settings":model_settings}, the_file, indent=2)
    ## Make Predictions Using Best Model for Each 
    print(">> Beginning Model Evaluation")
    ## Prediction Cache
    y_pred = {task_id:{"train":None,"dev":None,"test":None} for task_id in range(n_tasks)}
    ## Map Between Optimal Model Checkpoints and Tasks
    task2best = {}
    for task_id, task_best in best_model.items():
        if task_best["steps"] not in task2best:
            task2best[task_best["steps"]] = []
        task2best[task_best["steps"]].append(task_id)
    ## Iterate Over Optimal Models
    for task_id_opt, opt_tasks in task2best.items():
        ## Identify Optimal Model and Load
        print(f"Loading best model for Task ID(s) {opt_tasks}: checkpoint-{task_id_opt}")
        if model_settings["classifier_only"]:
            model = model_bert.MultitaskClassifier(task_targets=task_class_2_id,
                                                   in_dim=768,
                                                   p_dropout=model_settings["p_dropout"],
                                                   random_state=random_state).to(device)
        else:
            model = model_bert.BERTMultitaskClassifier(task_targets=task_class_2_id,
                                                       checkpoint=model_path,
                                                       use_bert_pooler=model_settings["use_bert_pooler"],
                                                       p_dropout=model_settings["p_dropout"],
                                                       random_state=random_state).to(device)
        _ = model.load_state_dict(torch.load(f"{model_dir}/checkpoint-{task_id_opt}/model.pth"))
        ## Put Model into Evaluation Mode
        model.eval()    
        ## Iterate Through Datasets to Generate Final Predictions
        for ds, dsname in zip([train_dataset, dev_dataset, test_dataset],["train","dev","test"]): 
            ## Skip Test Dataset if Not Doing Evaluation
            if dsname == "test" and not eval_test:
                continue
            ## Get Predictions
            _, ds_predictions = model_bert.classification_evaluate_model(model=model,
                                                                         dataset=ds,
                                                                         n_tasks=n_tasks,
                                                                         loss_fcn=None,
                                                                         is_token=not model_settings["classifier_only"],
                                                                         batch_size=model_settings["eval_batch_size"],
                                                                         verbose=not model_settings["classifier_only"],
                                                                         eval_id=dsname.title(),
                                                                         score=False,
                                                                         device=device)
            ## Store Predictions If They Are Relevant For The Model
            for tid, tpred in enumerate(ds_predictions):
                ## Skip If Not Aligned
                if tid not in opt_tasks:
                    continue
                ## Cache Numpy Softmax Probabilities
                y_pred[tid][dsname] = tpred.to("cpu").detach().numpy()
    ## Scoring Cache
    scores = []
    errors = []
    group_scores = []
    ## Iterate Over Tasks (i.e., Keyword Categories)
    for tid in range(n_tasks):
        ## Task Information
        tid_name = id_2_task[tid]
        tid_tar_labels = id_2_task_class[tid]
        tid_tar_labels_int = list(range(len(tid_tar_labels)))
        task_train_size = None
        ## Iterate Over Splits
        for _tid, _y_true, _y_pred, _split in zip([train_dataset._task_ids, dev_dataset._task_ids, test_dataset._task_ids],
                                                  [train_dataset._labels, dev_dataset._labels, test_dataset._labels],
                                                  [y_pred[tid]["train"], y_pred[tid]["dev"], y_pred[tid]["test"]],
                                                  ["train","dev","test"]):
            ## Skip Test Set if Not Doing Test Evaluation
            if _split == "test" and not eval_test:
                continue
            ## Get Task Mask
            _tid_mask = [i for i, x in enumerate(_tid) if x == tid]
            if _split == "train":
                task_train_size = len(_tid_mask)
            ## Get Important Data for Scoring
            tid_y_pred = _y_pred[_tid_mask]
            tid_y_pred_argmax = tid_y_pred.argmax(axis=1)
            tid_y_true = np.array([_y_true[i] for i in _tid_mask])
            tid_source = fold_data[_split]["source"].iloc[_tid_mask]
            tid_meta = fold_data[_split]["metadata"].iloc[_tid_mask]
            tid_target_name = [fold_data[_split]["labels_name"][i] for i in _tid_mask][0]
            ## Get Errors
            eind = (tid_y_pred_argmax != tid_y_true).nonzero()[0]
            e_split_source = tid_source.iloc[eind].copy()
            ## Add Information
            e_split_source["y_true"] = [tid_tar_labels[i] for i in tid_y_true[eind]]
            e_split_source["y_pred"] = [tid_tar_labels[i] for i in tid_y_pred_argmax[eind]]
            e_split_source["split"] = _split
            e_split_source["model_type"] = "bert"
            e_split_source["feature_set"] = "bert"
            e_split_source["keyword_category"] = tid_name
            ## Cache Errors
            errors.append(e_split_source)
            ## Iterate Over Interest Groups
            for col in tid_meta.columns:
                ## Re-Cache Overall Performance
                group_scores.append({"support":len(tid_y_true),
                                     "accuracy":metrics.accuracy_score(tid_y_true, tid_y_pred_argmax),
                                     "f1":metrics.f1_score(tid_y_true, tid_y_pred_argmax, average="macro", labels=tid_tar_labels_int, zero_division=0),
                                     "precision":metrics.precision_score(tid_y_true, tid_y_pred_argmax, average="macro", labels=tid_tar_labels_int, zero_division=0),
                                     "recall":metrics.recall_score(tid_y_true, tid_y_pred_argmax, average="macro", labels=tid_tar_labels_int, zero_division=0),
                                     "class_balance":{tid_tar_labels[lbl]:(tid_y_true == lbl).sum() for lbl in tid_tar_labels_int},
                                     "model_type":"bert",
                                     "feature_set":"bert",
                                     "split":_split,
                                     "learning_curve_train_proportion":100,
                                     "learning_curve_train_iteration":0,
                                     "learning_curve_train_size":task_train_size,
                                     "group_category":col,
                                     "group_subcategory":"overall",
                                     "target":tid_target_name,
                                     "keyword_category":tid_name})
                ## Compute Group-level Performance
                if dataset_id.startswith("mimic-iv"):
                    col_unique = settings.MIMIC_IV_PERFORMANCE_GROUPS[col]
                else:
                    col_unique = []
                for colu in col_unique:
                    colu_ind = (tid_meta[col] == colu).values.nonzero()[0]
                    if len(colu_ind) != 0:
                        group_scores.append({"support":len(colu_ind),
                                             "accuracy":metrics.accuracy_score(tid_y_true[colu_ind], tid_y_pred_argmax[colu_ind]),
                                             "f1":metrics.f1_score(tid_y_true[colu_ind], tid_y_pred_argmax[colu_ind], average="macro", zero_division=0),
                                             "precision":metrics.precision_score(tid_y_true[colu_ind], tid_y_pred_argmax[colu_ind], average="macro", zero_division=0),
                                             "recall":metrics.recall_score(tid_y_true[colu_ind], tid_y_pred_argmax[colu_ind], average="macro", zero_division=0),
                                             "class_balance":{tid_tar_labels[lbl]:(tid_y_true[colu_ind] == lbl).sum() for lbl in tid_tar_labels_int},
                                             "model_type":"bert",
                                             "feature_set":"bert",
                                             "split":_split,
                                             "learning_curve_train_proportion":100,
                                             "learning_curve_train_iteration":0,
                                             "learning_curve_train_size":task_train_size,
                                             "group_category":col,
                                             "group_subcategory":colu,
                                             "target":tid_target_name,
                                             "keyword_category":tid_name})
            ## Overall Classification Report
            _cls_report = metrics.classification_report(tid_y_true, tid_y_pred_argmax, zero_division=0, output_dict=True, labels=tid_tar_labels_int)
            _cls_report = {tid_tar_labels[int(x)]:y for x, y in _cls_report.items() if x.isdigit()}
            ## Per Class ROC/AUC (One vs. Rest)
            for tl, tl_lbl in enumerate(tid_tar_labels):
                tl_fpr, tl_tpr, tl_thresholds = metrics.roc_curve(y_true=(np.array(tid_y_true) == tl).astype(int), y_score=tid_y_pred[:,tl], pos_label=1)
                tl_auc = metrics.auc(tl_fpr, tl_tpr)
                _cls_report[tl_lbl]["roc"] = {"fpr":tl_fpr,"tpr":tl_tpr,"thresholds":tl_thresholds,"auc":tl_auc}
            ## Confusion
            _cm = metrics.confusion_matrix(tid_y_true, tid_y_pred_argmax, labels=tid_tar_labels_int)
            _cm = pd.DataFrame(_cm,
                            index=[f"true_{l}" for l in tid_tar_labels],
                            columns=[f"pred_{l}" for l in tid_tar_labels])
            ## Overall
            _scores = {
                "model_type":"bert",
                "feature_set":"bert",
                "split":_split,
                "accuracy":metrics.accuracy_score(tid_y_true, tid_y_pred_argmax),
                "recall_macro":metrics.recall_score(tid_y_true, tid_y_pred_argmax, zero_division=0,average="macro"),
                "recall_micro":metrics.recall_score(tid_y_true, tid_y_pred_argmax, zero_division=0,average="micro"),
                "precision_macro":metrics.precision_score(tid_y_true, tid_y_pred_argmax, zero_division=0,average="macro"),
                "precision_micro":metrics.precision_score(tid_y_true, tid_y_pred_argmax, zero_division=0,average="micro"),
                "f1_macro":metrics.f1_score(tid_y_true, tid_y_pred_argmax, zero_division=0,average="macro"),
                "f1_micro":metrics.f1_score(tid_y_true, tid_y_pred_argmax, zero_division=0,average="micro"),
                "per_class":_cls_report,
                "confusion":_cm,
                "learning_curve_train_proportion":100,
                "learning_curve_train_iteration":0,
                "learning_curve_train_size":task_train_size,
                "target":tid_target_name,
                "keyword_category":tid_name,
            }
            ## Cache
            scores.append(_scores)
    print(">> Training Run Complete")
    ## Merge Errors
    print(">> Formatting Errors")
    errors = pd.concat(errors, axis=0).reset_index(drop=True)
    ## Remove Model
    if rm_final_model:
        print(">> Removing Model Directory")
        _ = os.system(f"rm -rf {model_dir}")
    ## Return
    return scores, group_scores, errors

def run_bert(dataset_id,
             dataset,
             targets,
             model_path,
             keyword_group,
             model_settings,
             cv=None,
             eval_test=False,
             random_state=42,
             run_prefix=None,
             rm_models=False,
             model_cache_dir=None,
             device="cpu"):
    """
    
    """
    ## Copy String
    dataset_id_str = dataset_id
    ## For consistency,first retrieve cross validation splits on each keyword category
    cv_splits = {}
    for dataset_id, dataset_data in dataset.items():
        cv_splits[dataset_id] = model_utils.get_cross_validation_splits(cv=cv,
                                                                        y_train=dataset_data["train"]["labels"],
                                                                        y_dev=dataset_data["dev"]["labels"],
                                                                        random_state=random_state)
    ## Reformat the Split Indices
    n_cv_fold = len(cv_splits[dataset_id][0])
    cv_splits_alt = [({},{}) for _ in range(n_cv_fold)]
    for i in range(n_cv_fold):
        for dataset_id in dataset.keys():
            cv_splits_alt[i][0][dataset_id] = cv_splits[dataset_id][0][i]
            cv_splits_alt[i][1][dataset_id] = cv_splits[dataset_id][1][i]
    ## Iterate Through Folds
    scores, group_scores, errors = [], [], []
    for fold, (train_, dev_) in enumerate(cv_splits_alt):
        print(f"[Beginning Fold {fold+1}/{n_cv_fold}: {keyword_group}]")
        ## Gather Desired Data for the Fold
        fold_data = {}
        for split in ["train","dev","test"]:
            if split == "test" and not eval_test:
                fold_data[split] = {"tokens":None, "token_masks":None, "labels":None, "labels_name":[], "metadata":None, "source":None, "task_ids":None}
            else:
                fold_data[split] = {"tokens":[], "token_masks":[], "labels":[], "labels_name":[], "metadata":[], "source":[], "task_ids":[]}
                for dataset_id, dataset_values in dataset.items():
                    ## Isolate Task Appropriate Indices
                    dtask_subset = dataset_values[split]["source"]["keyword_category"].isin(keyword_group).values.nonzero()[0]
                    ## See If Relevant
                    if len(dtask_subset) == 0:
                        continue
                    ## Combine Task Appropriate With Indices
                    dtask_ind = train_[dataset_id] if split == "train" else dev_[dataset_id] if split == "dev" else list(range(dataset_values["test"]["source"].shape[0]))
                    ## Filter
                    dtask_ind_subset = sorted(set(dtask_subset) & set(dtask_ind))
                    ## Check Length
                    if len(dtask_ind_subset) == 0:
                        continue
                    ## Update Fold Data
                    fold_data[split]["tokens"].extend([dataset_values[split]["tokens"][ind] for ind in dtask_ind_subset])
                    fold_data[split]["token_masks"].extend([dataset_values[split]["token_masks"][ind] for ind in dtask_ind_subset])
                    fold_data[split]["labels"].extend([targets[dataset_id]["labels"][dataset_values[split]["labels"][ind]] for ind in dtask_ind_subset])
                    fold_data[split]["labels_name"].extend([targets[dataset_id]["name"] for _ in dtask_ind_subset])
                    fold_data[split]["metadata"].append(dataset_values[split]["metadata"].iloc[dtask_ind_subset])
                    fold_data[split]["source"].append(dataset_values[split]["source"].iloc[dtask_ind_subset])
                    fold_data[split]["task_ids"].extend(dataset_values[split]["source"].iloc[dtask_ind_subset]["keyword_category"].tolist())
                ## Merge
                fold_data[split]["metadata"] = pd.concat(fold_data[split]["metadata"],axis=0)
                fold_data[split]["source"] = pd.concat(fold_data[split]["source"],axis=0)
        ## Run Procedure
        fold_scores, fold_group_scores, fold_errors = _run_bert(dataset_id=dataset_id_str,
                                                                model_path=model_path,
                                                                fold_data=fold_data,
                                                                eval_test=eval_test,
                                                                random_state=random_state,
                                                                model_settings=model_settings,
                                                                run_id=f"{run_prefix}_fold-{fold}" if run_prefix is not None else f"fold-{fold}",
                                                                rm_final_model=rm_models,
                                                                model_cache_dir=model_cache_dir,
                                                                device=device)
        ## Add Metadata
        for fs in fold_scores:
            fs["fold"] = fold
            fs["target"] = targets[fs["keyword_category"]]["name"]
        for fs in fold_group_scores:
            fs["fold"] = fold
            fs["target"] = targets[fs["keyword_category"]]["name"]
        fold_errors["fold"] = fold
        fold_errors["target"] = fold_errors["keyword_category"].map(lambda i: targets[i]["name"])
        ## Cache
        scores.extend(fold_scores)
        group_scores.extend(fold_group_scores)
        errors.append(fold_errors)
    ## Format
    errors = pd.concat(errors, axis=0).reset_index(drop=True)
    return scores, group_scores, errors

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Output Directory
    print(f"[Will Store Outputs in '{args.output_dir}']")
    if os.path.exists(args.output_dir) and args.rm_existing:
        _ = os.system(f"rm -rf {args.output_dir}")
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    ## Store Command Line Arguments
    print("[Caching Command Line Arguments]")
    with open(f"{args.output_dir}/cli.config.json","w") as the_file:
        json.dump(vars(args), the_file, indent=1)
    ## Load Model Settings
    print("[Loading Model Settings]")
    with open(args.model_settings,"r") as the_file:
        model_settings = json.load(the_file)
    ## Load
    print("[Loading Dataset]")
    dataset, targets, annotations = load_dataset(dataset_id=args.dataset,
                                                 original_context=args.eval_original_context,
                                                 target_context=args.eval_target_context,
                                                 rm_keywords=args.eval_rm_keywords)
    ## Identify Tasks To Run Together/Separately
    print("[Gathering Task Groupings]")
    keyword_groups = get_keyword_groups(annotations=annotations,
                                        keyword_groups=args.keyword_groups)
    ## Load Tokenizer
    print("[Loading Transformer Tokenizer]")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ## Apply Tokenization
    print("[Applying Transformer Tokenizer]")
    if model_settings["use_bert_pooler"]:
        print(">> Warning: Current settings have use_bert_pooler set to True. Token masks will be irrelevant.")
    for dataset_id, dataset_items in dataset.items():
        ## Iterate Through Splits
        for _, split_items in dataset_items.items():
            split_items["tokens"], split_items["token_masks"] = model_bert.tokenize_and_mask(text=split_items["text"],
                                                                                             keywords=split_items["keywords"],
                                                                                             tokenizer=tokenizer,
                                                                                             mask_type=model_settings["mask_type"])
        ## Cache Updated Tokens
        dataset[dataset_id] = dataset_items
    ## Iterate Through Task Groups
    print("[Running Classification Procedure]")
    scores, group_scores, errors = {}, {}, {}
    for keyword_group_name, keyword_group in keyword_groups.items():
        ## Run Experiment For the Group of Keyword Categories
        task_scores, task_group_scores, task_errors = run_bert(dataset_id=args.dataset,
                                                               model_path=args.model,
                                                               dataset=dataset,
                                                               targets=targets,
                                                               keyword_group=keyword_group,
                                                               model_settings=model_settings,
                                                               cv=args.eval_cv,
                                                               eval_test=args.eval_test,
                                                               random_state=args.eval_random_state,
                                                               run_prefix=keyword_group_name,
                                                               rm_models=args.rm_models,
                                                               model_cache_dir=args.model_cache_dir,
                                                               device=args.device)
        ## Add Metadata
        for ks in task_scores:
            ks["keyword_group"] = keyword_group_name
        for ks in task_group_scores:
            ks["keyword_group"] = keyword_group_name
        task_errors["keyword_group"] = keyword_group_name
        ## Cache Results based on Keyword Category
        for keyword_cat in keyword_group:
            scores[keyword_cat] = list(filter(lambda i: i["keyword_category"]==keyword_cat, task_scores))
            group_scores[keyword_cat] = list(filter(lambda i: i["keyword_category"]==keyword_cat, task_group_scores))
            errors[keyword_cat] = task_errors.loc[task_errors["keyword_category"]==keyword_cat]
    ## Cache Errors
    if args.cache_errors:
        print("[Caching Model Errors]")
        _ = train_utils.cache_all_errors(errors,
                                         output_dir=args.output_dir)
    ## Cache Scores
    print("[Caching Performance Outcomes]")
    _ = train_utils.cache_scores(scores,
                                 output_dir=args.output_dir,
                                 model_settings=model_settings,
                                 is_bert=True)
    ## Plotting
    print("[Plotting Performance Outcomes]")
    _ = train_utils.plot_all_performance(scores=scores,
                                         learning_curves=None,
                                         output_dir=args.output_dir,
                                         model_settings=model_settings,
                                         is_bert=True)
    print("[Plotting Group-Based Performance Outcomes]")
    _ = train_utils.plot_all_performance_group(dataset_id=args.dataset,
                                               group_scores=group_scores,
                                               output_dir=args.output_dir,
                                               model_settings=model_settings,
                                               is_bert=True)
    print("[Script Complete]")

#######################
### Execution
#######################

## Execute the Program
if __name__ == "__main__":
    _ = main()
