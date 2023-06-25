
"""
Baseline model performance
"""

####################
### Imports
####################

## Standard Library
import os
import sys
import json
import argparse
import warnings

## External
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import dummy, linear_model, ensemble, metrics, model_selection

## Local
from stigma import settings
from stigma import text_utils
from stigma.model import util as model_utils
from stigma.model.baseline import ConditionalMajorityClassifier

## Filter Warnings (Main Purposes is Annoying Glyph Issues in Plots)
warnings.filterwarnings("ignore")

## Train Utilities
sys.path.append(os.path.dirname(__file__))
import util as train_utils

####################
### Functions
####################

def parse_command_line():
    """
    
    """
    ## Build Parser
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--dataset",
                            type=str,
                            default="mimic-iv-discharge",
                            choices={"mimic-iv-discharge"})
    _ = parser.add_argument("--model_settings",
                            type=str,
                            default="./configs/model/train/baseline-statistical.default.json",
                            help="Path to model suite settings.")
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
    _ = parser.add_argument("--eval_learning_curves",
                            default=None,
                            nargs="*",
                            type=int,
                            help="If single integer, number of linear bins. Otherwise, percetiles. Use in combination with --eval_learning_curves_iter")
    _ = parser.add_argument("--eval_learning_curves_iter",
                            default=10,
                            type=int,
                            help="Number of resampling iterations to execute at each learning curve sample size.")
    _ = parser.add_argument("--eval_random_state",
                            default=42,
                            type=int,
                            help="Evaluation random seed.")
    _ = parser.add_argument("--eval_test",
                            action="store_true",
                            default=False,
                            help="If included, evaluate baselines on the test split.")
    _ = parser.add_argument("--phraser_passes",
                            type=int,
                            default=0,
                            help="Number of phraser passes over the data to learn n-grams. Will be learned via PMI scoring. Default (0) is no phrase learning.")
    _ = parser.add_argument("--phraser_min_count",
                            type=int,
                            default=3,
                            help="Minimum occurrence threshold in phrase learning.")
    _ = parser.add_argument("--phraser_threshold",
                            type=float,
                            default=10,
                            help="Minimum phrase score threshold.")
    _ = parser.add_argument("--cache_errors",
                            action="store_true",
                            default=False,
                            help="If included, cache model errors to disk.")
    _ = parser.add_argument("--output_dir",
                            default="./",
                            type=str,
                            help="Where to store plots.")
    _ = parser.add_argument("--model_cache_dir",
                            default=None,
                            type=str,
                            help="If included, should be root path for storing classifiers. Will do additional breakdowns by model / target / etc. automatically.")
    _ = parser.add_argument("--rm_existing",
                            action="store_true",
                            default=False,
                            help="If included and output directory exists, remove it.")
    _ = parser.add_argument("--accelerator",
                            default="cpu",
                            type=str,
                            help="Choose the CPU/GPU accelerator. Not used, but included to prevent breaking changes.")
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
    ## Format Learning Curves
    if args.eval_learning_curves is not None:
        if len(args.eval_learning_curves) == 1:
            args.eval_learning_curves = args.eval_learning_curves[0]
        if args.eval_learning_curves_iter == 0:
            raise ValueError("Must use --eval_learning_curves_iter > 0 if asking to --eval_learning_curves")
    ## Format Context Arguments
    if args.eval_target_context == -1:
        args.eval_target_context = None
    if args.eval_original_context == -1:
        args.eval_original_context = None
    ## Validate Settings
    if args.model_settings is None or not os.path.exists(args.model_settings):
        raise FileNotFoundError("Could not find model settings.")
    return args
    
def _logistic_regression_grid_search(X_train,
                                     y_train,
                                     X_dev,
                                     y_dev,
                                     Cs=[0.01,0.03,0.1,0.3,1,3,5,10]):
    """
    
    """
    ## Track Performance
    best_c = None
    best_score = -np.inf
    ## Iterate Throigh Cs
    for c in Cs:
        ## Parameterize
        cmodel = linear_model.LogisticRegression(max_iter=10000,
                                                 solver="lbfgs",
                                                 class_weight="balanced",
                                                 C=c,
                                                 random_state=42,
                                                 multi_class="multinomial")
        ## Fit and Predict
        cmodel = cmodel.fit(X_train, y_train)
        cpred = cmodel.predict(X_dev)
        ## Score
        cscore = metrics.f1_score(y_dev, cpred, average="macro", zero_division=0)
        ## Update Best Performer
        if cscore > best_score:
            best_score = cscore
            best_c = c
    ## Reparameterize
    model = linear_model.LogisticRegression(max_iter=10000,
                                            solver="lbfgs",
                                            class_weight="balanced",
                                            C=best_c,
                                            random_state=42,
                                            multi_class="multinomial")
    return model

def _run_baseline(dataset_id,
                  X_train,
                  X_dev,
                  X_test,
                  y_train,
                  y_dev,
                  y_test,
                  meta_train,
                  meta_dev,
                  meta_test,
                  features,
                  target_labels,
                  model_type,
                  preprocessor,
                  eval_test,
                  model_cache_dir=None):
    """
    
    """
    ## Initialize Output Directory
    if model_cache_dir is not None and not os.path.exists(model_cache_dir):
        _ = os.makedirs(model_cache_dir)
    ## Compute Full Feature Representation (Only Relevant for Non-Majority Classifier)
    if model_type != "majority":
        features, X_train, X_dev, X_test = preprocessor.transform_base_feature_set(X_train=X_train,
                                                                                   X_dev=X_dev,
                                                                                   X_test=X_test,
                                                                                   features=features)
    ## Internal CV Generator (For Hyperparameter Grid Searches)
    gs_cv = model_utils.ResampleCVConstraint(n_min_train=1,
                                             n_min_test=0,
                                             n_samples=5,
                                             test_size=0.2,
                                             random_state=42)
    ## Initialize Model
    if model_type == "majority":
        model = dummy.DummyClassifier(strategy="prior")
    elif model_type == "linear":
        ## For Singular Binary Features, Use Conditional Majority
        if preprocessor._feature_set.get("conditional_majority_support",False):
            model = ConditionalMajorityClassifier(alpha=1)
        ## For Everything Else, Use Logistic Regression
        else:
            model = _logistic_regression_grid_search(X_train=X_train,
                                                     y_train=y_train,
                                                     X_dev=X_dev,
                                                     y_dev=y_dev,
                                                     Cs=[0.01,0.03,0.1,0.3,1,3,5,10])
    elif model_type == "random_forest":
        model = model_selection.GridSearchCV(estimator=ensemble.RandomForestClassifier(),
                                             param_grid={
                                                "n_estimators":[100],
                                                "class_weight":["balanced"],
                                                "max_depth":[None, 5, 10, 50],
                                                "min_samples_split":[2, 5, 10],
                                                "max_features":["auto"],
                                                "bootstrap":[True]
                                             },
                                             scoring="f1",
                                             refit=True,
                                             error_score=0,
                                             cv=gs_cv)
    else:
        raise ValueError("Unknown model_type")
    ## Fit Model
    model = model.fit(X_train, y_train)
    ## Save Model and Preprocessor (if Desired)
    if model_cache_dir is not None:
        ## Cache Targets
        with open(f"{model_cache_dir}/targets.txt","w") as the_file:
            the_file.write("\n".join(target_labels))
        ## Cache Preprocessor
        _ = preprocessor.save(f"{model_cache_dir}/preprocessor.joblib")
        ## Cache Classifier
        _ = joblib.dump(model, f"{model_cache_dir}/classifier.joblib")
    ## Predictions
    y_pred_train = model.predict_proba(X_train)
    y_pred_dev = model.predict_proba(X_dev)
    y_pred_test = model.predict_proba(X_test) if eval_test else None
    ## Performance
    scores = []
    errors = []
    group_scores = []
    for y_true, y_pred, split_name, meta in zip([y_train, y_dev, y_test],
                                                [y_pred_train,y_pred_dev,y_pred_test],
                                                ["train","dev","test"],
                                                [meta_train, meta_dev, meta_test]):
        ## Check
        if y_pred is None:
            continue
        ## Argmax
        y_pred_argmax = y_pred.argmax(axis=1)
        ## Performance By Metadata Group
        tar_labels = list(range(len(target_labels)))
        for col in meta.columns:
            ## Re-Cache Overall Performance
            group_scores.append({"support":len(y_true),
                                 "accuracy":metrics.accuracy_score(y_true, y_pred_argmax),
                                 "f1":metrics.f1_score(y_true, y_pred_argmax, average="macro", labels=tar_labels, zero_division=0),
                                 "precision":metrics.precision_score(y_true, y_pred_argmax, average="macro", labels=tar_labels, zero_division=0),
                                 "recall":metrics.recall_score(y_true, y_pred_argmax, average="macro", labels=tar_labels, zero_division=0),
                                 "class_balance":{target_labels[lbl]:(y_true == lbl).sum() for lbl in tar_labels},
                                 "model_type":model_type,
                                 "split":split_name,
                                 "group_category":col,
                                 "group_subcategory":"overall"})
            ## Group-level Performance
            if dataset_id.startswith("mimic-iv"):
                col_unique = settings.MIMIC_IV_PERFORMANCE_GROUPS[col]
            else:
                col_unique = []
            for colu in col_unique:
                colu_ind = (meta[col] == colu).values.nonzero()[0]
                if len(colu_ind) != 0:
                    group_scores.append({"support":len(colu_ind),
                                         "accuracy":metrics.accuracy_score(y_true[colu_ind], y_pred_argmax[colu_ind]),
                                         "f1":metrics.f1_score(y_true[colu_ind], y_pred_argmax[colu_ind], average="macro", zero_division=0),
                                         "precision":metrics.precision_score(y_true[colu_ind], y_pred_argmax[colu_ind], average="macro", zero_division=0),
                                         "recall":metrics.recall_score(y_true[colu_ind], y_pred_argmax[colu_ind], average="macro", zero_division=0),
                                         "class_balance":{target_labels[lbl]:(y_true[colu_ind] == lbl).sum() for lbl in tar_labels},
                                         "model_type":model_type,
                                         "split":split_name,
                                         "group_category":col,
                                         "group_subcategory":colu})
        ## Error Identification
        error_inds = (y_pred_argmax != y_true).nonzero()[0]
        error_true = [target_labels[i] for i in y_true[error_inds]]
        error_pred = [target_labels[i] for i in y_pred_argmax[error_inds]]
        ## Error Formatting
        error_df = pd.DataFrame({"index":error_inds,"y_true":error_true,"y_pred":error_pred})
        error_df["split"] = split_name
        error_df["model_type"] = model_type
        errors.append(error_df)
        ## Per Class Classification Report
        _cls_report = metrics.classification_report(y_true, y_pred_argmax, zero_division=0, output_dict=True, labels=range(len(target_labels)))
        _cls_report = {target_labels[int(x)]:y for x, y in _cls_report.items() if x.isdigit()}
        ## Per Class ROC/AUC (One vs. Rest)
        for tl, tl_lbl in enumerate(target_labels):
            tl_fpr, tl_tpr, tl_thresholds = metrics.roc_curve(y_true=(y_true == tl).astype(int), y_score=y_pred[:,tl], pos_label=1)
            tl_auc = metrics.auc(tl_fpr, tl_tpr)
            _cls_report[tl_lbl]["roc"] = {"fpr":tl_fpr,"tpr":tl_tpr,"thresholds":tl_thresholds,"auc":tl_auc}
        ## Confusion
        _cm = metrics.confusion_matrix(y_true, y_pred_argmax, labels=list(range(len(target_labels))))
        _cm = pd.DataFrame(_cm,
                           index=[f"true_{l}" for l in target_labels],
                           columns=[f"pred_{l}" for l in target_labels])
        ## Overall
        _scores = {
            "model_type":model_type,
            "split":split_name,
            "accuracy":metrics.accuracy_score(y_true, y_pred_argmax),
            "recall_macro":metrics.recall_score(y_true, y_pred_argmax, zero_division=0,average="macro"),
            "recall_micro":metrics.recall_score(y_true, y_pred_argmax, zero_division=0,average="micro"),
            "precision_macro":metrics.precision_score(y_true, y_pred_argmax, zero_division=0,average="macro"),
            "precision_micro":metrics.precision_score(y_true, y_pred_argmax, zero_division=0,average="micro"),
            "f1_macro":metrics.f1_score(y_true, y_pred_argmax, zero_division=0,average="macro"),
            "f1_micro":metrics.f1_score(y_true, y_pred_argmax, zero_division=0,average="micro"),
            "per_class":_cls_report,
            "confusion":_cm
        }
        ## Cache
        scores.append(_scores)
    ## Model Analysis
    if model_type == "linear":
        if len(target_labels) == 2 and not isinstance(model, ConditionalMajorityClassifier):
            coefs = pd.DataFrame(model.coef_.T, index=features, columns=target_labels[-1:])
        else:
            coefs = pd.DataFrame(model.coef_.T, columns=target_labels, index=features)
    elif model_type == "random_forest":
        coefs = np.vstack([[e.feature_importances_ for e in model.best_estimator_.estimators_]])
        coefs = coefs.mean(axis=0)
        coefs = pd.Series(coefs, index=features).to_frame("overall")
    else:
        coefs = None
    ## Errors
    errors = pd.concat(errors, axis=0, sort=False)
    ## Metadata
    if coefs is not None:
        coefs["model_type"] = model_type
    return scores, group_scores, coefs, errors

def run_baseline(dataset_id,
                 annotations,
                 target,
                 keycat_id,
                 preprocessor,
                 eval_test=False,
                 model_type=None,
                 error_cache=["dev","test"],
                 cv=None,
                 learning_curves=None,
                 learning_curves_iter=10,
                 exclude_keywords=set(),
                 random_state=42,
                 model_cache_dir=None):
    """
    
    """
    ## Check Model
    if model_type is None or model_type not in ["majority","linear","random_forest"]:
        raise ValueError("Model not supported.")
    ## Copy the Data
    adata = annotations.copy()
    ## Apply Exclusion Criteria
    adata = adata.loc[~adata["keyword"].isin(exclude_keywords)]
    ## Drop Null Targets
    adata = adata.dropna(subset=[target])
    ## Metadata (For Error Analysis)
    if dataset_id.startswith("mimic-iv"):
        meta_cols = list(settings.MIMIC_IV_PERFORMANCE_GROUPS.keys())
    else:
        meta_cols = None
    if meta_cols is None:
        raise ValueError("Dataset Unrecognized.")
    missing_meta = [m for m in meta_cols if m not in annotations.columns]
    meta_cols = [m for m in meta_cols if m in annotations.columns]
    if len(missing_meta) > 0:
        print(">> WARNING: Some of the expected group metadata attributes are missing. This may be expected.")
        print(">> WARNING: The following metadata columns are missing: {}".format(missing_meta))
    adata_train = adata.loc[adata["split"]=="train"][meta_cols].reset_index(drop=True)
    adata_dev = adata.loc[adata["split"]=="dev"][meta_cols].reset_index(drop=True)
    adata_test = adata.loc[adata["split"]=="test"][meta_cols].reset_index(drop=True)
    ## Build Targets
    target_labels = adata[target].value_counts().sort_values().index.tolist()
    y_train = model_utils.encode_targets(adata.loc[adata["split"]=="train", target].values, target_labels)
    y_dev = model_utils.encode_targets(adata.loc[adata["split"]=="dev", target].values, target_labels)
    y_test = model_utils.encode_targets(adata.loc[adata["split"]=="test", target].values, target_labels)
    ## Get (Cross-Validation/Training) Splits
    fold_train_mask, fold_dev_mask = model_utils.get_cross_validation_splits(cv=cv,
                                                                             y_train=y_train,
                                                                             y_dev=y_dev,
                                                                             random_state=random_state)
    ## Format Wrapper
    wrapper = enumerate(zip(fold_train_mask, fold_dev_mask))
    if cv is not None:
        wrapper = tqdm(wrapper, total=len(fold_train_mask), desc="[Fold]", file=sys.stdout, position=4, leave=False)
    ## Build Base Feature Set (Count Based - No Training Data Specific Transforms)
    features, X_train, X_dev, X_test = preprocessor.build_base_feature_set(annotations=adata, eval_test=eval_test)
    ## Iterate Over Folds
    scores, group_scores, coefs, errors = [], [], [], []
    for fold, (train_, dev_) in wrapper:
        ## Learning Curve Splits
        train_lc_ = model_utils.get_learning_curve_splits(train_index=train_,
                                                          train_targets=y_train[train_],
                                                          learning_curves=learning_curves,
                                                          learning_curves_iter=learning_curves_iter,
                                                          random_state=random_state)
        ## Wrapper
        train_lc_wrapper = train_lc_
        if learning_curves is not None:
            train_lc_wrapper = tqdm(train_lc_wrapper, desc="[Learning Curve Sample]", file=sys.stdout, leave=False, position=4 if cv is None else 5)
        ## Iterate Over Learning Curves
        for lc_inds_train, lc_iter, lc_percentile in train_lc_wrapper:
            ## Format Output Directory
            if model_cache_dir is None or lc_percentile != 100:
                lc_outdir = None
            else:
                lc_outdir = f"{model_cache_dir}/{keycat_id}_fold-{fold}"
            ## Run Fold
            lc_fold_scores, lc_fold_group_scores, lc_fold_coefs, lc_fold_errors = _run_baseline(dataset_id=dataset_id,
                                                                                                X_train=X_train[lc_inds_train],
                                                                                                X_dev=X_dev[dev_],
                                                                                                X_test=X_test,
                                                                                                y_train=y_train[lc_inds_train],
                                                                                                y_dev=y_dev[dev_],
                                                                                                y_test=y_test,
                                                                                                meta_train=adata_train.iloc[lc_inds_train],
                                                                                                meta_dev=adata_dev.iloc[dev_],
                                                                                                meta_test=adata_test,
                                                                                                features=features,
                                                                                                target_labels=target_labels,
                                                                                                model_type=model_type,
                                                                                                preprocessor=preprocessor,
                                                                                                eval_test=eval_test,
                                                                                                model_cache_dir=lc_outdir)
            ## Error Alignment (Full Set Only)
            if lc_percentile == 100:
                fold_errors_merged = []
                for group, group_inds in zip(["train","dev","test"],[lc_inds_train, dev_, None if not eval_test else np.arange(X_test.shape[0])]):
                    ## Skip Test Eval if Not Included
                    if group_inds is None:
                        continue
                    ## Skip If Not Asked For
                    if error_cache is None or group not in error_cache:
                        continue
                    ## Get Global Index
                    group_errors = lc_fold_errors.loc[lc_fold_errors["split"]==group].reset_index(drop=True)
                    group_errors_split_index = group_errors["index"].map(lambda i: group_inds[i])
                    ## Get Group Raw Data
                    group_adata = adata.loc[adata["split"]==group].iloc[group_errors_split_index.tolist()]
                    group_adata = group_adata[["enterprise_mrn","encounter_id","encounter_note_id","encounter_type","keyword_category","keyword","note_text"]]
                    group_adata = group_adata.reset_index(drop=True)
                    ## Merge Errors and Raw Data
                    merged_errors = pd.merge(group_adata, group_errors.drop("index",axis=1), left_index=True, right_index=True)
                    merged_errors["fold"] = fold
                    ## Cache Merged
                    fold_errors_merged.append(merged_errors)
                ## Concatenate
                fold_errors_merged = pd.concat(fold_errors_merged, axis=0, sort=False).reset_index(drop=True) if len(fold_errors_merged) > 0 else None
                fold_errors_merged["fold"] = fold
                fold_errors_merged["target"] = target
                ## Cache
                errors.append(fold_errors_merged)
            ## Coefficients (Full Set Only)
            if lc_percentile == 100 and lc_fold_coefs is not None:
                ## Check Columns
                if "fold" in lc_fold_coefs.columns:
                    raise KeyError("Did not expect fold to already exist in coefficients.")
                ## Add Metadata
                lc_fold_coefs["fold"] = fold
                lc_fold_coefs["target"] = target
                ## Cache
                coefs.append(lc_fold_coefs)
            ## Score Metadata
            for fs in lc_fold_scores:
                fs["fold"] = fold
                fs["target"] = target
                fs["learning_curve_train_proportion"] = lc_percentile
                fs["learning_curve_train_iteration"] = lc_iter
                fs["learning_curve_train_size"] = len(lc_inds_train)
            ## Group Score Metadata
            for gs in lc_fold_group_scores:
                gs["fold"] = fold
                gs["target"] = target
                gs["learning_curve_train_proportion"] = lc_percentile
                gs["learning_curve_train_iteration"] = lc_iter
                gs["learning_curve_train_size"] = len(lc_inds_train)
            ## Score Cache
            scores.extend(lc_fold_scores)
            group_scores.extend(lc_fold_group_scores)
    ## Merge Coefficients
    coefs = pd.concat(coefs, axis=0, sort=False) if len(coefs) > 0 else None
    errors = pd.concat(errors, axis=0, sort=False).reset_index(drop=True) if len(errors) > 0 else None
    return scores, group_scores, coefs, errors

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
    if args.model_cache_dir is not None and os.path.exists(args.model_cache_dir) and args.rm_existing:
        _ = os.system(f"rm -rf {args.model_cache_dir}")
    if args.model_cache_dir is not None and not os.path.exists(args.model_cache_dir):
        _ = os.makedirs(args.model_cache_dir)
    ## Load Model Settings
    print("[Loading Model Settings]")
    with open(args.model_settings,"r") as the_file:
        model_settings = json.load(the_file)
    with open(f"{args.model_cache_dir}/model_settings.json","w") as the_file:
        json.dump(model_settings, the_file, indent=1)
    ## Store Command Line Arguments
    print("[Caching Command Line Arguments]")
    with open(f"{args.output_dir}/cli.config.json","w") as the_file:
        json.dump(vars(args), the_file, indent=1)
    ## Load Annotations
    print("[Preparing Data For Modeling]")
    annotations, prepare_cache = model_utils.prepare_model_data(dataset=args.dataset,
                                                                eval_original_context=args.eval_original_context,
                                                                eval_target_context=args.eval_target_context,
                                                                eval_rm_keywords=args.eval_rm_keywords,
                                                                phraser_passes=args.phraser_passes,
                                                                phraser_min_count=args.phraser_min_count,
                                                                phraser_threshold=args.phraser_threshold,
                                                                tokenize=True,
                                                                tokenizer=None,
                                                                phrasers=None,
                                                                return_cache=True)
    ## Store Model Data Preparation Cache (Parameters/Tokenizer/Phrasers)
    if args.model_cache_dir is not None:
        print("[Caching Model Data Preprocessing Objects")
        _ = joblib.dump(prepare_cache, f"{args.model_cache_dir}/preprocessing.params.joblib")
    ## Isolate Relevant Keyword Categories
    print("[Isolating Appropriate Keyword Categories]")
    unique_keycats = list(settings.CAT2KEYS.keys())
    annotations = annotations.loc[annotations["keyword_category"].isin(unique_keycats)]
    ## Global Training Vocabulary
    print("[Extracting Model Vocabulary]")
    vocabulary, _ = text_utils.get_vocabulary(annotations.loc[annotations["split"]=="train"]["tokens"],
                                              rm_top=10,
                                              min_freq=3,
                                              max_freq=None,
                                              stopwords=["patient","pt"])
    ## Cache Vocabulary
    if args.model_cache_dir is not None:
        print("[Caching Base Vocabulary]")
        _ = joblib.dump(vocabulary, f"{args.model_cache_dir}/vocabulary.joblib")
    ## Run Baselines
    print("[Runnning Model Loop]")
    scores, group_scores, coefs, errors = {}, {}, {}, {}
    for keycat_id in tqdm(unique_keycats, desc="[Keyword Category]", position=0, leave=True, file=sys.stdout):
        ## Iterate Through Feature Sets
        scores[keycat_id], group_scores[keycat_id], coefs[keycat_id], errors[keycat_id] = [], [], [], []
        for fs, fs_params in tqdm(model_settings["feature_sets"].items(), total=len(model_settings["feature_sets"]), desc="[Feature Set]", position=1, leave=False, file=sys.stdout):
            ## Initialize Feature Set Constructor
            preprocessor = model_utils.FeaturePreprocessor(feature_set=fs_params)
            ## Iterate Over Classifier Types
            for mtype in tqdm(model_settings["models"], desc="[Model Type]", position=2, leave=False, file=sys.stdout):                
                ## Iterate Over Targets
                target = "label"
                ## Run Model
                keycat_scores, keycat_group_scores, keycat_coefs, keycat_errors = run_baseline(dataset_id=args.dataset,
                                                                                                annotations=annotations.loc[annotations["keyword_category"]==keycat_id],
                                                                                                target=target,
                                                                                                keycat_id=keycat_id,
                                                                                                preprocessor=preprocessor,
                                                                                                model_type=mtype,
                                                                                                cv=args.eval_cv,
                                                                                                learning_curves=args.eval_learning_curves,
                                                                                                learning_curves_iter=args.eval_learning_curves_iter,
                                                                                                random_state=args.eval_random_state,
                                                                                                error_cache=["dev","test"] if args.cache_errors is not None else None,
                                                                                                eval_test=args.eval_test,
                                                                                                model_cache_dir=None if args.model_cache_dir is None else f"{args.model_cache_dir}/{fs}/{mtype}/")
                ## Add Loop Metadata
                for kc in keycat_scores:
                    kc["keyword_category"] = keycat_id
                    kc["feature_set"] = fs
                for kcg in keycat_group_scores:
                    kcg["keyword_category"] = keycat_id
                    kcg["feature_set"] = fs
                if keycat_errors is not None:
                    keycat_errors["feature_set"] = fs
                ## Keyword Coefficients (Linear Model)
                if mtype == "linear" or mtype == "random_forest":
                    keycat_coefs["keyword_category"]  = keycat_id
                    keycat_coefs["feature_set"] = fs
                ## Cache
                scores[keycat_id].extend(keycat_scores)
                group_scores[keycat_id].extend(keycat_group_scores)
                if keycat_errors is not None:
                    errors[keycat_id].append(keycat_errors)
                if mtype == "linear" or mtype == "random_forest":
                    coefs[keycat_id].append(keycat_coefs)
    ## Cache Errors
    if args.cache_errors:
        print("[Caching Model Errors]")
        _ = train_utils.cache_all_errors(errors,
                                         output_dir=args.output_dir)
    ## Cache Scores
    print("[Caching Performance Outcomes]")
    _ = train_utils.cache_scores(scores,
                                 output_dir=args.output_dir,
                                 model_settings=model_settings)
    ## Plotting
    print("[Plotting Performance Outcomes]")
    _ = train_utils.plot_all_performance(scores=scores,
                                         learning_curves=args.eval_learning_curves,
                                         model_settings=model_settings,
                                         output_dir=args.output_dir)
    print("[Plotting Group-Based Performance Outcomes]")
    _ = train_utils.plot_all_performance_group(dataset_id=args.dataset,
                                               group_scores=group_scores,
                                               model_settings=model_settings,
                                               output_dir=args.output_dir)
    ## Plot Model Coefficients
    print("[Plotting Model Coefficients]")
    _ = train_utils.plot_all_coefficients(coefs=coefs,
                                          model_settings=model_settings,
                                          output_dir=args.output_dir)
    ## Done
    print("[Script Complete]")

####################
### Execute
####################

if __name__ == "__main__":
    _ = main()

