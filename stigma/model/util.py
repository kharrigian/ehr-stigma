#########################
### Imports
#########################

## Standard Libary
import os
import json
from copy import copy
from glob import glob
from collections import Counter

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction import DictVectorizer

## Local
from .. import (text_utils,
                util,
                settings)

#########################
### Functions
#########################

def _load_mimic_iv_discharge_annotations(context_update=False):
    """
    
    """
    ## Annotations
    print("[Loading MIMIC-IV Discharge Annotations]")
    annotations = util.load_annotations_mimic_iv_discharge(annotation_file=f"{settings.DEFAULT_ANNOTATIONS_DIR}/annotations.augmented.csv")
    ## Metadata
    print("[Loading MIMIC-IV Discharge Metadata]")
    metadata, _ = util.load_annotations_metadata_mimic_iv_discharge(annotations=annotations,
                                                                    clean_text=True,
                                                                    normalize_text=True,
                                                                    load_all=False,
                                                                    load_source=context_update)
    ## Merge
    print("[Merging MIMIC-IV Discharge Notes and Metadata]")
    drop_cols = ["encounter_note_unit","encounter_note_service","encounter_type","encounter_date","encounter_id","enterprise_mrn"] if context_update else ["enterprise_mrn"]
    merge_cols = ["encounter_note_id"] if context_update else ["encounter_id"]
    merged = pd.merge(annotations,
                      metadata.drop(drop_cols,axis=1),
                      on=merge_cols,
                      how="left")
    ## Return
    return merged

def prepare_model_data(dataset="mimic-iv-discharge",
                       eval_original_context=10,
                       eval_target_context=10,
                       eval_rm_keywords=None):
    """
    
    """
    ## Check Dataset
    if isinstance(dataset, str) and dataset not in ["mimic-iv-discharge"]:
        raise KeyError("Dataset not recognized.")
    ## Context Update
    context_update = eval_original_context != eval_target_context
    ## Default Set of Expected Columns
    expected_cols = ["encounter_type",
                     "enterprise_mrn",
                     "encounter_id",
                     "encounter_note_id",
                     "keyword",
                     "note_text",
                     "keyword",
                     "keyword_category"]
    if context_update:
        expected_cols.extend(["start","end","note_text_full"])
    ## Load Input Data
    if isinstance(dataset, str) and dataset == "mimic-iv-discharge":
        ## Load MIMIC-IV
        model_data = _load_mimic_iv_discharge_annotations(context_update=context_update)
    elif isinstance(dataset, pd.DataFrame):
        ## Use Input Set of Annotations
        model_data = dataset
    else:
        raise ValueError("Data `dataset` not recognized.")
    ## Validate Annotations Have What we Need
    ec_missing = list(filter(lambda ec: ec not in model_data.columns, expected_cols))
    if len(ec_missing) > 0:
        raise ValueError("Model data appears to be missing columns: {}".format(ec_missing))
    ## Update Context
    if context_update:
        ## Run Update
        print(f"[Updating Context Size ({eval_original_context} -> {eval_target_context})]")
        ## Make Update
        model_data["note_text"] = model_data.apply(lambda row: text_utils.get_context_window(text=row["note_text_full"],
                                                                                             start=row["start"],
                                                                                             end=row["end"],
                                                                                             window_size=eval_target_context,
                                                                                             clean_normalize=True,
                                                                                             strip_operators=False))
    ## Extract Negated Keyword
    print("[Extracting Negated Keywords]")
    model_data["keyword_negated"] = model_data.apply(lambda row: text_utils.extract_negated_keyword(row["keyword"],row["note_text"]),axis=1)
    ## Remove Keyword Removal
    if eval_rm_keywords is not None:
        print("[Removing Removal Set Keywords]")
        rm_keywords_missing = [k for k in eval_rm_keywords if k not in model_data["keyword"].values and k not in model_data["keyword_negated"].values]
        if rm_keywords_missing:
            print("[Warning! Some keywords passed for removal were not found: {}]".format(rm_keywords_missing))
        model_data = model_data.loc[~model_data["keyword"].isin(eval_rm_keywords)]
        model_data = model_data.loc[~model_data["keyword_negated"].isin(eval_rm_keywords)]
        model_data = model_data.reset_index(drop=True)
    ## Return
    return model_data

def get_vectorizer(feature_names):
    """
    
    """
    dvec = DictVectorizer(dtype=int, separator=":", sparse=True, sort=False)
    dvec.feature_names_ = feature_names
    dvec.vocabulary_ = {f:i for i, f in enumerate(feature_names)}
    if hasattr(dvec, "get_feature_names_out"): ## Use Alias to Address Scikit-learn API Breaking Change
        dvec.get_feature_names = dvec.get_feature_names_out
    return dvec

def encode_targets(targets, labels):
    """
    
    """
    lbl2ind = dict(zip(labels, range(len(labels))))
    targets_encoded = np.array([lbl2ind.get(tg) for tg in targets])
    return targets_encoded

def get_cross_validation_splits(cv,
                                y_train,
                                y_dev,
                                random_state=42):
    """
    Args:
        cv:
            None - No splitting
            int - Non-stratified K-fold
            (int, float) - Resampling
    """
    ## Prepare Training/Evaluation Splits/Samples
    if cv is None:
        fold_train_mask = [np.arange(y_train.shape[0])]
        fold_dev_mask = [np.arange(y_dev.shape[0])]
    elif not isinstance(cv, int) and not isinstance(cv, tuple):
        raise TypeError("Parameter 'cv' type not recognized. Expected integer for fold or tuple with (n_iter, proportion).")
    elif isinstance(cv, int):
        ## Initialze Seed
        seed = np.random.RandomState(random_state)
        ## Fold Assigment (Ensures Close to Equal Counts Per Fold)
        fold_train = np.array([i % cv for i in range(y_train.shape[0])]); _ = seed.shuffle(fold_train)
        fold_dev = np.array([i % cv for i in range(y_dev.shape[0])]); _ = seed.shuffle(fold_dev)
        ## Index Masks
        fold_train_mask = [(fold_train != fold).nonzero()[0] for fold in range(cv)]
        fold_dev_mask = [(fold_dev == fold).nonzero()[0] for fold in range(cv)]
    elif isinstance(cv, tuple):
        ## Check
        if not isinstance(cv[0], int) and isinstance(cv[1], float):
            raise TypeError("Tuple cv should be of type (int, float) (# iterations, sample proportion)")
        ## Initialize Seed
        seed = np.random.RandomState(random_state)
        ## Sample Sizes
        n_train = int(np.floor(y_train.shape[0] * cv[1]))
        n_dev = int(np.floor(y_dev.shape[0] * cv[1]))
        ## Fold Assignment
        fold_train_mask = [seed.choice(y_train.shape[0], n_train, replace=False) for _ in range(cv[0])]
        fold_dev_mask = [seed.choice(y_dev.shape[0], n_dev, replace=False) for _ in range(cv[0])]
    else:
        raise ValueError("This shouldn't happen based on the logic above.")
    ## Return
    return fold_train_mask, fold_dev_mask

def get_learning_curve_splits(train_index,
                              train_targets,
                              learning_curves=None,
                              learning_curves_iter=10,
                              random_state=42):
    """
    Generates learning curve splits (ensures that at least one sample from each
    class in the full training set is present at each iteration)
    """
    ## Format
    if not isinstance(train_index, np.ndarray):
        train_index = np.array(train_index)
    if not isinstance(train_targets, np.ndarray):
        train_targets = np.array(train_targets)
    ## Learning Curve Formatting
    if learning_curves is None:
        ## Only One Size
        train_lc_ = [(train_index, 0, 100)]
    else:
        ## Get Sizes
        if isinstance(learning_curves, int):
            lc_percs = np.linspace(0, 100, learning_curves + 1)[1:]
            lc_percs_diff = np.diff(lc_percs)[0]
            lc_sizes = np.floor((lc_percs/ 100) * train_index.shape[0]).astype(int)
            lc_sizes = [(s, lc_percs_diff * (i+1)) for i, s in enumerate(lc_sizes)]
        elif isinstance(learning_curves, list):
            lc_percs = copy(learning_curves)
            if max(lc_percs) != 100:
                lc_percs.append(100)
            lc_sizes = np.floor((np.array(lc_percs)/ 100) * train_index.shape[0]).astype(int)
            lc_sizes = [(s, p) for s, p in zip(lc_sizes, lc_percs)]
        else:
            raise TypeError("Learning curve parameter not understood.")
        ## Sample Seed
        seed = np.random.RandomState(random_state)
        ## Sample Process
        train_lc_ = []
        for size, size_p in lc_sizes:
            ## 100% Sample Once
            if size_p == 100:
                train_lc_.append((train_index, 0, 100))
                continue
            ## All Other Sample Sizes
            for lc_iter in range(learning_curves_iter):
                ## First: Sample 1 Instance from Each Class
                lc_samp = set()
                for tar_val in set(train_targets):
                    lc_samp.add(seed.choice(train_index[(train_targets == tar_val).nonzero()[0]], 1)[0])
                ## Second: Sample Reamining Instances
                lc_samp.update(seed.choice(list(filter(lambda i: i not in lc_samp, train_index)), size-len(lc_samp), replace=False))
                ## Sort
                lc_samp = sorted(lc_samp)
                
                ## Cache Sample
                train_lc_.append((lc_samp, lc_iter, size_p))
    return train_lc_

#########################
### Classes
#########################

class ResampleCVConstraint(object):
    
    """
    ResampleCVConstraint
    """

    def __init__(self,
                 n_min_train=1,
                 n_min_test=0,
                 n_samples=5,
                 test_size=0.2,
                 random_state=42):
        """
        ResampleCVConstraint

        Provides a resampling splitter that enforces a minimum sample size for each class
        in the train/test indices.
        """
        self._n_min_train = n_min_train
        self._n_min_test = n_min_test
        self._n_samples = n_samples
        self._test_size = test_size
        self._random_state = random_state
    
    def get_n_splits(self, X=None, y=None, groups=None, **kwargs):
        """
        
        """
        return self._n_samples
    
    def split(self, X, y, groups=None, **kwargs):
        """
        
        """
        ## Type Check
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        ## Dimension Check
        assert X.shape[0] == y.shape[0]
        ## Count Check
        y_counts = Counter(y)
        for tar, tar_count in y_counts.items():
            if self._n_min_train + self._n_min_test > tar_count:
                raise ValueError("Not enough data is available to support minimum counts.")
        ## Initialize Random Seed
        seed = np.random.RandomState(self._random_state)
        ## Yield Samples
        for _ in range(self._n_samples):
            ## Initialize Sample Cache
            train_ = set()
            test_ = set()
            ## Sample Minimums
            for target in sorted(y_counts.keys()):
                target_indices = (y == target).nonzero()[0]
                train_.update(seed.choice(target_indices, self._n_min_train, replace=False))
                test_.update(seed.choice([t for t in target_indices if t not in train_], self._n_min_test, replace=False))
            sampled_ = train_ | test_
            ## Sample Remaining
            train_.update(seed.choice(list(filter(lambda ind: ind not in sampled_, range(X.shape[0]))),
                          int(np.floor((1-self._test_size) * X.shape[0])) - len(train_),
                          replace=False))
            test_.update(list(filter(lambda ind: ind not in train_ and ind not in test_, range(X.shape[0]))))
            ## Sort
            train_ = sorted(train_)
            test_ = sorted(test_)
            ## Validate Sizes
            assert len(train_) + len(test_) == X.shape[0]
            ## Yield
            yield train_, test_