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
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

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
                       eval_rm_keywords=None,
                       phraser_passes=0,
                       phraser_min_count=3,
                       phraser_threshold=10,
                       tokenize=False,
                       tokenizer=None,
                       phrasers=None,
                       return_cache=False,
                       **kwargs):
    """
    
    """
    ## Check Dataset
    if isinstance(dataset, str) and dataset not in ["mimic-iv-discharge"]:
        raise KeyError("Dataset not recognized.")
    ## Check Inputs
    if phrasers is not None:
        if not isinstance(phrasers, dict) or not all(i in phrasers.keys() for i in ["tokens","tokens_full"]):
            raise TypeError("Input phrasers not recognized.")
    if tokenizer is not None and not isinstance(tokenizer, text_utils.Tokenizer):
        raise TypeError("Input tokenizer not recognized.")
    ## Context Update
    context_update = eval_original_context != eval_target_context
    ## Initialize Cache Of Parameters/Learned Transformers
    prepare_cache = {
        "params":{
            "phraser_passes":phraser_passes,
            "phraser_min_count":phraser_min_count,
            "phraser_threshold":phraser_threshold,
            "eval_original_context":eval_original_context,
            "eval_target_context":eval_target_context,
            "eval_rm_keywords":eval_rm_keywords,
            "tokenize":tokenize
        },
        "tokenizer":None,
        "phrasers":None
    }
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
    ## If Relevant,Tokenize and Re-Phrase
    if tokenize:
        ## Tokenization (Do Before Data Removal to Standardize Phrase Learning Across Runs with Same Context Window)
        print("[Tokenizing Text Data]")
        if tokenizer is None:
            tokenizer = text_utils.Tokenizer(filter_numeric=True,
                                             filter_punc=True,
                                             negate_handling=True,
                                             preserve_case=False)
        model_data["tokens"] = model_data.apply(lambda row: tokenizer.tokenize(row["note_text"], remove=[row["keyword"]]), axis=1)
        model_data["tokens_full"] = model_data.apply(lambda row: tokenizer.tokenize(row["note_text"], remove=[]), axis=1)
        ## Cache Parameterized Tokenizer
        prepare_cache["tokenizer"] = tokenizer
        ## N-Gram Transformation (If Desired)
        if phraser_passes is not None and phraser_passes > 0:
            ## Tokens w/o Keyword
            print("[Learning N-Grams within Vocabulary (Tokens w/o Keyword)]")
            if phrasers is None:
                phraser_no_key = text_utils.PhraseLearner(passes=phraser_passes,
                                                          min_count=phraser_min_count,
                                                          threshold=phraser_threshold,
                                                          verbose=True)
                if "split" in model_data.columns and "train" in model_data["split"].values:
                    phraser_no_key = phraser_no_key.fit(model_data.loc[model_data["split"]=="train","tokens"].tolist())
                else:
                    phraser_no_key = phraser_no_key.fit(model_data["tokens"].tolist())
            else:
                phraser_no_key = phrasers["tokens"]
            model_data["tokens"] = phraser_no_key.transform(model_data["tokens"].tolist())
            ## Tokens w/ Keyword
            print("[Learning N-Grams within Vocabulary (Tokens w/ Keyword)]")
            if phrasers is None:
                phraser_key = text_utils.PhraseLearner(passes=phraser_passes,
                                                       min_count=phraser_min_count,
                                                       threshold=phraser_threshold,
                                                       verbose=True)
                if "split" in model_data.columns and "train" in model_data["split"].values:
                    phraser_key = phraser_key.fit(model_data.loc[model_data["split"]=="train","tokens_full"].tolist())
                else:
                    phraser_key = phraser_key.fit(model_data["tokens_full"].tolist())
            else:
                phraser_key = phrasers["tokens_full"]
            model_data["tokens_full"] = phraser_key.transform(model_data["tokens_full"].tolist())
            ## Cache
            prepare_cache["phrasers"] = {"tokens":phraser_no_key, "tokens_full":phraser_key}
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
    return (model_data, prepare_cache) if return_cache else model_data


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

class FeaturePreprocessor(object):

    """
    
    """

    def __init__(self,
                 feature_set):
        """
        
        """
        ## Attributes
        self._feature_set = feature_set
        ## Variable Working Space
        self._base_features = None
        self._base_vectorizers = {}
        self._transformers_features = None
        self._transformers = {}
    
    def __repr__(self):
        """
        
        """
        return "FeaturePreprocessor({})".format(self._feature_set)
        
    def build_base_feature_set(self,
                               annotations,
                               eval_test=False):
        """
        
        """
        ## Initialize Cache
        features = []
        X_train, X_dev, X_test = [], [], ([] if eval_test else None)
        ## Build Components
        sources = set()
        for fs_component in self._feature_set["components"]:
            ## Check Source
            if fs_component["source"] in sources:
                continue
            ## Binary
            if fs_component["source"] in ["keyword","keyword_std","setting"]:
                ## Target Attribute
                fs_attr = {"keyword":"keyword_negated","keyword_std":"keyword_standardized","setting":"encounter_type"}
                fs_attr = fs_attr.get(fs_component["source"])
                ## Get Vectorizer
                vectorizer = get_vectorizer(annotations.loc[annotations["split"]=="train",fs_attr].unique())
                ## Extend Names
                features.extend([(fs_component["source"],k) for k in vectorizer.get_feature_names()])
                ## Add to Cache
                X_train.append(vectorizer.transform(annotations.loc[annotations["split"]=="train",fs_attr].map(lambda i: {i:True}).values))
                if (annotations["split"]=="dev").any():
                    X_dev.append(vectorizer.transform(annotations.loc[annotations["split"]=="dev",fs_attr].map(lambda i: {i:True}).values))
                if eval_test and (annotations["split"]=="test").any():
                    X_test.append(vectorizer.transform(annotations.loc[annotations["split"]=="test",fs_attr].map(lambda i: {i:True}).values))
                ## Cache Vectorizer
                self._base_vectorizers[fs_component["source"]] = vectorizer
            ## Counts
            elif fs_component["source"] == "tokens" or fs_component["source"] == "tokens_full":
                ## Get Vocabulary
                vocabulary, _ = text_utils.get_vocabulary(annotations.loc[annotations["split"]=="train"][fs_component["source"]],
                                                          rm_top=10,
                                                          min_freq=3,
                                                          max_freq=None,
                                                          stopwords=["patient","pt"])
                ## Get Vectorizer
                vectorizer = get_vectorizer(vocabulary)
                ## Extend Names
                features.extend([(fs_component["source"],k) for k in vectorizer.get_feature_names()])
                ## Add to Cache
                X_train.append(vectorizer.transform(annotations.loc[annotations["split"]=="train",fs_component["source"]].map(lambda i: Counter(i)).values))
                if (annotations["split"]=="dev").any():
                    X_dev.append(vectorizer.transform(annotations.loc[annotations["split"]=="dev",fs_component["source"]].map(lambda i: Counter(i)).values))
                if eval_test and (annotations["split"]=="test").any():
                    X_test.append(vectorizer.transform(annotations.loc[annotations["split"]=="test",fs_component["source"]].map(lambda i: Counter(i)).values))
                ## Cache Vectorizer
                self._base_vectorizers[fs_component["source"]] = vectorizer
        ## Stack
        X_train = hstack(X_train).tocsr()
        X_dev = hstack(X_dev).tocsr() if len(X_dev) > 0 else None
        X_test = hstack(X_test).tocsr() if eval_test and len(X_test) > 0 else None
        ## Cache Features
        self._base_features = features
        ## Return
        return features, X_train, X_dev, X_test
    
    def build_base_feature_set_independent(self,
                                           annotations):
        """
        
        """
        ## Initialize Cache
        features = []
        X_out = []
        ## Build Components
        sources = set()
        for fs_component in self._feature_set["components"]:
            ## Check Source
            if fs_component["source"] in sources:
                continue
            ## Binary
            if fs_component["source"] in ["keyword","keyword_std","setting"]:
                ## Target Attribute
                fs_attr = {"keyword":"keyword_negated","keyword_std":"keyword_standardized","setting":"encounter_type"}
                fs_attr = fs_attr.get(fs_component["source"])
                ## Get Vectorizer
                vectorizer = self._base_vectorizers[fs_component["source"]]
                ## Extend Names
                features.extend([(fs_component["source"],k) for k in vectorizer.get_feature_names()])
                ## Add to Cache
                X_out.append(vectorizer.transform(annotations[fs_attr].map(lambda i: {i:True}).values))
            ## Counts
            elif fs_component["source"] == "tokens" or fs_component["source"] == "tokens_full":
                ## Get Vectorizer
                vectorizer = self._base_vectorizers[fs_component["source"]]
                ## Extend Names
                features.extend([(fs_component["source"],k) for k in vectorizer.get_feature_names()])
                ## Add to Cache
                X_out.append(vectorizer.transform(annotations[fs_component["source"]].map(lambda i: Counter(i)).values))
        ## Stack
        X_out = hstack(X_out).tocsr()
        ## Check Features
        if features != self._base_features:
            raise ValueError("Found a mismatch between extracted feature set and learned feature set.")
        return X_out, features

    def transform_base_feature_set(self,
                                   X_train,
                                   features,
                                   X_dev=None,
                                   X_test=None):
        """
        
        """
        ## Re-initialize Transformers
        self._transformers = {"components":{}, "interactions":{}, "postprocessing":{}}
        ## Individual Component Representations
        features_t = []
        X_train_t, X_dev_t, X_test_t = [], [], ([] if X_test is not None else None)
        features_comp_names = set()
        for fs_component in self._feature_set["components"]:
            ## Track Component Names
            features_comp_names.add(fs_component["name"])
            ## Find Appropriate Source Indices
            fs_component_inds = [i for i, (fsrc,fval) in enumerate(features) if fsrc == fs_component["source"]]
            ## Representation (Binary, Counts, Encoding -> No Change)
            if fs_component["type"] == "binary" or fs_component["type"] == "counts":
                ## Add to Cache
                X_train_t.append(X_train[:,fs_component_inds])
                if X_dev is not None:
                    X_dev_t.append(X_dev[:,fs_component_inds])
                if X_test is not None:
                    X_test_t.append(X_test[:,fs_component_inds])
                ## Update Features
                features_t.extend([(fs_component["name"], features[ind][1]) for ind in fs_component_inds])
            ## Representation (Tf-IDF)
            elif fs_component["type"] == "tfidf":
                ## Fit Transformer
                tfidf_transformer = TfidfTransformer(**fs_component["type_kwargs"])
                tfidf_transformer = tfidf_transformer.fit(X_train[:,fs_component_inds])
                ## Store Transformer
                self._transformers["components"][fs_component["name"]] = tfidf_transformer
                ## Apply Transform and Cache
                X_train_t.append(tfidf_transformer.transform(X_train[:,fs_component_inds]))
                if X_dev is not None:
                    X_dev_t.append(tfidf_transformer.transform(X_dev[:,fs_component_inds]))
                if X_test is not None:
                    X_test_t.append(tfidf_transformer.transform(X_test[:,fs_component_inds]))
                ## Update Features
                features_t.extend([(fs_component["name"], features[ind][1]) for ind in fs_component_inds])
            ## Other
            else:
                raise NotImplementedError("Feature type '{}' not recognized".format(fs_component["type"]))
        ## Concatenate Transformed Features
        X_train_t = hstack(X_train_t).tocsr()
        X_dev_t = hstack(X_dev_t).tocsr() if X_dev is not None else None
        X_test_t = hstack(X_test_t).tocsr() if X_test is not None else None
        ## Feature Set Interactions
        features_int = []
        X_train_int, X_dev_int, X_test_int = [], [], ([] if X_test is not None else None)
        features_int_names = set()
        for fs_interaction in self._feature_set["interactions"]:
            ## Check Sources
            if fs_interaction["component_1"] not in features_comp_names:
                raise KeyError("Component 1 feature not found: '{}'".format(fs_interaction["component_1"]))
            if fs_interaction["component_2"] not in features_comp_names:
                raise KeyError("Component 2 feature not found: '{}'".format(fs_interaction["component_2"]))
            ## Keep Track of Interaction Ids
            features_int_names.add(fs_interaction["name"])
            ## Interaction Indices
            comp_1_inds = [i for i, f in enumerate(features_t) if f[0] == fs_interaction["component_1"]]
            comp_2_inds = [i for i, f in enumerate(features_t) if f[0] == fs_interaction["component_2"]]
            if len(comp_1_inds) < len(comp_2_inds):
                comp_1_inds, comp_2_inds = comp_2_inds, comp_1_inds
            ## Compute Interactions
            X_comp_int_train = hstack([X_train_t[:,comp_1_inds].multiply(X_train_t[:,comp_2_inds][:,i]) for i in range(len(comp_2_inds))]).tocsr()
            if X_dev is not None:
                X_comp_int_dev = hstack([X_dev_t[:,comp_1_inds].multiply(X_dev_t[:,comp_2_inds][:,i]) for i in range(len(comp_2_inds))]).tocsr()
            if X_test is not None:
                X_comp_int_test = hstack([X_test_t[:,comp_1_inds].multiply(X_test_t[:,comp_2_inds][:,i]) for i in range(len(comp_2_inds))]).tocsr()
            feats_comp_int = [(fs_interaction["name"], "({})x({})".format("{}={}".format(*features_t[nind]),"{}={}".format(*features_t[tind]))) for nind in comp_2_inds for tind in comp_1_inds]
            ## Interaction Transformation
            if fs_interaction["type"] is not None:
                ## TF-IDF (Post-Interaction)
                if fs_interaction["type"] == "tfidf":
                    ## Fit
                    tfidf_transformer = TfidfTransformer(**fs_interaction["type_kwargs"])
                    tfidf_transformer = tfidf_transformer.fit(X_comp_int_train)
                    ## Cache Transformer
                    self._transformers["interaction"][fs_interaction["name"]] = tfidf_transformer
                    ## Transform
                    X_comp_int_train = tfidf_transformer.transform(X_comp_int_train)
                    if X_dev is not None:
                        X_comp_int_dev = tfidf_transformer.transform(X_comp_int_dev)
                    if X_test is not None:
                        X_comp_int_test = tfidf_transformer.transform(X_comp_int_test)
                else:
                    raise NotImplementedError("Interaction type not recognized: '{}'".format(fs_interaction["type"]))
            ## Cache
            X_train_int.append(X_comp_int_train)
            if X_dev is not None:
                X_dev_int.append(X_comp_int_dev)
            if X_test is not None:
                X_test_int.append(X_comp_int_test)
            ## Update Features
            features_int.extend(feats_comp_int)
        ## Concatenate Interaction Features
        X_train_int = hstack(X_train_int).tocsr() if len(X_train_int) > 0 else None
        X_dev_int = hstack(X_dev_int).tocsr() if len(X_dev_int) > 0 else None
        X_test_int = hstack(X_test_int).tocsr() if X_test is not None and len(X_test_int) > 0 else None
        ## Final Representation
        features_r = []
        X_train_r, X_dev_r, X_test_r = [], [], ([] if X_test is not None else None)
        for fs_rep_name in self._feature_set["representation"]:
            ## Ensure Exists
            if fs_rep_name not in features_comp_names | features_int_names:
                raise KeyError("Feature representation not found: '{}'".format(fs_rep_name))
            ## See if Interaction or Component
            if fs_rep_name in features_comp_names:
                ## Get Appropriate Indices
                fs_rep_ind = [i for i, f in enumerate(features_t) if f[0] == fs_rep_name]
                ## Cache
                X_train_r.append(X_train_t[:,fs_rep_ind])
                if X_dev is not None:
                    X_dev_r.append(X_dev_t[:,fs_rep_ind])
                if X_test is not None:
                    X_test_r.append(X_test_t[:,fs_rep_ind])
                ## Update Features
                features_r.extend([features_t[ind] for ind in fs_rep_ind])
            elif fs_rep_name in features_int_names:
                ## Get Appropriate Indices
                fs_rep_ind = [i for i, f in enumerate(features_int) if f[0] == fs_rep_name]
                ## Cache
                X_train_r.append(X_train_int[:,fs_rep_ind])
                if X_dev is not None:
                    X_dev_r.append(X_dev_int[:,fs_rep_ind])
                if X_test is not None:
                    X_test_r.append(X_test_int[:,fs_rep_ind])
                ## Update Features
                features_r.extend([features_int[ind] for ind in fs_rep_ind])
        ## Concatenate Final Feature Representation
        X_train_r = hstack(X_train_r).tocsr() if len(X_train_r) > 0 else None
        X_dev_r = hstack(X_dev_r).tocsr() if len(X_dev_r) > 0 else None
        X_test_r = hstack(X_test_r).tocsr() if X_test is not None and len(X_test_r) > 0 else None
        ## Standardization
        cent = self._feature_set.get("standardize",{}).get("center",False)
        scal = self._feature_set.get("standardize",{}).get("scale",False)
        if cent or scal:
            scaler = StandardScaler(with_mean=cent, with_std=scal)
            X_train_r = csr_matrix(scaler.fit_transform(X_train_r.A)) if cent else scaler.fit_transform(X_train_r)
            if X_dev is not None:
                X_dev_r = csr_matrix(scaler.transform(X_dev_r.A)) if cent else scaler.transform(X_dev_r)
            if X_test is not None:
                X_test_r = csr_matrix(scaler.transform(X_test_r.A)) if cent else scaler.transform(X_test_r)
            self._transformers["postprocessing"]["standardize"] = scaler
        ## Cache Features
        self._transformers_features = features_r
        ## Return
        return features_r, X_train_r, X_dev_r, X_test_r
    
    def transform_base_feature_set_independent(self,
                                               X_out,
                                               features):
        """
        
        """
        ## Individual Component Representations
        features_t = []
        X_out_t = []
        features_comp_names = set()
        for fs_component in self._feature_set["components"]:
            ## Track Component Names
            features_comp_names.add(fs_component["name"])
            ## Find Appropriate Source Indices
            fs_component_inds = [i for i, (fsrc,fval) in enumerate(features) if fsrc == fs_component["source"]]
            ## Representation (Binary, Counts, Encoding -> No Change)
            if fs_component["type"] == "binary" or fs_component["type"] == "counts":
                ## Add to Cache
                X_out_t.append(X_out[:,fs_component_inds])
                ## Update Features
                features_t.extend([(fs_component["name"], features[ind][1]) for ind in fs_component_inds])
            ## Representation (Tf-IDF)
            elif fs_component["type"] == "tfidf":
                ## Get Transformer
                tfidf_transformer = self._transformers["components"][fs_component["name"]]
                ## Apply Transform and Cache
                X_out_t.append(tfidf_transformer.transform(X_out[:,fs_component_inds]))
                ## Update Features
                features_t.extend([(fs_component["name"], features[ind][1]) for ind in fs_component_inds])
            ## Other
            else:
                raise NotImplementedError("Feature type '{}' not recognized".format(fs_component["type"]))
        ## Concatenate Transformed Features
        X_out_t = hstack(X_out_t).tocsr()
        ## Feature Set Interactions
        features_int = []
        X_out_int = []
        features_int_names = set()
        for fs_interaction in self._feature_set["interactions"]:
            ## Check Sources
            if fs_interaction["component_1"] not in features_comp_names:
                raise KeyError("Component 1 feature not found: '{}'".format(fs_interaction["component_1"]))
            if fs_interaction["component_2"] not in features_comp_names:
                raise KeyError("Component 2 feature not found: '{}'".format(fs_interaction["component_2"]))
            ## Keep Track of Interaction Ids
            features_int_names.add(fs_interaction["name"])
            ## Interaction Indices
            comp_1_inds = [i for i, f in enumerate(features_t) if f[0] == fs_interaction["component_1"]]
            comp_2_inds = [i for i, f in enumerate(features_t) if f[0] == fs_interaction["component_2"]]
            if len(comp_1_inds) < len(comp_2_inds):
                comp_1_inds, comp_2_inds = comp_2_inds, comp_1_inds
            ## Compute Interactions
            X_comp_int_out = hstack([X_out_int[:,comp_1_inds].multiply(X_out_int[:,comp_2_inds][:,i]) for i in range(len(comp_2_inds))]).tocsr()
            feats_comp_int = [(fs_interaction["name"], "({})x({})".format("{}={}".format(*features_t[nind]),"{}={}".format(*features_t[tind]))) for nind in comp_2_inds for tind in comp_1_inds]
            ## Interaction Transformation
            if fs_interaction["type"] is not None:
                ## TF-IDF (Post-Interaction)
                if fs_interaction["type"] == "tfidf":
                    ## Get Transformer
                    tfidf_transformer = self._transformers["interaction"][fs_interaction["name"]]
                    ## Transform
                    X_comp_int_out = tfidf_transformer.transform(X_comp_int_out)
                else:
                    raise NotImplementedError("Interaction type not recognized: '{}'".format(fs_interaction["type"]))
            ## Cache
            X_out_int.append(X_comp_int_out)
            ## Update Features
            features_int.extend(feats_comp_int)
        ## Concatenate Interaction Features
        X_out_int = hstack(X_out_int).tocsr() if len(X_out_int) > 0 else None
        ## Final Representation
        features_r = []
        X_out_r = []
        for fs_rep_name in self._feature_set["representation"]:
            ## Ensure Exists
            if fs_rep_name not in features_comp_names | features_int_names:
                raise KeyError("Feature representation not found: '{}'".format(fs_rep_name))
            ## See if Interaction or Component
            if fs_rep_name in features_comp_names:
                ## Get Appropriate Indices
                fs_rep_ind = [i for i, f in enumerate(features_t) if f[0] == fs_rep_name]
                ## Cache
                X_out_r.append(X_out_t[:,fs_rep_ind])
                ## Update Features
                features_r.extend([features_t[ind] for ind in fs_rep_ind])
            elif fs_rep_name in features_int_names:
                ## Get Appropriate Indices
                fs_rep_ind = [i for i, f in enumerate(features_int) if f[0] == fs_rep_name]
                ## Cache
                X_out_r.append(X_out_int[:,fs_rep_ind])
                ## Update Features
                features_r.extend([features_int[ind] for ind in fs_rep_ind])
        ## Concatenate Final Feature Representation
        X_out_r = hstack(X_out_r).tocsr() if len(X_out_r) > 0 else None
        ## Standardization
        cent = self._feature_set.get("standardize",{}).get("center",False)
        scal = self._feature_set.get("standardize",{}).get("scale",False)
        if cent or scal:
            scaler = self._transformers["postprocessing"]["standardize"]
            X_out_r = csr_matrix(scaler.transform(X_out_r.A)) if cent else scaler.transform(X_out_r)
        ## Check Against Cache Features
        if features_r != self._transformers_features:
            raise ValueError("Found mismatch between learned feature transform and extracted features.")
        ## Return
        return X_out_r, features_r
    
    def save(self,
             filename):
        """
        
        """
        ## Save
        _ = joblib.dump(self, filename)