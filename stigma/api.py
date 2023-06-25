
"""
High-level wrapper to simplify working with existing stigmatizing language models
"""

#####################
### Imports
#####################

## Standard Library
import os
import json

## External Libraries
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import dummy
from transformers import AutoTokenizer

## Local Libraries
from . import util
from . import settings
from . import text_utils
from .model import bert as model_bert
from .model import util as model_utils

#####################
### Classes
#####################

class StigmaSearch(object):

    """
    StigmaSearch
    """

    def __init__(self,
                 task2keyword=settings.CAT2KEYS,
                 context_size=10):
        """
        Identify candidate instances of stigmatizing language in free text

        Args:
            task2keyword (dict): {task:[word 1, word 2, ...]}. Contains mapping of words associated
                                with each stigmatizing language task (i.e., adamant, compliance, other).
                                By default, we pass the currently supported set of keywords and tasks.
            context_size (int): Number of words to the left and right of a matched keyword to maintain
                                as context for downstream modeling. Default value is 10.
        """
        self._keyword_search = text_utils.KeywordSearch(task2keyword)
        self._context_size = context_size
    
    def __repr__(self):
        """
        
        """
        return f"StigmaSearch(context_size={self._context_size})"
    
    @staticmethod
    def show_default_keyword_categories():
        """
        See the default set of keyword categories and their associated keywords

        Args:
            None
        
        Returns:
            None (prints a string)
        """
        keycat_str = json.dumps(settings.CAT2KEYS, indent=1)
        print(keycat_str)

    def search(self,
               text):
        """
        Identify spans of stigmatizing keyword candidates in free text.

        Args:
            text (list): A list of input strings (e.g., clinical notes) you wish to search
                        for matches within.
        
        Returns:
            matches (DataFrame): Matches and their associated metadata
        """
        if isinstance(text, str):
            print(">> WARNING - Expected list of strings as input. Transforming str object to list.")
            text = [text]
        if not isinstance(text, list):
            raise ValueError("Text should be a list of strings.")
        ## Find Matches
        matches = self._keyword_search.search_batch(text)
        ## Flatten Matches and Extract Context
        matches_flat = []
        for doc_id, match in enumerate(matches):
            for match_dict in match:
                matches_flat.append({
                    "document_id":doc_id,
                    **match_dict,
                    "context":text_utils.get_context_window(text=text[doc_id],
                                                            start=match_dict["start"],
                                                            end=match_dict["end"],
                                                            window_size=self._context_size,
                                                            clean_normalize=True,
                                                            strip_operators=True)
                    })
        ## Format
        matches_flat = pd.DataFrame(matches_flat).loc[:, ["document_id","task","start","end","keyword","context"]]
        matches_flat = matches_flat.rename(columns={"task":"keyword_category", "context":"text"})
        ## Return
        return matches_flat

    def format_for_model(self,
                         keyword_category,
                         search_results):
        """
        Transform matches found by search() method to a form which can be used by StigmaModel

        Args:
            keyword_category (str): Which of the keyword categories (tasks) you wish to isolate
            search_results (DataFrame): Output dataframe from search() method. Should contain
                                    document_id, keyword_category, start, end, keyword, and text
                                    columns.
        
        Returns:
            document_ids (list): List of integer indices of the source documents
            keywords (list): List of str keywords associated with each candidate instance
            text (list): List of str context windows containing the potentially stigmatizing keywords
        """
        subset = search_results.loc[search_results["keyword_category"]==keyword_category,["document_id","text","keyword"]]
        if subset.shape == 0:
            print(">> WARNING: No matches for the specified keyword category were found in the search results.")
            return (None, None, None)
        return subset["document_id"].tolist(), subset["keyword"].tolist(), subset["text"].tolist()


class StigmaBaselineModel(object):

    """
    StigmaBaselineModel
    """

    def __init__(self,
                 model,
                 keyword_category,
                 preprocessing_params=None,
                 batch_size=None,
                 **kwargs):
        """
        Infer the impact (or lack thereof) of stigmatizing language candidates.

        Args:
            model (str): Either an ID fround in settings.MODELS or a raw model path.
            keyword_category (str): Keyword category you wish to load model for
                                    (e.g., "adamant", "compliance", "other")
            preprocessing_params (str or None): Either a path to preprocessing_params.joblib
                                file  associated with the model or None if using a default model
            batch_size (int): Number of instances to process simultaneously. Choose
                            based on your available compute resources.
        """
        ## Check Keyword Category
        if keyword_category not in settings.CAT2KEYS:
            print(">> WARNING: Received unexpected keyword_category not found in settings.CAT2KEYS")
        ## Validate Keyword Category
        self._keyword_category = keyword_category
        ## Validate Model
        self._model_name = model
        if model not in settings.MODELS:
            ## Model Might Be A Path
            if not os.path.exists(model):
                raise FileNotFoundError(f"Model not found ('{model}')")
            self._model = model
        else:
            if keyword_category not in settings.MODELS[model]["tasks"]:
                raise KeyError(f"Keyword category ({keyword_category}) not found in model directory.")
            if not os.path.exists(settings.MODELS[model]["tasks"][keyword_category]):
                raise FileNotFoundError("Model not found ('{}')".format(settings.MODELS[model]["tasks"][keyword_category]))
            if settings.MODELS[model]["model_type"] != "baseline":
                raise ValueError("Specified a non-baseline model. For BERT models, please use StigmaBertModel instead.")
            self._model = settings.MODELS[model]["tasks"][keyword_category]
        ## Preprocessing Parameters
        if preprocessing_params is None and model not in settings.MODELS:
            raise ValueError("Must provide a preprocessing_params.joblib file if not using a default baseline model.")
        elif preprocessing_params is None and model in settings.MODELS:
            self._preprocessing_params = settings.MODELS[model]["preprocessing"]
        else:
            self._preprocessing_params = preprocessing_params
        ## Initialization
        _ = self._initialize_preprocessing_params(self._preprocessing_params)
        _ = self._initialize_model(self._model)
        ## Other Class Parameters
        self._batch_size = batch_size
        
    def __repr__(self):
        """
        Show model information

        Args:
            None
        
        Returns:
            None
        """
        return f"StigmaBaselineModel(model='{self._model_name}', keyword_category='{self._keyword_category}')"

    def _initialize_preprocessing_params(self,
                                         preprocessing_params):
        """
        
        """
        ## Check for File
        if not isinstance(preprocessing_params, str) or not os.path.exists(preprocessing_params):
            raise FileNotFoundError("Unable to locate desired preprocessing_params.joblib file.")
        ## Load and Drop Unneccesary
        prepare_params = joblib.load(preprocessing_params)
        _ = prepare_params["params"].pop("tokenize", None)
        ## Store Attribute
        self._prepare_params = prepare_params
    
    def _initialize_model(self,
                          model):
        """
        Intialize the model classifier

        Args:
            model (str): Path to pretrained model directory. Should contain init.pth and model.pth files.
        
        Returns:
            None: Updates self.model_dict in place
        """
        ## Check
        if not isinstance(model, str) or not os.path.exists(model) or not all(os.path.exists(f"{model}/{s}") for s in ["preprocessor.joblib","classifier.joblib","targets.txt"]):
            raise FileNotFoundError("Unable to locate desired model directory and/or subfiles.")
        ## Initialize Dictionary
        self._model_dict = {}
        ## Load Tokenizer and Classification Model
        self._model_dict["preprocessor"] = joblib.load(f"{model}/preprocessor.joblib")
        ## Load Model
        self._model_dict["classifier"] = joblib.load(f"{model}/classifier.joblib")
        ## Load Targets
        with open(f"{model}/targets.txt","r") as the_file:
            self._model_dict["targets"] = the_file.read().split("\n")
        
    def update_eval_batch_size(self,
                               batch_size):
        """
        Update the processing batch size

        Args:
            batch_size (int): Desired batch size for inference
        
        Returns:
            None: Updates attribute self._batch_size in place
        """
        self._batch_size = batch_size
    
    @staticmethod
    def show_default_models():
        """
        See the avalable set of default models.

        Args:
            None
        
        Returns:
            None (prints available models)
        """
        models_str = json.dumps({x:y for x, y in settings.MODELS.items() if y.get("model_type",None)=="baseline"},indent=1)
        print(models_str)

    def predict(self,
                text,
                keywords):
        """
        Infer the nature of stigmatizing language candidate instances.

        Args:
            text (list of str): Context windows containing stigmatizing candidate keywords
            keywords (list of str): Keywords associated with each context window
        
        Returns:
            predictions (DataFrame): One row per input instance. Each column contains the 
                            predicted class probability for the respective task classes.
        """
        ## Format Input Text as a DataFrame
        note_df = pd.DataFrame({"note_text":text,"keyword":keywords})
        note_df["keyword_category"] = self._keyword_category
        for col in ["encounter_type","enterprise_mrn","encounter_id","encounter_note_id"]:
            note_df[col] = list(range(note_df.shape[0]))
        ## Prepare Data For Input into Mdoel
        note_df = model_utils.prepare_model_data(dataset=note_df,
                                                 tokenize=True,
                                                 tokenizer=self._prepare_params["tokenizer"],
                                                 phrasers=self._prepare_params["phrasers"],
                                                 return_cache=False,
                                                 **self._prepare_params["params"])
        ## Get Groups of Notes
        note_index_chunks = list(util.chunks(note_df.index.tolist(), self._batch_size)) if self._batch_size is not None else [note_df.index.tolist()]
        ## Iterate Through Chunks
        predictions = []
        for note_indices in tqdm(note_index_chunks, desc="[Making Predictions]", total=len(note_index_chunks)):
            ## Get Subset
            note_df_subset = note_df.loc[note_indices].copy()
            ## Build Base Feature Set
            X, features = self._model_dict["preprocessor"].build_base_feature_set_independent(annotations=note_df_subset)
            ## Transform the Base Feature Set
            if not isinstance(self._model_dict["classifier"], dummy.DummyClassifier):
                X, features = self._model_dict["preprocessor"].transform_base_feature_set_independent(X_out=X, features=features)
            ## Make Predictions
            predictions.append(self._model_dict["classifier"].predict_proba(X))
        ## Format All Predictions
        predictions = pd.DataFrame(np.vstack(predictions), columns=self._model_dict["targets"])
        predictions = predictions.sort_index(axis=1)
        ## Return
        return predictions


class StigmaBertModel(object):

    """
    StigmaBertModel
    """

    def __init__(self,
                 model,
                 keyword_category,
                 tokenizer=None,
                 batch_size=16,
                 device="cpu",
                 **kwargs):
        """
        Infer the impact (or lack thereof) of stigmatizing language candidates.

        Args:
            model (str): Either an ID fround in settings.MODELS or a raw model path.
            keyword_category (str): Keyword category you wish to load model for
                                    (e.g., "adamant", "compliance", "other")
            tokenizer (str or None): Either a path to tokenizer associated with the model,
                                    or None if using a default model
            batch_size (int): Number of instances to process simultaneously. Choose
                            based on your available compute resources.
            device (str): Either 'cpu' or 'cuda' depending on desired GPU acceleration
        
        """
        ## Check Keyword Category
        if keyword_category not in settings.CAT2KEYS:
            print(">> WARNING: Received unexpected keyword_category not found in settings.CAT2KEYS")
        if device not in ["cpu","cuda"]:
            raise ValueError("Expected 'device' to be one of 'cpu' or 'cuda'.")
        if batch_size <= 0:
            raise ValueError("Batch size should be at least 1.")
        ## Device
        self._device = device
        ## Validate Keyword Category
        self._keyword_category = keyword_category
        ## Validate Model
        self._model_name = model
        if model not in settings.MODELS:
            ## Model Might Be A Path
            if not os.path.exists(model):
                raise FileNotFoundError(f"Model not found ('{model}')")
            self._model = model
        else:
            if keyword_category not in settings.MODELS[model]["tasks"]:
                raise KeyError(f"Keyword category ({keyword_category}) not found in model directory.")
            if not os.path.exists(settings.MODELS[model]["tasks"][keyword_category]):
                raise FileNotFoundError("Model not found ('{}')".format(settings.MODELS[model]["tasks"][keyword_category]))
            if settings.MODELS[model]["model_type"] != "bert":
                raise ValueError("Specified a non-bert model. For baseline models, please use StigmaBaselineModel instead.")
            self._model = settings.MODELS[model]["tasks"][keyword_category]
        ## Tokenizer
        if tokenizer is None:
            if model not in settings.TOKENIZERS:
                raise KeyError("Must provide a tokenizer if not using a standard model from settings.MODELS")
            tokenizer = settings.TOKENIZERS[model]
        self._tokenizer = tokenizer
        ## Initialization
        _ = self._initialize_tokenizer(self._tokenizer)
        _ = self._initialize_model(self._model)
        ## Miscellaneous Class Attributes
        _ = self.update_eval_batch_size(batch_size)

    def __repr__(self):
        """
        
        """
        return f"StigmaBertModel(model='{self._model_name}', keyword_category='{self._keyword_category}')"
    
    @staticmethod
    def show_default_models():
        """
        See the avalable set of default models.

        Args:
            None
        
        Returns:
            None (prints available models)
        """
        models_str = json.dumps({x:y for x, y in settings.MODELS.items() if y.get("model_type",None)=="bert"},indent=1)
        print(models_str)

    def update_eval_batch_size(self,
                               batch_size):
        """
        Update the processing batch size

        Args:
            batch_size (int): Desired batch size for inference
        
        Returns:
            None: Updates attribute self._batch_size in place
        """
        self._batch_size = batch_size
    
    def _initialize_tokenizer(self,
                              tokenizer):
        """
        Initialize the model tokenizer

        Args:
            tokenizer (str): Path or name of the tokenizer. Uses Hugging Face API
        
        Returns:
            None: Updates self.tokenizer in place
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def _initialize_model(self,
                          model):
        """
        Intialize the model classifier

        Args:
            model (str): Path to pretrained model directory. Should contain init.pth and model.pth files.
        
        Returns:
            None: Updates self.model_dict in place
        """
        ## Verify Model Arguments Exist
        if not os.path.exists(f"{model}/init.pth"):
            raise FileNotFoundError(f"Unable to find expected model init file: {model}/init.pth")
        if not os.path.exists(f"{model}/model.pth"):
            raise FileNotFoundError(f"Unable to find expected model weights file: {model}/model.pth")
        ## Load Initialization Parameters
        print("[Loading Model Parameters]")
        init_param_file = f"{model}/init.pth"
        init_params = torch.load(init_param_file)
        ## Task ID Info
        task_id = {y:x for x, y in enumerate(init_params["task_targets"])}
        if self._keyword_category not in task_id:
            raise KeyError(f"Model keyword_category ('{self._keyword_category}') not found in specified model task set ({task_id})")
        task_id = task_id[self._keyword_category]
        targets = sorted(init_params["task_targets"][self._keyword_category], key=lambda x: init_params["task_targets"][self._keyword_category][x])
        ## Initialize Model Dictionary
        self.model_dict = {
            "task_id":task_id,
            "targets":targets,
            "init_params":init_params,
        }
        ## Dataset Encoding
        if init_params["classifier_only"]:
            print("[Initializing Encoder]")
            self.model_dict["encoder"] = model_bert.BERTEncoder(checkpoint=init_params["checkpoint"],
                                                                pool=True,
                                                                use_bert_pooler=False if "use_bert_pooler" not in init_params else init_params["use_bert_pooler"],
                                                                random_state=init_params["random_state"]).to(self._device)
        ## Initialize Model and Weights
        print("[Initializing Model Architecture]")
        if init_params["classifier_only"]:
            self.model_dict["classifier"] = model_bert.MultitaskClassifier(task_targets=init_params["task_targets"],
                                                                           in_dim=768,
                                                                           p_dropout=init_params["p_dropout"],
                                                                           random_state=init_params["random_state"]).to(self._device)
        else:
            self.model_dict["classifier"] = model_bert.BERTMultitaskClassifier(task_targets=init_params["task_targets"],
                                                                               checkpoint=init_params["checkpoint"],
                                                                               p_dropout=init_params["p_dropout"],
                                                                               use_bert_pooler=False if "use_bert_pooler" not in init_params else init_params["use_bert_pooler"],
                                                                               random_state=init_params["random_state"]).to(self._device)
        ## Load Weights
        print("[Initializing Model Weights]")
        _ = self.model_dict["classifier"].load_state_dict(torch.load(f"{model}/model.pth", map_location=torch.device('cpu')))
    
    def predict(self,
                text,
                keywords):
        """
        Infer the nature of stigmatizing language candidate instances.

        Args:
            text (list of str): Context windows containing stigmatizing candidate keywords
            keywords (list of str): Keywords associated with each context window
        
        Returns:
            predictions (DataFrame): One row per input instance. Each column contains the 
                            predicted class probability for the respective task classes.
        """
        ## Tokenize Data
        tokens, token_masks = model_bert.tokenize_and_mask(text=text,
                                                           keywords=keywords,
                                                           tokenizer=self.tokenizer,
                                                           mask_type=self.model_dict["init_params"]["mask_type"] if "mask_type" in self.model_dict["init_params"] else "keyword_all" if self.model_dict["init_params"]["include_all"] else "keyword")
        ## Initialize Token Dataset
        dataset = model_bert.ClassificationTokenDataset(tokens=tokens,
                                                        token_masks=token_masks,
                                                        labels=[-1 for _ in tokens],
                                                        task_ids=[self.model_dict["task_id"] for _ in tokens],
                                                        device=self._device)
        ## If Necessary, Encode Dataset
        if self.model_dict["init_params"]["classifier_only"]:
            dataset = model_bert.encode_dataset(dataset=dataset,
                                                bert=self.model_dict["encoder"],
                                                batch_size=self._batch_size,
                                                device=self._device)
        ## Make Predictions
        _, predictions = model_bert.classification_evaluate_model(model=self.model_dict["classifier"],
                                                                  dataset=dataset,
                                                                  n_tasks=len(self.model_dict["init_params"]["task_targets"]),
                                                                  loss_fcn=None,
                                                                  is_token=not self.model_dict["init_params"]["classifier_only"],
                                                                  batch_size=self._batch_size,
                                                                  verbose=True,
                                                                  eval_id=None,
                                                                  score=False,
                                                                  device=self._device)
        ## Isolate Appropriate Prediction Set
        predictions = predictions[self.model_dict["task_id"]].to("cpu").numpy()
        predictions = pd.DataFrame(data=predictions, columns=self.model_dict["targets"])
        predictions = predictions.sort_index(axis=1)
        ## Return
        return predictions
