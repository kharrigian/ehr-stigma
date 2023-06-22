
########################
### Imports
########################

## Standard Libary
import sys

## External Libraries
import torch
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Dataset
from transformers import AutoModel

## Private
from ..util import chunks

########################
### Datasets
########################

class MLMTokenDataset(Dataset):

    """
    MLMTokenDataset
    """

    def __init__(self,
                 data,
                 device="cpu",
                 **kwargs):
        """
        Args:
            data (list of list): List of examples. Each example is a truncated token list.
        """
        self._data = data
        self._device = device
    
    def __len__(self):
        """
        
        """
        return len(self._data)
    
    def __getitem__(self, idx):
        """
        
        """
        ## Get Item
        tokens = self._data[idx]
        ## Prepare
        item = {
                "input_ids":torch.tensor(tokens).to(self._device) if not isinstance(tokens, torch.Tensor) else tokens.to(self._device),
                "attention_mask":torch.tensor([1] * len(tokens)).to(self._device),
                "labels":torch.tensor(tokens).to(self._device) if not isinstance(tokens, torch.Tensor) else tokens.to(self._device)
                }
        return item

class ClassificationDataset(Dataset):

    """
    ClassificationDataset
    """

    def __init__(self,
                 data,
                 labels=None,
                 task_ids=None,
                 as_tensor=True,
                 feature_names=None,
                 device="cpu"):
        """
        
        """
        self._data = data
        self._labels = labels
        self._task_ids = task_ids
        self._as_tensor = as_tensor
        self._feature_names = feature_names
        self._device = device
    
    def __len__(self):
        """
        
        """
        return len(self._data)
    
    def __getitem__(self, idx):
        """
        
        """
        ## Extract
        datum = self._data[idx]
        label = self._labels[idx] if self._labels is not None else -1
        task_id = self._task_ids[idx] if self._labels is not None else 0
        ## Format
        if self._as_tensor:
            datum = torch.tensor(datum).to(self._device) if not isinstance(datum, torch.Tensor) else datum.to(self._device)
            label = torch.tensor(label).to(self._device) if not isinstance(label, torch.Tensor) else label.to(self._device)
            task_id = torch.tensor(task_id).to(self._device) if not isinstance(task_id, torch.Tensor) else task_id.to(self._device)
        ## Return
        return {"data":datum, "label":label, "task_id":task_id}

class ClassificationTokenDataset(Dataset):

    """
    ClassificationTokenDataset
    """

    def __init__(self,
                 tokens,
                 token_masks=None,
                 labels=None,
                 task_ids=None,
                 device="cpu",
                 **kwargs):
        """
        Args:
            tokens (list of list): List of examples. Each examples is a truncated token list.
            token_masks (list of list): List of boolean masks. Corresponds to input tokens (tokens).
            labels (list): Integer encodings of supervised labels if available. Otherwise, None.
            task_id (list): Integer task assignment if available. Otherwise, None
        """
        ## Store
        self._tokens = tokens
        self._token_masks = token_masks
        self._labels = labels
        self._task_ids = task_ids
        self._device = device
    
    def __len__(self):
        """
        
        """
        return len(self._tokens)
    
    def __getitem__(self, idx):
        """
        
        """
        ## Get Item
        tokens = self._tokens[idx]
        label = self._labels[idx] if self._labels is not None else None
        task_id = self._task_ids[idx] if self._task_ids is not None else 0
        token_masks = self._token_masks[idx] if self._token_masks is not None else [True for _ in tokens]
        ## Prepare
        item = {
                "input_ids":torch.tensor(tokens).to(self._device) if not isinstance(tokens, torch.Tensor) else tokens.to(self._device),
                "input_mask":torch.tensor(token_masks).to(self._device) if not isinstance(token_masks, torch.Tensor) else token_masks.to(self._device),
                "attention_mask":torch.tensor([1] * len(tokens)).to(self._device),
                "label":torch.tensor(label).to(self._device) if label is not None and not isinstance(label, torch.Tensor) else label.to(self._device) if label is not None and isinstance(label, torch.Tensor) else None,
                "task_id":torch.tensor(task_id).to(self._device) if not isinstance(task_id, torch.Tensor) else task_id.to(self._device),
                }
        return item

########################
### Models/Encoders
########################

## Classifier For Bert
class MultitaskClassifier(torch.nn.Module):

    """
    MultitaskClassifier
    """

    def __init__(self,
                 task_targets,
                 in_dim=768,
                 p_dropout=0.1,
                 random_state=None):
        """
        
        """
        ## Inheritence
        _ = super(MultitaskClassifier, self).__init__()
        ## Random Initialization
        if random_state is not None:
            _ = torch.manual_seed(random_state)
            if torch.cuda.is_available():
                rng = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng)
            else:
                rng = torch.get_rng_state()
                torch.set_rng_state(rng)
        ## Attributes
        self._random_state = random_state
        self._p_dropout = p_dropout
        self._in_dim = in_dim
        self._task_targets = task_targets
        ## Layers
        self._fc_dropout = torch.nn.Dropout(p=p_dropout)
        self._fc_layers = torch.nn.ModuleList()
        for task_id, task_labels in task_targets.items():
            self._fc_layers.append(torch.nn.Linear(in_features=in_dim, out_features=len(task_labels), bias=True))

    def forward(self,
                inputs,
                inputs_task=None):
        """
        
        """
        ## Parse Inputs
        data, task_ids = None, None
        if isinstance(inputs, dict) and inputs_task is None:
            data = inputs["data"]
            task_ids = inputs["task_id"]
        elif isinstance(inputs, torch.Tensor):
            data = inputs
            if inputs_task is None:
                raise ValueError("No task ids were passed.")
            task_ids = inputs_task
        ## Validate Inputs
        if data is None or task_ids is None:
            raise ValueError("Missing necessary data for forward pass.")
        ## Initialize Output Cache
        outputs = [None for _ in range(len(self._fc_layers))]
        task_masks = [None for _ in range(len(self._fc_layers))]
        ## DropOut
        out = self._fc_dropout(data)
        ## Get Output Logits and Task Masks
        for tid, task_layer in enumerate(self._fc_layers):
            outputs[tid] = task_layer(out)
            task_masks[tid] = task_ids == tid
        return outputs, task_masks

class BERTEncoder(torch.nn.Module):

    """
    BERTEncoder
    """

    def __init__(self,
                 checkpoint="emilyalsentzer/Bio_ClinicalBERT",
                 pool=True,
                 random_state=None,
                 use_bert_pooler=False):
        """
        
        """
        ## Inheritence
        _ = super(BERTEncoder, self).__init__()
        ## Random seed
        if random_state is not None:
            _ = torch.manual_seed(random_state)
            
            if torch.cuda.is_available():
                rng = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng)
            else:
                rng = torch.get_rng_state()
                torch.set_rng_state(rng)
        ## Attributes
        self._pool = pool
        self._random_state = random_state
        self._use_bert_pooler = use_bert_pooler
        ## Initialize BERT Layer
        self._bert = AutoModel.from_pretrained(checkpoint)

    def forward(self,
                inputs):
        """
        
        """
        ## Validate
        if "input_ids" not in inputs or "input_mask" not in inputs or "attention_mask" not in inputs:
            raise KeyError("Missing required keys from inputs.")
        ## Get Inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_masks = inputs["input_mask"] if self._pool and not self._use_bert_pooler else None
        ## Format Input Size
        if len(input_ids.size()) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            input_masks = input_masks.unsqueeze(0) if self._pool and not self._use_bert_pooler else None
        ## Get Token-Level Outputs
        bert_token_outputs = self._bert(input_ids, attention_mask=attention_mask)
        ## Early Return (No Pooling)
        if not self._pool:
            return bert_token_outputs
        ## Optionally Bert pooler (Ignoring Input Masks)
        if self._use_bert_pooler:
            return bert_token_outputs.pooler_output
        ## Custom Pooling of Last Hidden State
        bert_token_outputs = bert_token_outputs.last_hidden_state
        ## Format Token Masks To Do Mean Pooling Of Masked Tokens
        pooler = input_masks.unsqueeze(1).float()
        pooler = pooler / pooler.sum(2).unsqueeze(1)
        ## Apply Mask-based Pooling
        bert_pooled_output = (pooler @ bert_token_outputs).squeeze(1)
        return bert_pooled_output

## BERT Classifier With Multiple Task Classifier Heads
class BERTMultitaskClassifier(torch.nn.Module):

    """
    
    """

    def __init__(self,
                 task_targets,
                 checkpoint="emilyalsentzer/Bio_ClinicalBERT",
                 p_dropout=0.1,
                 use_bert_pooler=False,
                 random_state=None):
        """
        Args:
            task_targets (dict): {task_name:{label_str1:0,label_str2:1,...}}
            checkpoint (str): BERT model checkpoint
            p_dropout (float): Drop-out rate between pooler and fully connected layers
            random_state (int): Random initialization
        """
        ## Inheritence
        _ = super(BERTMultitaskClassifier, self).__init__()
        ## Random seed
        if random_state is not None:
            _ = torch.manual_seed(random_state)
            if torch.cuda.is_available():
                rng = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng)
            else:
                rng = torch.get_rng_state()
                torch.set_rng_state(rng)
        ## Parameter Validation
        if not isinstance(task_targets, dict):
            raise TypeError("'task_targets' should be a dictionary mapping each target to possible label set.")
        for task, labels in task_targets.items():
            if not isinstance(labels, dict):
                raise TypeError("Each value in task_targets dict should be a dictionary mapping label name to an integer ID")
        ## Key Attributes
        self._task_targets = task_targets
        self._checkpoint = checkpoint
        self._p_dropout = p_dropout
        self._random_state = random_state
        ## BERT Encoding
        self._bert = BERTEncoder(checkpoint=checkpoint,
                                 pool=True,
                                 use_bert_pooler=use_bert_pooler,
                                 random_state=random_state)
        ## Classification Layer
        self._classifier = MultitaskClassifier(task_targets=task_targets,
                                               in_dim=768,
                                               p_dropout=p_dropout,
                                               random_state=random_state)

    def forward(self,
                inputs):
        """
        
        """
        ## Run Token Inputs Through BERT
        bert_out = self._bert(inputs)
        ## Get Classifier Outputs
        outputs, task_masks = self._classifier(inputs=bert_out,
                                               inputs_task=inputs["task_id"])
        ## Return
        return outputs, task_masks

########################
### Functions
########################

def classification_collate_batch(batch, is_token=True, device="cpu"):
    """
    Args:
        batch (list of Dataset instances)
        is_token (bool): If True, expect ClassificationTokenDataset. Otherwise, ClassificationDataset
    
    Returns:
        items (dict): Collated data
    """
    ## Separate Components
    if is_token:
        ## Token-dataset Inputs
        input_ids = [b["input_ids"] for b in batch]
        input_masks = [b["input_mask"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        ## Get Max Length in Batch
        padmax = min(512, max(list(map(lambda x: x.shape[0], input_ids))))
        ## Add Padding to Input-Length Items
        input_ids = [torch.nn.functional.pad(i, (0, padmax-i.shape[0]), "constant", 0) for i in input_ids]
        attention_mask = [torch.nn.functional.pad(i, (0, padmax-i.shape[0]), "constant", 0) for i in attention_mask]
        input_masks = [torch.nn.functional.pad(i, (0, padmax-i.shape[0]), "constant", False) for i in input_masks]
    else:
        ## General Dataset Inputs
        data = [b["data"] for b in batch]
    ## All Dataset Inputs
    task_ids = list(map(lambda b: b["task_id"] if "task_id" in b else 0, batch))
    labels = [b["label"] for b in batch]
    ## Reformat Batch
    items = {
        "input_ids":torch.stack(input_ids).to(device) if is_token else None,
        "input_mask":torch.stack(input_masks).to(device) if is_token else None,
        "attention_mask":torch.stack(attention_mask).to(device) if is_token else None,
        "data":torch.stack(data).to(device) if not is_token else None,
        "label":torch.stack(labels).to(device) if not any(i is None for i in labels) else None,
        "task_id":torch.stack(task_ids).to(device),
    }
    return items

## Model Evaluation
def classification_evaluate_model(model,
                                  dataset,
                                  n_tasks,
                                  loss_fcn,
                                  batch_size=16,
                                  is_token=True,
                                  verbose=True,
                                  eval_id=None,
                                  score=True,
                                  score_average="macro",
                                  device="cpu"):
    """
    
    """
    ## Prediction Cache
    raw_predictions = [[] for _ in range(n_tasks)]
    ## Scoring Caches
    predictions = []
    labels = []
    task_ids = []
    global_index = []
    task_loss = torch.zeros(n_tasks).to(device)
    task_instances = torch.zeros(n_tasks).to(device)
    ## Put Model into Eval Mode
    model.eval()
    ## Run Forward Pass (And Compute Loss)
    with torch.no_grad():
        ## Batch Index
        batch_inds = [list(range(len(dataset)))]
        if batch_size is not None:
            batch_inds = list(chunks(list(range(len(dataset))), batch_size))
        ## Format Batch Index Wrapper
        batch_inds = tqdm(batch_inds, file=sys.stdout, desc=f"[Running Evaluation ({eval_id})]" if eval_id is not None else "[Running Evaluation]") if verbose else batch_inds
        ## Iterate Through Batches
        for indices in batch_inds:
            ## Collate The Batch of Data
            dataset_batch = classification_collate_batch([dataset[ind] for ind in indices],
                                                         is_token=is_token,
                                                         device=device)
            ## Run Forward Pass
            batch_outputs = model(dataset_batch)
            ## Iterate Through Tasks
            for tid, (tlogit, tmask) in enumerate(zip(*batch_outputs)):
                ## Store Predictions
                raw_predictions[tid].append(tlogit)
                ## Check For Instances to Score
                tn = tmask.sum()
                if tn == 0:
                    continue
                ## Score Predictions
                if score:
                    ## Update Number of Instances
                    task_instances[tid] += tn
                    ## Compute Loss
                    task_loss[tid] += loss_fcn[tid](tlogit[tmask], dataset_batch["label"][tmask]) * tn
                    ## Store Predictions and Group Truth
                    global_index.extend([indices[i] for i in torch.nonzero(tmask, as_tuple=True)[0]])
                    predictions.append(tlogit[tmask].argmax(1))
                    labels.append(dataset_batch["label"][tmask])
                    task_ids.append(dataset_batch["task_id"][tmask])
    ## Reset Model to Training Mode
    model.train()
    ## Predicted Probabilities for Each Task
    raw_predictions = [torch.softmax(torch.vstack(l), 1) for l in raw_predictions]
    ## Scoring
    if score:
        ## Compute Average Loss
        task_loss = task_loss / task_instances
        ## Merge
        predictions = torch.hstack(predictions).to("cpu")
        labels = torch.hstack(labels).to("cpu")
        task_ids = torch.hstack(task_ids).to("cpu")
        ## Task-Specific Performance
        task_performance = {}
        for task_id in torch.unique(task_ids, sorted=True, return_counts=False):
            task_id = task_id.item()
            task_performance[task_id] = {"loss":task_loss[task_id].item()}
            task_id_mask = task_ids == task_id
            for met, met_name in zip([metrics.accuracy_score, metrics.f1_score, metrics.recall_score, metrics.precision_score],
                                     ["accuracy","f1","recall","precision"]):
                if met_name != "accuracy":
                    task_performance[task_id][met_name] = met(labels[task_id_mask].numpy(), predictions[task_id_mask].numpy(), average=score_average, zero_division=0)
                else:
                    task_performance[task_id][met_name] = met(labels[task_id_mask].numpy(), predictions[task_id_mask].numpy())
    else:
        task_performance = None
    ## Return
    return task_performance, raw_predictions

## Task Weights
def get_loss_class_weights(dataset,
                           task_targets,
                           balance=True,
                           device="cpu"):
    """
    
    """
    ## Get Label Counts for Each Task
    counts = {}
    for instance in dataset:
        lbl = instance["label"].item()
        task = instance["task_id"].item()
        if task not in counts:
            counts[task] = {}
        if lbl not in counts[task]:
            counts[task][lbl] = 0
        counts[task][lbl] += 1
    for t_id, (task_name, task_label_id_dict) in enumerate(task_targets.items()):
        for l_id, (label_name, label_id) in enumerate(task_label_id_dict.items()):
            ## Ensure Order Makes Sense
            assert l_id == label_id
            ## Add if Necessary
            if label_id not in counts[t_id]:
                counts[t_id][label_id] = 0
    ## Sort Task Labels
    for task, task_counts in counts.items():
        counts[task] = [task_counts[i] for i in sorted(task_counts.keys())]
    ## Get Weights (If Balancing, Aligns with Sklearns Weighting for Balanced Classes)
    weights = {}
    for task, task_counts in counts.items():
        ## Task Info
        n_samples = sum(task_counts)
        n_classes = len(task_counts)
        ## Weight
        if balance:
            weights[task] = list(map(lambda c: n_samples / (n_classes * c) if c > 0 else 1, task_counts))
        else:
            weights[task] = [1 for c in task_counts]
        ## Store
        weights[task] = torch.tensor(weights[task]).to(device)
    return weights

## Dataset Encoding
def encode_dataset(dataset,
                   bert,
                   batch_size=None,
                   device="cpu"):
    """
    
    """
    ## Get Chunks (If Desired)
    indices = [list(range(len(dataset)))]
    if batch_size is not None:
        indices = list(chunks(list(range(len(dataset))), batch_size))
    ## Collate Batches and Encode
    dataset_out = None
    with torch.no_grad():
        ## Iterate Through Batches
        data = []
        for batch in tqdm(indices, total=len(indices), file=sys.stdout, desc="[Encoding Dataset]"):
            ## Collate 
            batch_inputs = classification_collate_batch([dataset[b] for b in batch],
                                                        is_token=True,
                                                        device=device)
            ## Encode
            batch_encoding = bert(batch_inputs)
            ## Score
            data.append(batch_encoding)
        ## Stack Data
        data = torch.vstack(data)
        ## Format New Dataset
        dataset_out = ClassificationDataset(data=data,
                                            labels=dataset._labels,
                                            task_ids=dataset._task_ids,
                                            device=device)

    return dataset_out

## Tokenize and Generate Token Masks
def tokenize_and_mask(text,
                      keywords,
                      tokenizer,
                      mask_type="keyword_all"):
    """
    
    """
    ## Check Parameters
    if mask_type not in ["none","keyword","keyword_all","cls","all","all_no_special"]:
        raise ValueError("Parameter mask_type not supported.")
    ## Run Tokenization/Masking
    tokens = []
    masks = []
    for txt, key in zip(text, keywords):
        ## Encode The Full Text Span
        txt_tok = tokenizer.encode(txt, add_special_tokens=True, truncation=True, max_length=512)
        ## Simple Masking
        if mask_type == "cls":
            key_mask = [True] + [False for _ in txt_tok[:-1]]
        elif mask_type == "all":
            key_mask = [True for _ in txt_tok]
        elif mask_type == "all_no_special":
            key_mask = [False] + [True for _ in txt_tok[1:-1]] + [False]
        elif mask_type == "none":
            key_mask = [False for _ in txt_tok]
        ## Keyword Masking
        elif mask_type in ["keyword","keyword_all"]:
            ## Encode the Keyword
            key_tok = tuple(tokenizer.encode(key, add_special_tokens=False, truncation=True, max_length=512))
            key_len = len(key_tok)
            ## Identify Locations of Keywords
            key_index_all = [i for i in range(len(txt_tok)) if tuple(txt_tok[i:i+key_len]) == key_tok]
            ## If Desired, Identify Primary Keyword Index (Starting Point - Middle of Context Input)
            if mask_type == "keyword" and len(key_index_all) > 1:
                key_index_all = [sorted(key_index_all, key=lambda x: abs(x - len(txt_tok) // 2))[0]]
            ## Create Mask
            key_mask = [False for _ in txt_tok]
            for ki in key_index_all:
                for i in range(key_len):
                    key_mask[ki + i] = True
        else:
            raise NotImplementedError("Mask type isn't supported.")
        ## Cache
        tokens.append(txt_tok)
        masks.append(key_mask)
    ## Return
    return tokens, masks