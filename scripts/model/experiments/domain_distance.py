
"""
Analysis of how learned embeddings encode auxiliary attributes
"""

##################
### Imports
##################

## Standard Library
import os
import re
import json
import argparse
from glob import glob

## External Libraries
import torch
import numpy as np
import pandas as pd
from umap import UMAP
from scipy import sparse, stats
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics

## Private
from stigma import settings
from stigma.model.baseline import ConditionalMajorityClassifier
from stigma.model import util as model_utils
from stigma.model import bert as bert_utils

##################
### Globals
##################

## Groups For Plotting Keyword Projections
KEYWORD_ANALYSIS_GROUPS = {
    "adamant":[
        ["claim","claims","claimed","claiming"],
        ["insist","insists","insisted","insisting","insistence"],
        ["adamant","adament","adamantly","adamently"],
    ],
    "compliance":[
        ["adhere","adheres","adhered","adherance","adherence","adherent","adhering","nonadherance","nonadherence","nonadherent"],
        ["compliance","compliant","complies","complied","comply","complying","noncompliance","noncompliant"],
        ["declined","declines","declining"],
        ["refuse","refusal","refused","refuses","refusing"]
    ],
    "other":[
        ["aggressive","aggressively","aggression","angry","anger","angers","angered","angrily","angrier","agitated","agitation"],
        ["argumentative","argumentatively","belligerence","belligerent","belligerently","combative","confronted","confrontational","defensive"],
        ["charming","pleasant","pleasantly","lovely","delightful","well groomed","well-groomed"],
        ["poorly groomed","poorly-groomed","unkempt","disheveled"],
        ["drug-seeking","drug seeking","narcotic-seeking","narcotic seeking","secondary gain","malinger","malingers","malingered","malingerer","malingering"],
        ["cooperative","cooperate","cooperates","cooperated","cooperation","cooperating","uncooperative"],
        ["historian"],
        ["exaggerate","exaggerates","exaggerating"],
        ["unmotivated","unwilling","unwillingly"],
        ["perseverate","perseverates","perseverated","perseveration","perseverating"],
    ]
}

## Identifier Swaps
IDENTIFER_SWAP = {
    "they":re.compile(r"\b((s?)he)\b", flags=re.IGNORECASE),
    "them":re.compile(r"\b(him|her)\b", flags=re.IGNORECASE),
    "their":re.compile(r"\b(his|hers)\b", flags=re.IGNORECASE),
    "themselves":re.compile(r"\b((him|her)self)\b", flags=re.IGNORECASE),
    "person":re.compile(r"\b(male|female|girl|boy|man|woman)\b", flags=re.IGNORECASE),
    "patient":re.compile(r"\b((mr|mrs|ms|miss)(\.?)([ ]))(\w*)\b", flags=re.IGNORECASE),
    "partner":re.compile(r"\b(husband|wife)\b", flags=re.IGNORECASE)
}

## Mimic-IV
MIMIC_IV_DIS_ENC_TYPE_COL = "encounter_note_service"

##################
### Functions
##################

def parse_command_line():
    """

    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--dataset_id", type=str, choices={"mimic-iv-discharge"}, default="mimic-iv-discharge")
    _ = parser.add_argument("--output_dir", type=str, default=None)
    _ = parser.add_argument("--model_root", type=str, default=None)
    _ = parser.add_argument("--target_domains", nargs="+", type=str, default=["keyword","label","encounter_type","patient_gender","patient_race"])
    _ = parser.add_argument("--verbose", action="store_true", default=False)
    _ = parser.add_argument("--race_ignore", nargs="*", type=str, default=["Unknown","Declined to Answer","Patient Declined To Answer","Unable To Obtain"])
    _ = parser.add_argument("--sex_ignore", nargs="*", type=str, default=["Unknown"])
    _ = parser.add_argument("--race_other", nargs="*", type=str, default=["Hispanic or Latino","Asian","American Indian or Native","American Indian or Alaska Native","Portuguese","South American","Mixed"])
    _ = parser.add_argument("--training_only", action="store_true", default=False)
    _ = parser.add_argument("--keywords_exclude", nargs="*", type=str, default=None)
    _ = parser.add_argument("--device", type=str, choices={"cpu","cuda"}, default="cpu")
    args = parser.parse_args()
    ## Validate
    if args.output_dir is None:
        raise FileNotFoundError("Must provide an --output_dir")
    if args.model_root is None or not os.path.exists(args.model_root):
        raise FileNotFoundError("Must provide a valid --model_root")
    return args


def swap_identifiers(text):
    """
    
    """
    for rep, pat in IDENTIFER_SWAP.items():
        text = pat.sub(rep, text)
    return text

def get_model_paths(model_root):
    """
    
    """
    ## Cache
    model_paths = {"adamant":{},"compliance":{},"other":{}}
    ## Look for Files
    glob_result = sorted(glob(f"{model_root}/*/best_model.json"))
    if len(glob_result) == 0:
        raise ValueError("Did not find any valid encoders.")
    ## Iterate Through Found Files
    for path in glob_result:
        ## Load Path Metadata
        with open(path.replace("best_model.json","targets.json"),"r") as the_file:
            path_targets = json.load(the_file)
        path_targets = list(path_targets["task_2_id"].keys())
        with open(path,"r") as the_file:
            path_best_model = json.load(the_file)
        path_best_model = {int(x):y for x, y in path_best_model.items()}        
        ## Fold
        path_fold = int(path.split("/")[-2].split("_fold-")[1])
        ## Identify Checkpoints
        for tid, tinfo in path_best_model.items():
            t_name = path_targets[tid]
            t_checkpoint = tinfo["steps"]
            t_path = path.replace("best_model.json",f"checkpoint-{t_checkpoint}/")
            model_paths[t_name][path_fold] = t_path
    ## Return
    return model_paths

def _plot_domain_results(domain, domain_results):
    """
    
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=False, sharey=False)
    cm = domain_results["confusion_matrix"].values
    cm_normed = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_normed, aspect="auto", cmap=plt.cm.Purples, alpha=0.4, interpolation="nearest")
    for r, row in enumerate(cm):
        for c, cell in enumerate(row):
            if cell == 0 or pd.isnull(cell):
                continue
            ax.text(c, r, "{:,d}".format(cell), ha="center", va="center", fontsize=8)
    for i in range(cm.shape[1]):
        ax.axvline(i - 0.5, color="black", alpha=0.4)
        ax.axhline(i - 0.5, color="black", alpha=0.5)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_xticklabels(domain_results["confusion_matrix"].columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(cm.shape[0]))
    ax.set_yticklabels(domain_results["confusion_matrix"].index.tolist())
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_title("Majority: {:.2f} +- {:.2f}\nKeyword: {:.2f} +- {:.2f}\nLabel: {:.2f} +- {:.2f}\nEncounter: {:.2f} +- {:.2f}\nKeyword x Label: {:.2f} +- {:.2f}\nKeyword x Encounter: {:.2f} +- {:.2f}\nLabel x Encounter: {:.2f} +- {:.2f}\nKeyword x Label x Encounter: {:.2f} +- {:.2f}\nModel: {:.2f} +- {:.2f}\nModel (Swapped): {:.2f} +- {:.2f}".format(
        domain_results["score"][:,0].mean(),
        domain_results["score"][:,0].std(),
        domain_results["score"][:,1].mean(),
        domain_results["score"][:,1].std(),
        domain_results["score"][:,2].mean(),
        domain_results["score"][:,2].std(),
        domain_results["score"][:,3].mean(),
        domain_results["score"][:,3].std(),
        domain_results["score"][:,4].mean(),
        domain_results["score"][:,4].std(),
        domain_results["score"][:,5].mean(),
        domain_results["score"][:,5].std(),
        domain_results["score"][:,6].mean(),
        domain_results["score"][:,6].std(),
        domain_results["score"][:,7].mean(),
        domain_results["score"][:,7].std(),
        domain_results["score"][:,8].mean(),
        domain_results["score"][:,8].std(),
        domain_results["score"][:,9].mean(),
        domain_results["score"][:,9].std()
    ), loc="left")
    fig.suptitle(f"Domain: {domain.title()}", fontweight="bold")
    fig.tight_layout()
    return fig

def _run_distribution_plot(distribution,
                           distribution_normed,
                           ax):
    """
    
    """
    ax.imshow(distribution_normed, aspect="auto", cmap=plt.cm.Purples, alpha=0.3, interpolation="nearest")
    for r, row in enumerate(distribution.values):
        for c, cell in enumerate(row):
            if pd.isnull(cell) or cell == 0:
                continue
            ax.text(c, r, "{:,d}".format(int(cell)), ha="center", va="center", fontsize=7)
    return ax

def run_distribution(annotations,
                     output_dir):
    """
    
    """
    ## Initialize Output Directory
    if not os.path.exists(output_dir):
        _ = os.makedirs(output_dir)
    ## Distributions
    for target in ["patient_gender","patient_race"]:
        for facet in [["keyword"],["label"],["encounter_type"],["label","keyword"],["encounter_type","keyword"],["label","encounter_type"],["label","encounter_type","keyword"]]:
            ## Count
            target_facet = annotations.groupby(facet + [target]).size().unstack()
            target_facet_norm = target_facet.apply(lambda row: row / row.sum(), axis=1)
            ncol = target_facet.shape[1]
            pid = "-".join(facet)
            ## Plot
            if len(facet) == 1:
                prows = target_facet.index
                fig, ax = plt.subplots(1, 1, figsize=(max(ncol * 0.5, 15), max(10, len(prows) * 0.25)))
                ax = _run_distribution_plot(target_facet, target_facet_norm, ax=ax)
                ax.set_xticks(range(ncol))
                ax.set_xticklabels(target_facet.columns, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(prows)))
                ax.set_yticklabels(prows, fontsize=7)
                ax.set_ylabel(facet[0], fontweight="bold")
                ax.set_xlabel(target, fontweight="bold")
            elif len(facet) == 2:
                scols = target_facet.index.levels[0]
                prows = target_facet.index.levels[1]
                fig, ax = plt.subplots(1, len(scols), figsize=(max(15, ncol * 0.5 * len(scols)), max(10, len(prows) * 0.25)), sharey=False)
                for c, col in enumerate(scols):
                    ax[c] = _run_distribution_plot(target_facet.loc[col].reindex(prows), target_facet_norm.loc[col].reindex(prows), ax[c])
                    ax[c].set_xticks(range(ncol))
                    ax[c].set_xticklabels(target_facet.columns, rotation=45, ha="right", fontsize=7)
                    ax[c].set_yticks(range(len(prows)))
                    if c == 0:
                        ax[c].set_yticklabels(prows, fontsize=7)
                        ax[c].set_ylabel(facet[0], fontweight="bold")
                    else:
                        ax[c].set_yticklabels([])
                    ax[c].set_title(col, fontweight="bold")
            elif len(facet) == 3:
                scols = target_facet.index.levels[0]
                srows = target_facet.index.levels[1]
                prows = target_facet.index.levels[2]
                fig, ax = plt.subplots(len(srows), len(scols), figsize=(max(15, ncol * 0.5 * len(scols)), max(10, len(prows) * 0.25 * len(srows))), sharex=False, sharey=False)
                for r, row in enumerate(srows):
                    for c, col in enumerate(scols):
                        if row in target_facet.loc[col].index:
                            ax[r,c] = _run_distribution_plot(target_facet.loc[col].loc[row].reindex(prows), target_facet_norm.loc[col].loc[row].reindex(prows), ax=ax[r,c])
                        else:
                            ax[r,c].imshow(np.zeros((len(prows), ncol)), cmap=plt.cm.Purples, alpha=0, interpolation="nearest", aspect="auto")
                        ax[r,c].set_xticks(range(ncol))
                        ax[r,c].set_yticks(range(len(prows)))
                        if r == len(srows) - 1:
                            ax[r,c].set_xticklabels(target_facet.columns, rotation=45, ha="right", fontsize=7)
                        else:
                            ax[r,c].set_xticklabels([])
                        if c == 0:
                            ax[r,c].set_yticklabels(prows, fontsize=7)
                            ax[r,c].set_ylabel(row, fontweight="bold")
                        else:
                            ax[r,c].set_yticklabels([])
                        if r == 0:
                            ax[r,c].set_title(col, fontweight="bold")
            else:
                raise NotImplementedError("Unexpected facet size.")
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{target}.{pid}.png", dpi=150)
            plt.close(fig)

def run_domain_discrimination(features,
                              features_swap,
                              annotations,
                              target_domains,
                              output_dir,
                              verbose=False):
    """
    
    """
    ## Initialize Output Directory
    if not os.path.exists(output_dir):
        _ = os.makedirs(output_dir)
    ## Identify Unique Patients
    patients = annotations["enterprise_mrn"].unique()
    ## Keyword
    kvec = model_utils.get_vectorizer(annotations["keyword"].unique())
    kfeatures = kvec.transform(annotations["keyword"].map(lambda i: {i:1}))
    ## Label
    lvec = model_utils.get_vectorizer(annotations["label"].unique())
    lfeatures = lvec.transform(annotations["label"].map(lambda i: {i:1}))
    ## Encounter
    evec = model_utils.get_vectorizer(annotations["encounter_type"].unique())
    efeatures = evec.transform(annotations["encounter_type"].map(lambda i: {i:1}))
    ## Keyword x Label
    klvec = model_utils.get_vectorizer(annotations[["keyword","label"]].apply(tuple, axis=1).unique())
    klfeatures = klvec.transform(annotations[["keyword","label"]].apply(tuple, axis=1).map(lambda i: {i:1}))
    klfeatures = sparse.hstack([kfeatures, lfeatures, klfeatures]).tocsr()
    ## Keyword x Encounter
    kevec = model_utils.get_vectorizer(annotations[["keyword","encounter_type"]].apply(tuple, axis=1).unique())
    kefeatures = kevec.transform(annotations[["keyword","encounter_type"]].apply(tuple, axis=1).map(lambda i: {i:1}))
    kefeatures = sparse.hstack([kfeatures, efeatures, kefeatures]).tocsr()
    ## Label x Encounter
    levec = model_utils.get_vectorizer(annotations[["label","encounter_type"]].apply(tuple, axis=1).unique())
    lefeatures = levec.transform(annotations[["label","encounter_type"]].apply(tuple, axis=1).map(lambda i: {i:1}))
    lefeatures = sparse.hstack([lfeatures, efeatures, lefeatures]).tocsr()
    ## Keyword X Label x Encounter
    klevec = model_utils.get_vectorizer(annotations[["keyword","label","encounter_type"]].apply(tuple,axis=1).unique())
    klefeatures = klevec.transform(annotations[["keyword","label","encounter_type"]].apply(tuple, axis=1).map(lambda i: {i:1}))
    klefeatures = sparse.hstack([kfeatures, lfeatures, efeatures, klefeatures]).tocsr()
    ## Generate Patient-level Splits
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(patients))
    ## Cache of Performance
    performance = {}
    performance_df = {}
    significance = {}
    special_cases = {t:[] for t in target_domains if t not in ["keyword","label","encounter_type"]}
    ## Domain Discrimination
    for target_dom in target_domains:
        if verbose:
            print(f"[Starting Domain Discrimination for '{target_dom}']")
        ## Get Target Attribute
        y = annotations[target_dom].values
        y_unique = sorted(set(y))
        y2id = {l:i for i, l in enumerate(y_unique)}
        y_encoded = np.array([y2id.get(i) for i in y])
        ## Run Cross Validation
        scores = np.zeros((5,10))
        cm = np.zeros((len(y_unique), len(y_unique)), dtype=int)
        for f, (train_, dev_) in enumerate(splits):
            ## Translate Patient IDs to Patients
            train_ = [patients[t] for t in train_]
            dev_ = [patients[t] for t in dev_]
            ## Translate Patients Indices to Instance Indices
            train_ind = annotations["enterprise_mrn"].isin(set(train_)).values.nonzero()[0]
            dev_ind = annotations["enterprise_mrn"].isin(set(dev_)).values.nonzero()[0]
            ## Get Subsets of Data
            x_train, x_dev = features[train_ind], features[dev_ind]
            xs_train, xs_dev = features_swap[train_ind], features_swap[dev_ind]
            k_train, k_dev = kfeatures[train_ind], kfeatures[dev_ind]
            l_train, l_dev = lfeatures[train_ind], lfeatures[dev_ind]
            e_train, e_dev = efeatures[train_ind], efeatures[dev_ind]
            kl_train, kl_dev = klfeatures[train_ind], klfeatures[dev_ind]
            ke_train, ke_dev = kefeatures[train_ind], kefeatures[dev_ind]
            le_train, le_dev = lefeatures[train_ind], lefeatures[dev_ind]
            kle_train, kle_dev = klefeatures[train_ind], klefeatures[dev_ind]
            y_train, y_dev = y_encoded[train_ind], y_encoded[dev_ind]
            ## Fit Dummy Classifier (Naive Baseline)
            dummy_discriminator = DummyClassifier(strategy="most_frequent")
            dummy_discriminator = dummy_discriminator.fit(x_train, y_train)
            ## Fit Keyword Classifier (Keyword Baseline)
            keyword_discriminator = ConditionalMajorityClassifier(1e-10)
            keyword_discriminator = keyword_discriminator.fit(k_train, y_train)
            ## Fit Label Classifier (Label Baseline)
            label_discriminator = ConditionalMajorityClassifier(1e-10)
            label_discriminator = label_discriminator.fit(l_train, y_train)
            ## Fit Encounter Classifier (Encounter Baseline)
            encounter_discriminator = ConditionalMajorityClassifier(1e-10)
            encounter_discriminator = encounter_discriminator.fit(e_train, y_train)
            ## Fit Keyword x Label Classifier (Keyword x Label Baseline)
            keyword_label_discriminator = LogisticRegression(penalty="none", max_iter=10000)
            keyword_label_discriminator = keyword_label_discriminator.fit(kl_train, y_train)
            ## Fit Keyword x Encounter Classifier (Keyword x Encounter Type Baseline)
            keyword_encounter_discriminator = LogisticRegression(penalty="none", max_iter=10000)
            keyword_encounter_discriminator = keyword_encounter_discriminator.fit(ke_train, y_train)
            ## Fit Label x Encounter Classifier
            label_encounter_discriminator = LogisticRegression(penalty="none", max_iter=10000)
            label_encounter_discriminator = label_encounter_discriminator.fit(le_train, y_train)
            ## Fit Keyword Classifier (Keyword x Label x Encounter Type Baseline)
            keyword_label_encounter_discriminator = LogisticRegression(penalty="none", max_iter=10000)
            keyword_label_encounter_discriminator = keyword_label_encounter_discriminator.fit(kle_train, y_train)
            ## Fit Data-informed Classifier
            domain_discriminator = LogisticRegression(penalty="none", max_iter=10000)
            domain_discriminator = domain_discriminator.fit(x_train, y_train)
            ## Fit Data-informed Classifier (Swapped Text)
            domain_discriminator_s = LogisticRegression(penalty="none", max_iter=10000)
            domain_discriminator_s = domain_discriminator_s.fit(xs_train, y_train)
            ## Make Predictions
            y_pred_dev_dummy = dummy_discriminator.predict(x_dev)
            y_pred_dev_keyword = keyword_discriminator.predict(k_dev)
            y_pred_dev_label = label_discriminator.predict(l_dev)
            y_pred_dev_encounter = encounter_discriminator.predict(e_dev)
            y_pred_dev_keyword_label = keyword_label_discriminator.predict(kl_dev)
            y_pred_dev_keyword_encounter = keyword_encounter_discriminator.predict(ke_dev)
            y_pred_dev_label_encounter = label_encounter_discriminator.predict(le_dev)
            y_pred_dev_keyword_label_encounter = keyword_label_encounter_discriminator.predict(kle_dev)
            y_pred_dev = domain_discriminator.predict(x_dev)
            y_pred_dev_s = domain_discriminator_s.predict(xs_dev)
            ## Model Correct Unique
            if target_dom in special_cases.keys():
                m_special = np.logical_and(y_pred_dev_dummy != y_dev, y_pred_dev_s == y_dev).nonzero()[0] ## Baseline = Incorrect, Swapped = Correct
                m_special_ind = [dev_ind[i] for i in m_special]
                if len(m_special_ind) > 0:
                    special_cases[target_dom].extend(annotations.iloc[m_special_ind][["note_text_swap_ids", target_dom]].values)
            ## Cache Dev Performance
            avg = "macro"
            cm += metrics.confusion_matrix(y_dev, y_pred_dev, labels=range(len(y_unique)))
            scores[f,:] = [metrics.f1_score(y_dev, y_pred_dev_dummy, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_keyword, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_label, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_encounter, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_keyword_label, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_keyword_encounter, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_label_encounter, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_keyword_label_encounter, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev, average=avg, zero_division=0),
                           metrics.f1_score(y_dev, y_pred_dev_s, average=avg, zero_division=0)]
        ## Format Confusion Matrix
        cm = pd.DataFrame(cm, index=y_unique, columns=y_unique)
        ## Display Performance
        if verbose:
            print(">> Validation Score (Majority): {:.2f} +- {:.2f}".format(scores[:,0].mean(), scores[:,0].std()))
            print(">> Validation Score (Keyword): {:.2f} +- {:.2f}".format(scores[:,1].mean(), scores[:,1].std()))
            print(">> Validation Score (Label): {:.2f} +- {:.2f}".format(scores[:,2].mean(), scores[:,2].std()))
            print(">> Validation Score (Encounter): {:.2f} +- {:.2f}".format(scores[:,3].mean(), scores[:,3].std()))
            print(">> Validation Score (Keyword x Label): {:.2f} +- {:.2f}".format(scores[:,4].mean(), scores[:,4].std()))
            print(">> Validation Score (Keyword x Encounter): {:.2f} +- {:.2f}".format(scores[:,5].mean(), scores[:,5].std()))
            print(">> Validation Score (Label x Encounter): {:.2f} +- {:.2f}".format(scores[:,6].mean(), scores[:,6].std()))
            print(">> Validation Score (Keyword x Label x Encounter): {:.2f} +- {:.2f}".format(scores[:,7].mean(), scores[:,7].std()))
            print(">> Validation Score (Discriminator): {:.2f} +- {:.2f}".format(scores[:,8].mean(), scores[:,8].std()))
            print(">> Validation Score (Discriminator Swapped): {:.2f} +- {:.2f}".format(scores[:,9].mean(), scores[:,9].std()))
            print(">> Validation Confusion Matrix:\n", cm.to_string())
        ## Significance Testing
        feature_sets = ["Majority","Anchor","Label","Encounter","Anchor x Label","Anchor x Encounter","Label x Encounter","Anchor x Label x Encounter","Embeddings","Embeddings (Gender Neutral)"]
        assert len(feature_sets) == scores.shape[1]
        sig_results = []
        for i, ifeat in enumerate(feature_sets):
            sig_results.append([])
            for j, jfeat in enumerate(feature_sets):
                if i != j:
                    sig_results[-1].append(stats.ttest_rel(scores[:,i], scores[:,j]).pvalue)
                else:
                    sig_results[-1].append(None)
        significance[target_dom] = pd.DataFrame(sig_results, index=feature_sets, columns=feature_sets).applymap(lambda i: "{:.4f}".format(i))
        ## Store Performance
        performance[target_dom] = {"score":scores, "confusion_matrix":cm}
        ## Plot Results
        fig = _plot_domain_results(domain=target_dom, domain_results=performance[target_dom])
        fig.savefig(f"{output_dir}/{target_dom}.png",dpi=300)
        plt.close(fig)
        ## Aggregate Results and Cache
        performance_df[target_dom] = pd.DataFrame([scores.mean(axis=0), scores.std(axis=0)],
                                                  index=["mean","std"],
                                                  columns=feature_sets).T
    ## Cache Significance
    with open(f"{output_dir}/significance.txt","w") as the_file:
        for target_dom, target_sig in significance.items():
            the_file.write(f">> Target: {target_dom}\n")
            the_file.write(target_sig.to_string()+"\n")
    ## Special Cases
    if verbose:
        for dom, dom_vals in special_cases.items():
            print(f"~~~~~~~~ Special Cases: {dom} ~~~~~~~~")
            for ex, ex_lbl in dom_vals:
                print(f"[{ex_lbl}] {ex}")
    ## Format Performance Df
    performance_df = pd.concat(performance_df,axis=0)
    performance_df["summary"] = performance_df.apply(lambda row: "{:.2f} +- {:.2f}".format(row["mean"],row["std"]), axis=1)
    performance_df = performance_df["summary"].unstack().T
    ## Cache
    _ = performance_df.to_csv(f"{output_dir}/scores.csv",index=True)
    ## Return
    return performance

def _run_keyword_subset_projection(features,
                                   annotations,
                                   keyword_subset,
                                   keyword_category_projection,
                                   target_domains,
                                   verbose=False):
    """
    
    """
    ## Isolate Relevant Keyword Subset
    keyword_subset_inds = (annotations["keyword"].isin(keyword_subset)).values.nonzero()[0]
    keyword_subset_annotations = annotations.iloc[keyword_subset_inds]
    if len(keyword_subset_inds) == 0:
        return None
    ## Fit Projection Using The Keyword Subset Alone
    if verbose:
        print("[Fitting Keyword Subset Projection]")
    keyword_subset_projector = UMAP(verbose=verbose, random_state=42, n_neighbors=50, min_dist=0.9, spread=1.0)
    keyword_subset_projection = keyword_subset_projector.fit_transform(features[keyword_subset_inds])
    ## Build the Projection Plot
    if verbose:
        print("[Generating Plot]")
    fig, ax = plt.subplots(2, len(target_domains), figsize=(max(15,len(target_domains)*3.5), 8), sharex=False, sharey=False)
    for p, (projection, projection_annotations) in enumerate(zip([keyword_category_projection, keyword_subset_projection],
                                                                 [annotations, keyword_subset_annotations])):
        for c, color in enumerate(target_domains):
            cunique = sorted(projection_annotations[color].unique())
            c_cunique = projection_annotations.loc[(projection_annotations["keyword"].isin(keyword_subset))][color].unique()
            if p == 0:
                ax[p, c].scatter(keyword_category_projection[:,0],
                                 keyword_category_projection[:,1],
                                 alpha=0.1,
                                 color="black",
                                 s=5)
            pbounds = [[np.inf, -np.inf],[np.inf,-np.inf]]
            for u, un in enumerate(cunique):
                if un not in c_cunique:
                    continue
                ind_mask = ((projection_annotations["keyword"].isin(keyword_subset))&
                            (projection_annotations[color]==un)).values.nonzero()[0]
                ax[p, c].scatter(projection[ind_mask,0],
                                projection[ind_mask,1],
                                alpha=0.3,
                                s=10,
                                marker="o",
                                color=f"C{u}",
                                label=un)
                if projection[ind_mask,0].min() < pbounds[0][0]:
                    pbounds[0][0] = projection[ind_mask,0].min()
                if projection[ind_mask,0].max() > pbounds[0][1]:
                    pbounds[0][1] = projection[ind_mask,0].max()
                if projection[ind_mask,1].min() < pbounds[1][0]:
                    pbounds[1][0] = projection[ind_mask,1].min()
                if projection[ind_mask,1].max() > pbounds[1][1]:
                    pbounds[1][1] = projection[ind_mask,1].max()
            # ax[p, c].set_xlim(pbounds[0][0]-0.5, pbounds[0][1]+0.5)
            # ax[p, c].set_ylim(pbounds[1][0]-0.5, pbounds[1][1]+0.5)
            leg = ax[p, c].legend(loc="best", title=color, fontsize=6)
            leg.get_title().set_fontsize('6')
            ax[p, c].spines["right"].set_visible(False)
            ax[p, c].spines["top"].set_visible(False)
            ax[p, c].set_xticklabels([])
            ax[p, c].set_yticklabels([])
    ax[0,0].set_title("Keyword Category Projections", loc="left", fontstyle="italic")
    ax[1,0].set_title("Keyword Subset Projections", loc="left", fontstyle="italic")
    fig.tight_layout()
    fig.suptitle("Keywords: {}".format(keyword_subset), y=0.95)
    fig.subplots_adjust(top=0.9)
    return fig

def run_projection(features,
                   annotations,
                   keyword_subsets,
                   target_domains,
                   output_dir,
                   verbose=False):
    """
    
    """
    ## Initialize Output Directory
    if not os.path.exists(output_dir):
        _ = os.makedirs(output_dir)
    if verbose:
        print("[Fitting Global Keyword Category Projection]")
    keyword_category_projector = UMAP(verbose=verbose, random_state=42, n_neighbors=50, min_dist=0.9, spread=1.0)
    keyword_category_projection = keyword_category_projector.fit_transform(features)
    ## Iterate Through Keyword Subsets
    if verbose:
        print("[Generating Keyword Subset Plots]")
    for keyword_subset in keyword_subsets:
        ## Format Output Name
        keyword_subset_clean = "-".join([i.replace(" ","_") for i in keyword_subset])
        ## Generate Figure
        fig = _run_keyword_subset_projection(features=features,
                                             annotations=annotations,
                                             keyword_subset=keyword_subset,
                                             keyword_category_projection=keyword_category_projection,
                                             target_domains=target_domains,
                                             verbose=verbose)
        if fig is None:
            continue
        fig.savefig(f"{output_dir}/{keyword_subset_clean}.png", dpi=300)
        plt.close(fig)

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Output Directory
    print("[Initializing Output Directory]")
    if os.path.exists(args.output_dir):
        _ = os.system("rm -rf {}".format(args.output_dir))
    _ = os.makedirs(args.output_dir)
    ## Model Paths
    print("[Identifying Encoder Model Paths]")
    model_paths = get_model_paths(model_root=args.model_root)
    ## Load Modeling Data/Annotations
    print("[Loading Raw Data/Annotations]")
    annotations = model_utils.prepare_model_data(dataset=args.dataset_id,
                                                 eval_original_context=10,
                                                 eval_target_context=10,
                                                 eval_rm_keywords=set([]))
    ## Encounter Type Override
    if args.dataset_id == "mimic-iv-discharge":
        annotations = annotations.drop("encounter_type",axis=1)
        annotations["encounter_type"] = annotations[MIMIC_IV_DIS_ENC_TYPE_COL]
    ## Drop Undesired Keywords
    if args.keywords_exclude is not None and len(args.keywords_exclude) > 0:
        print("[Excluding Keywords: {}]".format(args.keywords_exclude))
        annotations = annotations.loc[~annotations["keyword"].isin(args.keywords_exclude),:].reset_index(drop=True).copy()    
    ## Clean Up The Dataset
    print("[Cleaning Up Annotation Data]")
    unique_keycats = list(settings.CAT2KEYS.keys())
    annotations = annotations.loc[annotations["keyword_category"].isin(unique_keycats)]
    annotations = annotations.loc[~annotations["label"].isnull()].reset_index(drop=True).copy()
    ## Metadata Encoding
    print("[Encoding Metadata]")
    task2id = dict(zip(unique_keycats, range(len(unique_keycats))))
    label2id = annotations.groupby(["keyword_category"])["label"].unique().map(lambda i: dict(zip(i, range(len(i))))).to_dict()
    annotations["task_id"] = annotations["keyword_category"].map(task2id.get)
    annotations["label_id"] = annotations.apply(lambda row: label2id[row["keyword_category"]][row["label"]], axis=1)    
    ## Update Races
    print("[Consolidating Races]")
    annotations["patient_race"] = annotations["patient_race"].map(lambda i: "Other" if i in args.race_other else i)
    ## Remove Unknown Demographic Identifiers
    print("[Filtering Race and Sex]")
    annotations = annotations.loc[~annotations["patient_race"].isin(args.race_ignore),:]
    annotations = annotations.loc[~annotations["patient_gender"].isin(args.sex_ignore),:]
    ## Pronound Swap
    print("[Swapping Identifiers]")
    annotations["note_text_swap_ids"] = annotations["note_text"].map(swap_identifiers)
    ## Load Tokenizer
    print("[Initializing Tokenizer]")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    ## Training Isolation
    if args.training_only:
        print("[Isolating Patients From Training Split]")
        if max(len(models) for _, models in model_paths.items()) > 1:
            print(">> WARNING: Training patients will not align perfectly with multiple folds.")
        annotations = annotations.loc[annotations["split"]=="train",:]
    ## Iterate Through Keyword Groups
    print("[Beginning Analysis and Modeling]")
    for k, keyword_category in enumerate(["adamant","compliance","other"]):
        print("[Beginning Keyword Category {}/{}: {}]".format(k+1, len(unique_keycats), keyword_category))
        ## Isolate Subset of Annotations for Keyword Category
        cat_annotations = annotations.loc[annotations["keyword_category"]==keyword_category].reset_index(drop=True)
        ## Distribution
        print("[Plotting Distributions]")
        _ = run_distribution(annotations=cat_annotations,
                             output_dir=f"{args.output_dir}/{keyword_category}/")
        ## BERT Tokenization
        if args.verbose:
            print("[Tokenizing for BERT]")
        tokens, token_masks = bert_utils.tokenize_and_mask(text=cat_annotations["note_text"].tolist(),
                                                           keywords=cat_annotations["keyword"].tolist(),
                                                           tokenizer=tokenizer,
                                                           mask_type="keyword_all")
        tokens_swap, token_masks_swap = bert_utils.tokenize_and_mask(text=cat_annotations["note_text_swap_ids"].tolist(),
                                                           keywords=cat_annotations["keyword"].tolist(),
                                                           tokenizer=tokenizer,
                                                           mask_type="keyword_all")
        ## Iterate Through Encoders
        for fold, mpath in model_paths[keyword_category].items():
            print("[Beginning Fold {}/{}]".format(fold+1, len(model_paths[keyword_category])))
            ## Load Encoder
            if args.verbose:
                print("[Initializing Encoder]")
            minit = torch.load(f"{mpath}/init.pth", map_location=torch.device("cpu"))
            model = bert_utils.BERTMultitaskClassifier(task_targets=minit["task_targets"],
                                                       use_bert_pooler=minit.get("use_bert_pooler",False),
                                                       checkpoint=minit["checkpoint"],
                                                       p_dropout=minit["settings"]["p_dropout"])
            _ = model.load_state_dict(torch.load(f"{mpath}/model.pth",torch.device("cpu")), strict=False)
            model = model.to(args.device)
            ## Encode Dataset
            dataset = bert_utils.ClassificationTokenDataset(tokens=tokens,
                                                            token_masks=token_masks,
                                                            labels=cat_annotations["label_id"].tolist(),
                                                            task_ids=cat_annotations["task_id"].tolist(),
                                                            device=args.device)
            dataset_encoded = bert_utils.encode_dataset(dataset=dataset,
                                                        bert=model._bert,
                                                        batch_size=32,
                                                        device=args.device)
            dataset_swap = bert_utils.ClassificationTokenDataset(tokens=tokens_swap,
                                                                 token_masks=token_masks_swap,
                                                                 labels=cat_annotations["label_id"].tolist(),
                                                                 task_ids=cat_annotations["task_id"].tolist(),
                                                                 device=args.device)
            dataset_swap_encoded = bert_utils.encode_dataset(dataset=dataset_swap,
                                                             bert=model._bert,
                                                             batch_size=32,
                                                             device=args.device)

            ## Stack Encoding
            if args.verbose:
                print("[Stacking Encoded Dataset]")
            dataset_encoded = torch.stack([d["data"] for d in dataset_encoded]).detach().to("cpu").numpy()
            dataset_swap_encoded = torch.stack([d["data"] for d in dataset_swap_encoded]).detach().to("cpu").numpy()
            ## Domain Discrimination Experiments
            print("[Running Domain Discrimination Experiments]")
            _ = run_domain_discrimination(features=dataset_encoded,
                                          features_swap=dataset_swap_encoded,
                                          annotations=cat_annotations,
                                          target_domains=args.target_domains,
                                          output_dir=f"{args.output_dir}/{keyword_category}/{fold}/discrimination/",
                                          verbose=args.verbose)
            ## Projection
            print("[Running Projection Experiments]")
            _ = run_projection(features=dataset_encoded,
                               annotations=cat_annotations,
                               keyword_subsets=KEYWORD_ANALYSIS_GROUPS[keyword_category],
                               target_domains=args.target_domains,
                               output_dir=f"{args.output_dir}/{keyword_category}/{fold}/projections/",
                               verbose=args.verbose)
    ## Done
    print("[Script Complete]")

######################
### Execute
######################

if __name__ == "__main__":
    _ = main()