
"""
Aggregate performance outcomes across multiple training runs, outputing
a set of simple summary visualizations. Use configuration section
to specify runs to aggregate.
"""

###########################
### Custom Configuration
###########################

## General Formatting
OUTPUT_DIR = f"data/results/model-comparison/mimic-iv/camera-ready/final-cv/"
SCORES = {
    ## Baseline Examples
    f"data/results/model/mimic-iv/camera-ready/final-cv/baseline-majority/scores.json":{
        "Majority Overall":{"model_type":"majority","feature_set":"keyword"},
    },
    f"data/results/model/mimic-iv/camera-ready/final-cv/baseline-statistical/scores.json":{
        "Majority Per Anchor":{"model_type":"linear","feature_set":"keyword"},
        "LR (Context)":{"model_type":"linear","feature_set":"tokens_tfidf"},
        "LR (Anchor + Context)":{"model_type":"linear","feature_set":"keyword_tokens_tfidf"},
    },
    # ## Bert Example
    # f"PREFIX_PATH/enhanced-base_bert-base-separate-full/scores.json":{
    #     "BERT (Web) - Anchor":{"model_type":"bert","feature_set":"bert"}
    # },
}

###########################
### Imports
###########################

## Standard Library
import os
import json
import argparse

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats as stats

## Private Library
from stigma.model.eval import std_error

###########################
### Functions
###########################

def parse_command_line():
    """

    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--plot_splits", type=str, nargs="+", choices={"train","dev","test"}, default=["train","dev","test"])
    args = parser.parse_args()
    return args

def load_scores(score_dict):
    """
    
    """
    ## Load Scores
    score_df = []
    score_df_full = []
    for sf, sf_details in score_dict.items():
        ## Reverse Details
        sf_details_r = {(y.get("model_type"),y.get("feature_set")):x for x, y in sf_details.items()}
        ## Read Data and Filter
        with open(sf, "r") as the_file:
            for line in the_file:
                ## Parse
                line_data = json.loads(line)
                ## Get Result Key
                line_data_mf = (line_data.get("model_type"), line_data.get("feature_set"))
                ## Check Relevance
                if line_data_mf not in sf_details_r:
                    continue
                ## Add Identifier (Clean)
                line_data["identifier"] = sf_details_r[line_data_mf]
                ## Expanded Score DataFrame
                if line_data["metric"] == "confusion":
                    line_data_full = confusion_to_score(line_data["score"])
                    for sname, sval in line_data_full.items():
                        score_df_full.append({**line_data, **{"score":sval, "metric":sname}})
                ## Cache
                score_df.append(line_data)
    ## Format
    score_df = pd.DataFrame(score_df)
    score_df_full = pd.DataFrame(score_df_full)
    return score_df, score_df_full

def confusion_to_score(confusion_dict):
    """
    
    """
    ## Build Out Predictions
    y_true = []
    y_pred = []
    for pred_lbl, pred_dict in confusion_dict.items():
        pred_lbl_f = pred_lbl[5:]
        for true_lbl, count in pred_dict.items():
            true_lbl_f = true_lbl[5:]
            for i in range(count):
                y_true.append(true_lbl_f)
                y_pred.append(pred_lbl_f)
    ## Unique Labels
    lbl_set = sorted(set(y_true))
    ## Score
    scores = {
        "accuracy":metrics.accuracy_score(y_true, y_pred),
        "f1_score_macro":metrics.f1_score(y_true, y_pred, labels=lbl_set, average="macro", zero_division=0),
        "f1_score_micro":metrics.f1_score(y_true, y_pred, labels=lbl_set, average="micro", zero_division=0),
        "recall_macro":metrics.recall_score(y_true, y_pred, labels=lbl_set, average="macro", zero_division=0),
        "recall_micro":metrics.recall_score(y_true, y_pred, labels=lbl_set, average="micro", zero_division=0),
        "precision_macro":metrics.precision_score(y_true, y_pred, labels=lbl_set, average="macro", zero_division=0),
        "precision_micro":metrics.precision_score(y_true, y_pred, labels=lbl_set, average="micro", zero_division=0),
    }
    return scores

def get_identifiers(score_dict):
    """
    
    """
    identifiers = [list(v.keys()) for s, v in score_dict.items()]
    identifiers = [x for s in identifiers for x in s]
    return identifiers

def plot_general(score_dict,
                 score_df,
                 keyword_category,
                 split="dev"):
    """
    
    """
    ## Get Split and Category
    plot_df = score_df.loc[(score_df["keyword_category"]==keyword_category)&
                           (score_df["split"]==split)&
                           ~(score_df["metric"].isin(["confusion","roc_auc"]))]
    ## Check For Missing Data
    if plot_df.shape[0] == 0:
        return None
    ## Format Subset
    plot_df = plot_df.copy()
    plot_df["score"] = plot_df["score"].astype(float)
    ## Relevant Attributes
    pmetrics = ["accuracy","f1","precision","recall"]
    identifiers = get_identifiers(score_dict)
    classes = ["overall"] + [c for c in plot_df["class"].unique() if c != "overall"]
    ## Aggregation
    plot_df_agg = plot_df.groupby(["metric","class","identifier"]).agg({"score":[np.mean, np.std, std_error],"support":sum})
    ## Plotting
    fig, ax = plt.subplots(1, len(pmetrics), figsize=(4.5*len(pmetrics), 3.5), sharex=True, sharey=True)
    for m, met in enumerate(pmetrics):
        for c, cls in enumerate(classes):
            if met == "accuracy" and cls != "overall":
                continue
            bwidth = 0.95 / len(identifiers)
            for i, idn in enumerate(identifiers):
                imc_data = plot_df_agg.loc[met].loc[cls].loc[idn]
                ax[m].bar(0.025 + i * bwidth + c,
                          imc_data["score"]["mean"],
                          yerr=imc_data["score"]["std"],
                          color=f"C{i}",
                          alpha=0.5,
                          capsize=2,
                          width=bwidth,
                          align="edge",
                          label=idn if c == 0 else None)
                if imc_data["score"]["mean"] > 0:
                    ax[m].text(0.025 + c + i * bwidth + bwidth/2,
                               imc_data["score"]["mean"] / 2,
                               "{:.2f}".format(imc_data["score"]["mean"]),
                               ha="center",
                               va="center",
                               fontsize=2)
        ax[m].spines["right"].set_visible(False)
        ax[m].spines["top"].set_visible(False)
        ax[m].set_xticks(np.arange(len(classes))+0.5)
        ax[m].set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        if m == 0:
            ax[m].set_ylabel("Score", fontweight="bold", fontsize=12)
        ax[m].set_title(met.title(), fontweight="bold", fontsize=8)
        if m == 0:
            ax[m].legend(loc="upper right", fontsize=4, frameon=False)
    fig.tight_layout()
    return fig

def plot_roc_confusion(score_dict,
                       score_df,
                       keyword_category,
                       split="dev"):
    """
    
    """
    ## Get Split and Category
    plot_df = score_df.loc[(score_df["keyword_category"]==keyword_category)&
                           (score_df["split"]==split)&
                           (score_df["metric"].isin(["confusion","roc_auc"]))]
    ## Check For Missing Data
    if plot_df.shape[0] == 0:
        return None
    ## Attributes
    identifiers = get_identifiers(score_dict)
    classes = [c for c in plot_df["class"].unique() if c != "overall"]
    ## Make Plots
    fig, ax = plt.subplots(2, len(identifiers), figsize=(4.5 * len(identifiers), 7), sharex=False, sharey=False)
    for i, idn in enumerate(identifiers):
        ## Get Confusion Matrix
        idn_cm = plot_df.loc[(plot_df["identifier"]==idn)&(plot_df["metric"]=="confusion")]
        idn_cm = sum(list(map(pd.DataFrame, idn_cm["score"].tolist())))
        idn_cm = idn_cm.loc[[f"true_{c}" for c in classes],[f"pred_{c}" for c in classes]]
        ## Normalize
        idn_cm_norm = idn_cm.apply(lambda row: row / sum(row), axis=1)
        ## Plot Confusion Matrix
        ax[0,i].imshow(idn_cm_norm.values,
                       cmap=plt.cm.Purples,
                       alpha=0.5,
                       aspect="auto",
                       interpolation="nearest")
        for r, row in enumerate(idn_cm.values):
            for c, cell in enumerate(row):
                if cell == 0:
                    continue
                ax[0,i].text(c, r, int(cell), ha="center", va="center", fontsize=12)
        ## Ticks
        if i == 0:
            ax[0,i].set_yticks(range(len(classes)))
            ax[0,i].set_yticklabels(classes)
        else:
            ax[0,i].set_yticks(range(len(classes)))
            ax[0,i].set_yticklabels([])
        ax[0,i].set_xticks(range(len(classes)))
        ax[0,i].set_xticklabels(classes, rotation=45, ha="right")
        ax[0,i].tick_params(labelsize=8)
        ## Identifier
        ax[0,i].set_title(idn, fontweight="bold", fontsize=12)
        ## Axes Labels
        if i == 0:
            ax[0,i].set_ylabel("True")
        ax[0,i].set_xlabel("Predicted")
        ## ROC/AUC
        ax[1,i].plot([0,1],[0,1],color="black",linestyle="--",label="Random",alpha=0.5)
        for c, cls in enumerate(classes):
            idn_roc = plot_df.loc[(plot_df["identifier"]==idn)&(plot_df["class"]==cls)&(plot_df["metric"]=="roc_auc")]
            idn_auc = [curve["auc"] for curve in idn_roc["score"].values if not pd.isnull(curve["auc"])]
            for cv, curve in enumerate(idn_roc["score"].tolist()):
                ax[1,i].plot(curve["fpr"],
                             curve["tpr"],
                             color=f"C{c}",
                             label=None if cv != idn_roc.shape[0] - 1 else "{} [AUC={:.3f} +/- {:.3f}]".format(cls, np.mean(idn_auc), np.std(idn_auc)),
                             alpha=0.5)
        ax[1,i].legend(loc="lower right", frameon=False, fontsize=6)
        ax[1,i].spines["right"].set_visible(False)
        ax[1,i].spines["top"].set_visible(False)
        if i == 0:
            ax[1,i].set_ylabel("True Positive Rate")
        ax[1,i].set_xlabel("False Positive Rate")
        ax[1,i].set_xlim(0,1)
        ax[1,i].set_ylim(0,1)
    fig.tight_layout()
    return fig

def pairwise_pvalue(score_dict, score_df_full, plot_splits):
    """
    
    """
    ## Aggregate and Sort
    score_df_full_c = score_df_full.copy()
    score_df_full_c["score_alt"] = score_df_full_c.apply(lambda row: (row["fold"], row["score"]),axis=1)
    score_list = score_df_full_c.groupby(["split","metric","keyword_category","identifier"]).agg({"score_alt":lambda x: [j[1] for j in sorted(x, key=lambda i: i[0])]})
    ## Get Relevant Identifiers
    score_ids = get_identifiers(score_dict)
    ## Iterate Through Splits
    out_str = []
    for split in plot_splits:
        ## Iterate Through Metrics
        for met in ["accuracy","f1_score_macro","f1_score_micro"]:
            ## Isolate Relevant Data
            out_str.append(f"***** {split} - {met} ****")
            split_met_vals = score_list.loc[(split, met)]
            keyword_cats = split_met_vals.index.levels[0]
            ## Compute Pairwise P-Values
            all_kc_pvals = {}
            for kc in keyword_cats:
                kc_pval = np.ones((len(score_ids), len(score_ids)))
                for i, ii in enumerate(score_ids):
                    for j, jj in enumerate(score_ids):
                        if i == j:
                            continue
                        xx = split_met_vals.loc[(kc,ii)].item()
                        yy = split_met_vals.loc[(kc,jj)].item()
                        kc_pval[i,j] = stats.ttest_rel(xx, yy).pvalue
                kc_pval = pd.DataFrame(kc_pval, index=score_ids, columns=score_ids)
                all_kc_pvals[kc] = kc_pval
            ## Merge and Format
            all_kc_pvals = pd.concat(all_kc_pvals)
            all_kc_pvals = all_kc_pvals.applymap(lambda i: "{:.2f}".format(i))
            ## Display for Each Keyword
            for keyword in all_kc_pvals.index.levels[0]:
                out_str.append(keyword)
                out_str.append(all_kc_pvals.loc[keyword].to_string())
    out_str = "\n".join(out_str)
    print(out_str)
    return out_str

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Setup
    print("[Identifying Parameters]")
    output_dir_, score_dict_ = OUTPUT_DIR, SCORES
    ## Output Directory
    print("[Initializing Output Directory]")
    if os.path.exists(output_dir_):
        _ = os.system("rm -rf {}".format(output_dir_))
    _ = os.makedirs(output_dir_)
    ## Load Scores
    print("[Loading Scores]")
    score_df, score_df_full = load_scores(score_dict_)
    ## Information
    print("[Gathering Split Information]")
    splits = score_df.groupby(["keyword_category"])["split"].unique()
    ## Generate Plots
    print("[Making Plots]")
    for keycat, keycat_splits in splits.items():
        for split in keycat_splits:
            ## Skip Undesired Splits
            if split not in args.plot_splits:
                continue
            ## General Plot
            fig = plot_general(score_dict_, score_df, keycat, split)
            if fig is not None:
                fig.savefig(f"{output_dir_}/general.{keycat}.{split}.png",dpi=300)
                plt.close(fig)
            ## Confusion Matrices/ROC
            fig = plot_roc_confusion(score_dict_, score_df, keycat, split)
            if fig is not None:
                fig.savefig(f"{output_dir_}/roc-confusion.{keycat}.{split}.png",dpi=300)
                plt.close(fig)
    ## Summary Computation
    print("[Creating Summary Dataframe]")
    sum_df_agg = pd.pivot_table(score_df_full,
                                index=["split","metric","identifier"],
                                columns="keyword_category",
                                values="score",
                                aggfunc=[np.mean,np.std])
    sum_df_agg.columns = sum_df_agg.columns.swaplevel(0, 1)
    sum_df_agg.sort_index(axis=1, level=0, inplace=True)
    sum_df_agg = sum_df_agg.applymap(lambda i: "{:.2f}".format(i))
    ## Display summary
    print("[Formatting Performance Summary]")
    identifiers_ordered = get_identifiers(score_dict_)
    out_str = []
    for split in args.plot_splits:
        for met in ["accuracy","f1_score_macro","f1_score_micro"]:
            ssub = sum_df_agg.loc[split].loc[met].loc[identifiers_ordered].to_string()
            out_str.append("~" * 100)
            out_str.append(f"Split: {split} || Metric: {met}")
            out_str.append("~" * 100)            
            out_str.append(ssub)
    out_str = "\n".join(out_str)
    ## Cache
    print("[Caching Performance Summary]")
    print(out_str)
    with open(f"{output_dir_}/summary.txt","w") as the_file:
        the_file.write(out_str)
    ## P-Values
    print("[Computing Pairwise Performance Differences]")
    out_str = pairwise_pvalue(score_dict_, score_df_full, plot_splits=args.plot_splits)
    with open(f"{output_dir_}/t-tests.txt","w") as the_file:
        the_file.write(out_str)
    ## Done
    print("[Script Complete]")

###########################
### Execution
###########################

if __name__ == "__main__":
    _ = main()