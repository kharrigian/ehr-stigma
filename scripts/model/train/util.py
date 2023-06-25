
"""
Model train/evaluation functions
"""

###################
### Imports
###################

## Standard Library
import os
import sys

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Private
from stigma import settings
from stigma.model import eval as model_eval

####################
### Data Processing
####################

def create_flat_score_df(category_scores, mtypes, ftypes):
    """
    
    """
    ## Metrics
    class_metrics = ["precision","recall","f1-score","roc_auc"]; class_support = "support"
    overall_metrics = ["precision_macro","recall_macro","f1_macro","accuracy","confusion"]
    ## Flatten Scores
    category_scores_flat = []
    for m, mt in enumerate(mtypes):
        for f, ft in enumerate(ftypes):
            ## Get Classes
            mf_res = list(filter(lambda s: s["model_type"]==mt and s["feature_set"]==ft, category_scores))
            ## Classes
            mf_classes = sorted(set.union(*[set(mfr["per_class"].keys()) for mfr in mf_res]))
            ## Cache
            for mrf in mf_res:
                ## Support
                mrf_support = [mrf["per_class"].get(x,{}).get(class_support,0) for x in mf_classes]
                mrf_support = [sum(mrf_support), *mrf_support]
                ## Overall Performance
                for o, om in enumerate(overall_metrics):
                    category_scores_flat.append({
                        "model_type":mt,
                        "feature_set":ft,
                        "target":mrf["target"],
                        "keyword_category":mrf["keyword_category"],
                        "fold":mrf["fold"],
                        "split":mrf["split"],
                        "learning_curve_train_proportion":mrf["learning_curve_train_proportion"],
                        "learning_curve_train_iteration":mrf["learning_curve_train_iteration"],
                        "learning_curve_train_size":mrf["learning_curve_train_size"],
                        "metric":om,
                        "class":"overall",
                        "score":mrf[om] if om != "confusion" else mrf[om].to_dict(),
                        "support":mrf_support[0],
                    })
                ## Class-specific
                for c, cls in enumerate(mf_classes):
                    for cm, cmet in enumerate(class_metrics):
                        if cmet != "roc_auc":
                            _scores = {"score":mrf["per_class"].get(cls, {}).get(cmet, np.nan)}
                        else:
                            _scores = {"score":mrf["per_class"].get(cls, {}).get("roc",{})}
                        category_scores_flat.append({
                            "model_type":mt,
                            "feature_set":ft,
                            "target":mrf["target"],
                            "keyword_category":mrf["keyword_category"],
                            "fold":mrf["fold"],
                            "split":mrf["split"],
                            "learning_curve_train_proportion":mrf["learning_curve_train_proportion"],
                            "learning_curve_train_iteration":mrf["learning_curve_train_iteration"],
                            "learning_curve_train_size":mrf["learning_curve_train_size"],
                            "metric":cmet,
                            "class":cls,
                            "support":mrf_support[c+1],
                            **_scores
                        })
    ## Format
    category_scores_flat = pd.DataFrame(category_scores_flat)
    category_scores_flat["metric"] = category_scores_flat["metric"].map(lambda i:{"f1_macro":"f1","f1-score":"f1","recall_macro":"recall","precision_macro":"precision"}.get(i, i))
    return category_scores_flat

####################
### Caching
####################

def cache_all_errors(errors,
                     output_dir):
    """
    
    """
    ## Output Directory
    if not os.path.exists(f"{output_dir}/errors/"):
        _ = os.makedirs(f"{output_dir}/errors/")
    ## Format Errors
    format_errors = lambda x: pd.concat(x).reset_index(drop=True) if isinstance(x, list) else x
    errors = {x:format_errors(dfs) for x, dfs in errors.items() if len(dfs) > 0}
    ## Cache Errors By Group
    for keycat, keycat_errors in errors.items():
        for (target, feature_set, model_type, split), inds in keycat_errors.groupby(["target","feature_set","model_type","split"]).groups.items():
            keycat_errors_cache = keycat_errors.loc[inds].drop(["feature_set","target","model_type","split"],axis=1).reset_index(drop=True)
            _ = keycat_errors_cache.to_csv(f"{output_dir}/errors/{keycat}.{target}.{feature_set}.{model_type}.{split}.csv", index=False)

def cache_scores(scores,
                 output_dir,
                 model_settings,
                 is_bert=False):
    """
    
    """
    ## Buld DataFrame
    scores_df = []
    for _, keyword_scores in scores.items():
        scores_df.append(create_flat_score_df(keyword_scores,
                                              mtypes=model_settings["models"] if not is_bert else ["bert"],
                                              ftypes=list(model_settings["feature_sets"].keys()) if not is_bert else ["bert"]))
    scores_df = pd.concat(scores_df, axis=0).reset_index(drop=True)
    ## Cache
    with open(f"{output_dir}/scores.json","w") as the_file:
        for _, row in scores_df.iterrows():
            _ = the_file.write(f"{row.to_json()}\n")

####################
### Plotting (Individual)
####################

def plot_label_bias(annotations,
                    feature,
                    targets,
                    row_stratify="keyword_category",
                    return_data=False):
    """
    
    """
    ## Mapping
    categories = annotations[row_stratify].unique()
    cat2features = annotations.groupby([row_stratify])[feature].unique().map(sorted)
    ## Data Cache
    data_cache = []
    ## Generate Plot
    fig, ax = plt.subplots(len(categories), len(targets), figsize=(4 * len(targets), 5 * len(categories)), sharex=False, sharey=False)
    for c, cat in enumerate(categories):
        for t, target in enumerate(targets):
            ## Get Axis
            if len(targets) == 1 and len(categories) == 1:
                pax = ax
            elif len(targets) == 1:
                pax = ax[c]
            elif len(categories) == 1:
                pax = ax[t]
            else:
                pax = ax[c, t]
            ## Get Data
            tc_df = annotations.loc[annotations[row_stratify]==cat].groupby([feature,target]).size().unstack()
            tc_df = tc_df.reindex(cat2features[cat]).fillna(0)
            ## Add Overall
            tc_df = pd.concat([tc_df.sum(axis=0).to_frame("overall").T, tc_df], axis=0)
            ## Cache
            if return_data:
                data_cache.append([cat, target, tc_df])
            ## Re-index and Normalize
            tc_df_norm = tc_df.apply(lambda x: x / sum(x), axis=1)
            ## Make Plot
            _ = pax.imshow(tc_df_norm, aspect="auto", cmap=plt.cm.Purples, alpha=0.5)
            for i, row in enumerate(tc_df.values):
                for j, cell in enumerate(row):
                    if cell == 0:
                        continue
                    pax.text(j, i, int(cell), ha="center", va="center", fontsize=4)
            ## Format
            pax.set_xticks(range(tc_df.shape[1]))
            pax.set_xticklabels(tc_df.columns.tolist(), ha="right", rotation=45, fontsize=4)
            if c == 0:
                pax.set_title(target, fontweight="bold")
            pax.set_yticks(range(tc_df.shape[0]))
            if t == 0:
                pax.set_yticklabels(tc_df.index.tolist(), fontsize=4)
                pax.set_ylabel(cat, fontweight="bold")
            else:
                pax.set_yticklabels([])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.275)
    if return_data:
        return fig, data_cache
    return fig

def plot_confusion_matrices(category_scores, mtypes, ftypes):
    """
    
    """
    ## Plot
    fig, ax = plt.subplots(len(mtypes), len(ftypes), figsize=(len(ftypes)*4, len(mtypes)*3), sharex=True, sharey=True)
    for m, mt in enumerate(mtypes):
        for f, ft in enumerate(ftypes):
            ## Get Axis
            if len(mtypes) == 1 and len(ftypes) == 1:
                pax = ax
            elif len(mtypes) == 1:
                pax = ax[f]
            elif len(ftypes) == 1:
                pax = ax[m]
            else:
                pax = ax[m, f]
            ## Get Result
            mf_res = list(filter(lambda s: s["model_type"]==mt and s["feature_set"]==ft and s["learning_curve_train_proportion"]==100, category_scores))
            ## Get Confusion
            mf_res_cm = sum([mrf["confusion"] for mrf in mf_res])
            ## Sort Confusion for Consistency
            mf_res_cm = mf_res_cm.loc[sorted(mf_res_cm.index),sorted(mf_res_cm.columns)]
            ## Normalize
            mf_res_cm_norm = np.divide(mf_res_cm.values,
                                       mf_res_cm.values.sum(axis=1,keepdims=True),
                                       where=mf_res_cm.values.sum(axis=1,keepdims=True)>0,
                                       out=np.zeros_like(mf_res_cm.values.astype(float)))
            ## Plot Confusion
            pax.imshow(mf_res_cm_norm, aspect="auto", cmap=plt.cm.Purples, alpha=0.5)
            for r, row in enumerate(mf_res_cm.values):
                for c, cell in enumerate(row):
                    if cell == 0:
                        continue
                    pax.text(c, r, cell, ha="center", va="center")
            ## Formatting
            if f == 0:
                pax.set_ylabel(mt, fontweight="bold")
            if m == 0:
                pax.set_title(ft, fontweight="bold")
            pax.set_xticks(list(range(mf_res_cm.shape[1])))
            pax.set_xticklabels(mf_res_cm.columns.tolist(), rotation=45, ha="right")
            pax.set_yticks(list(range(mf_res_cm.shape[0])))
            pax.set_yticklabels(mf_res_cm.index.tolist())
    fig.tight_layout()
    return fig

def plot_roc_curves(category_scores, mtypes, ftypes):
    """
    
    """
    ## Make Plot
    fig, ax = plt.subplots(len(mtypes), len(ftypes), figsize=(len(ftypes)*4, len(mtypes)*3), sharex=True, sharey=True)
    for m, mt in enumerate(mtypes):
        for f, ft in enumerate(ftypes):
            ## Axis
            if len(mtypes) == 1 and len(ftypes) == 1:
                pax = ax
            elif len(mtypes) == 1:
                pax = ax[f]
            elif len(ftypes) == 1:
                pax = ax[m]
            else:
                pax = ax[m, f]
            ## Threshold
            pax.plot([0,1],[0,1],linestyle="--",alpha=0.4,color="black",label="Random")
            ## Get Result
            mf_res = list(filter(lambda s: s["model_type"]==mt and s["feature_set"]==ft and s["learning_curve_train_proportion"]==100, category_scores))
            ## Classes
            mf_classes = sorted(set.union(*[set(mfr["per_class"].keys()) for mfr in mf_res]))
            ## Plot ROC Curves
            for cc, cls in enumerate(mf_classes):
                cf_cls_scores = []
                for mm, mfr in enumerate(mf_res):
                    ## Get Scores
                    x = mfr["per_class"][cls]["roc"]["fpr"]
                    y = mfr["per_class"][cls]["roc"]["tpr"]
                    auc = mfr["per_class"][cls]["roc"]["auc"]
                    ## Cache
                    cf_cls_scores.append(auc)
                    ## Label
                    label = None if mm != len(mf_res) - 1 else "{} [AUC={:.2f} (+/- {:.2f})]".format(cls, np.nanmean(cf_cls_scores), model_eval.std_error(cf_cls_scores))
                    pax.plot(x, y, color=f"C{cc}", linestyle="-", alpha=0.4, label=label)
            ## Format
            pax.legend(loc="lower right", fontsize=6)
            pax.spines["right"].set_visible(False)
            pax.spines["top"].set_visible(False)
            pax.set_xlim(0, 1)
            pax.set_ylim(0, 1)
            if m == len(mtypes) - 1:
                pax.set_xlabel("False Positive Rate", fontweight="bold")
            if m == 0:
                pax.set_title(ft, fontweight="bold")
            if f == 0:
                pax.set_ylabel("{}\nTrue Positive Rate".format(mt), fontweight="bold")
    fig.tight_layout()
    return fig

def plot_performance_metrics_comparison(category_scores, mtypes, ftypes):
    """
    
    """
    ## Helpers
    float_formatter = lambda number: "{:.2f}".format(number)
    ## Get Score Dataframe
    category_scores_flat = create_flat_score_df(category_scores, mtypes, ftypes)
    ## Drop Duplicate of Majority Baseline (Since They All Give Same Outcome)
    mt_compare = []
    maj_seen = False
    for m in mtypes:
        for f in ftypes:
            if m == "majority" and maj_seen:
                continue
            elif m == "majority" and not maj_seen:
                maj_seen = True
            mt_compare.append((m, f))
    category_scores_flat = pd.concat([category_scores_flat.loc[(category_scores_flat["model_type"]==m)&
                                                               (category_scores_flat["feature_set"]==f)&
                                                               (category_scores_flat["metric"]!="roc_auc")&
                                                               (category_scores_flat["metric"]!="confusion")&
                                                               (category_scores_flat["learning_curve_train_proportion"]==100)] for m, f in mt_compare])
    category_scores_flat["score"] = category_scores_flat["score"].astype(float)
    ## Aggregate
    category_scores_flat_agg = category_scores_flat.groupby(["metric","class","model_type","feature_set"]).agg({"score":[np.nanmean, model_eval.std_error],"support":np.mean})
    ## Metric List
    pmets = category_scores_flat_agg.index.levels[0]
    ## Create Plot
    fig, ax = plt.subplots(1, len(pmets), figsize=(len(pmets)*4, 3), sharex=False, sharey=True)
    for m, met in enumerate(pmets):
        ## Get Metric Scores and Classes
        metscores = category_scores_flat_agg.loc[met]
        metcls = ["overall"] + sorted([i for i in metscores.index.levels[0] if i != "overall"])
        ## Plot a Bar for Each Class
        for c, cls in enumerate(metcls):
            ## Isolate Class Information
            if cls not in metscores.index:
                continue
            bwidth = 0.9 / len(mt_compare)
            ## Plot A Bar for Each Model/Feature Set
            for ii, (mtype, ftype) in enumerate(mt_compare):
                iiscore = metscores.loc[cls].loc[mtype].loc[ftype]
                ax[m].bar(0.05 + c + ii * bwidth,
                          iiscore["score"]["nanmean"],
                          yerr=iiscore["score"]["std_error"],
                          color=f"C{ii}",
                          alpha=0.5,
                          label=None if c != 0 else "{}:{}".format(mtype, ftype) if mtype != "majority" else "majority",
                          align="edge",
                          width=bwidth,
                          capsize=2)
                if np.nan_to_num(iiscore["score"]["nanmean"]) > 0:
                    ax[m].text(0.05 + c + ii * bwidth + bwidth/2, iiscore["score"]["nanmean"] / 2, float_formatter(iiscore["score"]["nanmean"]), ha="center", va="center", fontsize=5)
        ax[m].set_title(met, fontweight="bold", pad=17.5)
        ax[m].spines["right"].set_visible(False)
        ax[m].spines["top"].set_visible(False)
        ax[m].set_xticks([i+0.5 for i in range(len(metcls))])
        ax[m].set_xticklabels(metcls, ha="center")
        if m == 0:
            ax[m].set_ylabel("Score", fontweight="bold")
        ax[m].legend(loc="upper right", frameon=True, fontsize=4)
    fig.tight_layout()
    return fig

def plot_learning_curves(category_scores, mtypes, ftypes):
    """
    
    """
    ## Get Score Dataframe
    category_scores_flat = create_flat_score_df(category_scores, mtypes, ftypes)
    ## Drop Duplicate of Majority Baseline (Since They All Give Same Outcome)
    mt_compare = []
    maj_seen = False
    for m in mtypes:
        for f in ftypes:
            if m == "majority" and maj_seen:
                continue
            elif m == "majority" and not maj_seen:
                maj_seen = True
            mt_compare.append((m, f))
    category_scores_flat = pd.concat([category_scores_flat.loc[(category_scores_flat["model_type"]==m)&
                                                               (category_scores_flat["feature_set"]==f)&
                                                               (category_scores_flat["metric"]!="roc_auc")&
                                                               (category_scores_flat["metric"]!="confusion")] for m, f in mt_compare])
    category_scores_flat["score"] = category_scores_flat["score"].astype(float)
    ## Aggregate
    aggs = category_scores_flat.groupby(["metric","class","model_type","feature_set","learning_curve_train_proportion"]).agg(
        {"score":[np.nanmean, model_eval.std_error],"learning_curve_train_size":np.median}
    )
    ## Create Plot (Class x Metric - One Line Per Model/Feature)
    mets = aggs.index.levels[0]
    classes = ["overall"] + sorted([i for i in aggs.index.levels[1] if i != "overall"])
    fig, ax = plt.subplots(len(mets), len(classes), figsize=(4 * len(classes), 3 * len(mets)), sharex=True, sharey=False)
    for m, met in enumerate(mets):
        for c, cls in enumerate(classes):
            pax = ax[m, c]
            plotted = False
            if cls in aggs.loc[met].index:
                plotted = True
                mc_data = aggs.loc[met].loc[cls]
                for ii, (mod, feat) in enumerate(mt_compare):
                    ii_mc_data = mc_data.loc[mod].loc[feat]
                    x = ii_mc_data[("learning_curve_train_size","median")].values
                    y = ii_mc_data[("score","nanmean")].values
                    yerr = ii_mc_data[("score","std_error")].values
                    pax.fill_between(x,
                                     y-yerr,
                                     y+yerr,
                                     color=f"C{ii}",
                                     alpha=0.4)
                    pax.plot(x,
                            y,
                            marker="o",
                            color=f"C{ii}",
                            alpha=0.8,
                            label="{}:{}".format(mod, feat) if mod != "majority" else "majority")
            pax.set_ylim(0, 1)
            pax.spines["right"].set_visible(False)
            pax.spines["top"].set_visible(False)
            if plotted:
                pax.legend(loc="upper left", fontsize=4)
            if m == 0:
                pax.set_title(cls, fontweight="bold")
            if c == 0:
                pax.set_ylabel(met, fontweight="bold")
            if m == len(mets) - 1:
                pax.set_xlabel("Training Size", fontweight="bold")
    fig.tight_layout()
    return fig

def plot_coefficients(coef_df, ktop=20):
    """
    
    """
    ## Labels
    labels = [c for c in coef_df.columns if c not in ["feature","keyword_category","feature_set","target","fold","model_type"]]
    labels = [l for l in labels if not coef_df[l].isnull().all()]
    ## Check For Multiple Types
    if len(coef_df["model_type"].unique()) > 1:
        raise ValueError("Only supports one model type per time.")
    ## Average
    coef_df_agg = coef_df.groupby("feature").agg({lbl:[np.mean, model_eval.std_error] for lbl in labels})
    ## Plot
    fig, ax = plt.subplots(1, len(labels), figsize=(len(labels)*4, 4))
    for l, lbl in enumerate(labels):
        pax = ax if len(labels) == 1 else ax[l]
        pdf = coef_df_agg[lbl].sort_values("mean", ascending=True).iloc[-ktop:].reset_index().copy()
        pdf["feature"] = pdf["feature"].map(lambda i: "{}='{}'".format(*i))
        pax.barh(list(range(pdf.shape[0])),
                 pdf["mean"].values,
                 xerr=pdf["std_error"].fillna(0),
                 color=f"C{l}",
                 alpha=0.5)
        pax.set_yticks(list(range(pdf.shape[0])))
        pax.set_yticklabels(pdf["feature"].tolist())
        pax.spines["right"].set_visible(False)
        pax.spines["top"].set_visible(False)
        pax.set_title(f"Label: '{lbl}'", loc="left")
        if pax.get_xlim()[0] < 0 and pax.get_xlim()[1] > 0:
            pax.axvline(0, color="black", alpha=0.3)
        pax.tick_params(axis="y", labelsize=6)
        pax.set_xlabel("Coefficient", fontsize=6)
        pax.set_ylim(-0.5, pdf.shape[0]-0.5)
    fig.tight_layout()
    return fig

####################
### Plotting (Multiple)
####################

def plot_all_label_bias(annotations, output_dir):
    """

    """
    ## Directory Creation
    if not os.path.exists(f"{output_dir}/bias/"):
        _ = os.makedirs(f"{output_dir}/bias/")
    ## Merged Plots
    fig = plot_label_bias(annotations, feature="keyword_negated", targets=["label_resolved","label_resolved_sentiment"], row_stratify="keyword_category")
    fig.savefig(f"{output_dir}/bias.keyword.png", dpi=200)
    plt.close(fig)
    fig = plot_label_bias(annotations, feature="encounter_type", targets=["label_resolved","label_resolved_sentiment"], row_stratify="keyword_category")
    fig.savefig(f"{output_dir}/bias.setting.png", dpi=200)
    plt.close(fig)
    ## Breakdown Plots
    for keyword_category in annotations["keyword_category"].unique():
        keyword_category_annotations = annotations.loc[annotations["keyword_category"]==keyword_category]
        for target in ["label_resolved","label_resolved_sentiment","encounter_type"]:
            ## Keyword Bias
            fig, kc_data = plot_label_bias(keyword_category_annotations, feature="keyword_negated", targets=[target], row_stratify="keyword_category", return_data=True)
            fig.savefig(f"{output_dir}/bias/keyword.{keyword_category}.{target}.png", dpi=200)
            plt.close(fig)
            _ = kc_data[0][2].to_csv(f"{output_dir}/bias/keyword.{keyword_category}.{target}.csv", index=True)
            ## Encounter Bias
            if target != "encounter_type":
                fig, kc_data = plot_label_bias(keyword_category_annotations, feature="encounter_type", targets=[target], row_stratify="keyword_category", return_data=True)
                fig.savefig(f"{output_dir}/bias/setting.{keyword_category}.{target}.png", dpi=200)
                plt.close(fig)
                _ = kc_data[0][2].to_csv(f"{output_dir}/bias/setting.{keyword_category}.{target}.csv", index=True)

def plot_all_performance(scores,
                         output_dir,
                         model_settings,
                         learning_curves=None,
                         is_bert=False):
    """
    
    """
    ## Isolate Score Types
    scores_dev = {x:list(filter(lambda s: s["split"]=="dev", y)) for x, y in scores.items()}
    scores_test = {x:list(filter(lambda s: s["split"]=="test", y)) for x, y in scores.items()}
    ## Iterate Over Keyword Categories
    mtypes = ["bert"] if is_bert else model_settings["models"]
    ftypes = ["bert"] if is_bert else list(model_settings["feature_sets"].keys())
    for cat in scores.keys():
        ## Iterate Over Splits
        for score_dict, score_type in zip([scores_dev, scores_test],["development","test"]):
            ## Hard-Code Target Column
            target = "label"
            ## Find Scores
            score_dict_cat_target = list(filter(lambda x: x["target"]==target, score_dict[cat]))
            ## No Data (e.g. No Tests, No Target)
            if len(score_dict_cat_target) == 0:
                continue
            ## Plot Performance Metric Head-to-Head
            fig = plot_performance_metrics_comparison(score_dict_cat_target, mtypes=mtypes, ftypes=ftypes)
            fig.savefig(f"{output_dir}/{cat}.performance.{target}.{score_type}.png", dpi=200)
            plt.close(fig)
            ## Confusion Matrices
            fig = plot_confusion_matrices(score_dict_cat_target, mtypes=mtypes, ftypes=ftypes)
            fig.savefig(f"{output_dir}/{cat}.confusion.{target}.{score_type}.png", dpi=200)
            plt.close(fig)
            ## ROC/AUC
            fig = plot_roc_curves(score_dict_cat_target, mtypes=mtypes, ftypes=ftypes)
            fig.savefig(f"{output_dir}/{cat}.roc_auc.{target}.{score_type}.png", dpi=200)
            plt.close(fig)
            ## Learning Curves
            if learning_curves is not None:
                fig = plot_learning_curves(score_dict_cat_target, mtypes=mtypes, ftypes=ftypes)
                fig.savefig(f"{output_dir}/{cat}.learning_curve.{target}.{score_type}.png",dpi=300)
                plt.close(fig)

def plot_performance_group(performance_group_scores, mtypes, ftypes):
    """
    
    """
    ## Helpers
    float_formatter = lambda number: "{:.2f}".format(number)
    ## Metrics To Plot
    pmetrics = ["accuracy","f1","precision","recall"]
    ## Model Comparisons
    mt_compare = []
    maj_seen = False
    for m in mtypes:
        for f in ftypes:
            if m == "majority" and maj_seen:
                continue
            elif m == "majority" and not maj_seen:
                maj_seen = True
            mt_compare.append((m, f))
    ## Apply Filter
    plot_subset = performance_group_scores.loc[performance_group_scores[["model_type","feature_set"]].apply(tuple,axis=1).isin(set(mt_compare))]
    ## Aggregate
    agg = plot_subset.groupby(["group_subcategory","model_type","feature_set"]).agg(
        {**{met:[np.nanmean, model_eval.std_error] for met in pmetrics}, "support":sum}
    )
    agg_subcats = list(filter(lambda s: s != "overall", agg.index.levels[0]))
    xticklabels = ["{}\n(n={})".format(s,agg.loc[s][("support","sum")].values[0]) for s in agg_subcats]
    ## Make Plot
    fig, ax = plt.subplots(1, len(pmetrics), figsize=(len(pmetrics)*(4 + (0.5 * len(xticklabels))), 3), sharex=True, sharey=True)
    for m, met in enumerate(pmetrics):
        ## Overall Performance
        for ii, (mod, fet) in enumerate(mt_compare):
            if (mod, fet) in agg.loc["overall"].index:
                ii_data = agg.loc["overall"].loc[mod].loc[fet][met]
                ax[m].fill_between([-.5, len(agg_subcats)+.5],
                                   ii_data["nanmean"] - ii_data["std_error"],
                                   ii_data["nanmean"] + ii_data["std_error"],
                                   color=f"C{ii}",
                                   alpha=0.3)
                ax[m].axhline(ii_data["nanmean"], color=f"C{ii}", linestyle="--", alpha=0.8)
        labeled = set()
        for gs, gsubcat in enumerate(agg_subcats):
            bwidth = 0.9 / len(mt_compare)
            for ii, (mod, fet) in enumerate(mt_compare):
                if (mod, fet) in agg.loc[gsubcat].index:
                    ii_data = agg.loc[gsubcat].loc[mod].loc[fet][met]
                    ax[m].bar(0.05 + gs + bwidth * ii,
                              ii_data["nanmean"],
                              yerr=ii_data["std_error"],
                              color=f"C{ii}",
                              alpha=0.4,
                              align="edge",
                              width=bwidth,
                              label=None if (mod, fet) in labeled else "{}:{}".format(mod, fet) if mod != "majority" else "majority",
                              capsize=2)
                    labeled.add((mod, fet))
                    if np.nan_to_num(ii_data["nanmean"]) > 0:
                        ax[m].text(0.05 + gs + bwidth * ii + bwidth/2, ii_data["nanmean"] / 2, float_formatter(ii_data["nanmean"]), ha="center", va="center", fontsize=5)
            ax[m].set_xticks(np.arange(len(agg_subcats))+.5)
            ax[m].set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=4)
            ax[m].set_title(met, fontweight="bold")
            ax[m].legend(loc="upper right", frameon=True, fontsize=4)
            ax[m].set_xlim(0, len(agg_subcats))
            if m == 0:
                ax[0].set_ylabel("Score", fontweight="bold")
    fig.tight_layout()
    return fig

def plot_all_performance_group(dataset_id,
                               group_scores,
                               output_dir,
                               model_settings,
                               is_bert=False):
    """
    
    """
    ## Directory
    if not os.path.exists(f"{output_dir}/group-scores/"):
        _ = os.makedirs(f"{output_dir}/group-scores/")
    ## Features/Models
    mtypes = ["bert"] if is_bert else model_settings["models"]
    ftypes = ["bert"] if is_bert else list(model_settings["feature_sets"].keys())
    ## Make Plots
    for keycat, keycat_group_scores in group_scores.items():
        ## Isolate Scores (Final Size Only, Dev Set)
        keycat_group_scores_df = pd.DataFrame(keycat_group_scores)
        keycat_group_scores_df = keycat_group_scores_df.loc[keycat_group_scores_df["learning_curve_train_proportion"]==100]
        keycat_group_scores_df = keycat_group_scores_df.loc[keycat_group_scores_df["split"]=="dev"]
        ## Iterate Through Semantic Performance Groups
        keycat_targets = keycat_group_scores_df["target"].unique()
        if dataset_id.startswith("mimic-iv"):
            perf_groups = settings.MIMIC_IV_PERFORMANCE_GROUPS
        else:
            perf_groups = []
        for perf_group in perf_groups:
            ## Iterate Through Targets
            for target in keycat_targets:
                perf_group_scores_df = keycat_group_scores_df.loc[keycat_group_scores_df["group_category"]==perf_group]
                fig = plot_performance_group(perf_group_scores_df, mtypes=mtypes, ftypes=ftypes)
                fig.savefig(f"{output_dir}/group-scores/{keycat}.{perf_group}.{target}.png",dpi=200)
                plt.close(fig)
    
def plot_all_coefficients(coefs,
                          output_dir,
                          model_settings):
    """
    
    """
    ## Initialize Output Directory
    if not os.path.exists(f"{output_dir}/coefficients/"):
        _ = os.makedirs(f"{output_dir}/coefficients/")
    ## Concatenate Coefficients
    coefs = {x:pd.concat(dfs).reset_index(drop=False).rename(columns={"index":"feature"}) for x, dfs in coefs.items() if len(dfs) > 0}
    ## Iterate Through Coefficient Groups
    for keycat, keycat_coefs in coefs.items():
        ## Iterate Through Models
        for mtype in [m for m in model_settings["models"] if m != "majority"]:
            ## Hardcoded Target Column
            target = "label"
            ## Iterate Through Feature Sets
            for fs in model_settings["feature_sets"].keys():
                ## Get Appropriate Coefs
                pcoefs = keycat_coefs.loc[(keycat_coefs["feature_set"]==fs)&
                                          (keycat_coefs["target"]==target)&
                                          (keycat_coefs["model_type"]==mtype)]
                ## Skip if Non-existant
                if pcoefs.shape[0] == 0:
                    continue
                ## Plot
                fig = plot_coefficients(pcoefs, ktop=30)
                fig.savefig(f"{output_dir}/coefficients/{keycat}.{target}.{fs}.{mtype}.png",dpi=200)
                plt.close(fig)
