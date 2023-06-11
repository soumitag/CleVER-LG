import os

import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import sklearn.metrics as metrics
import statistics

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, auc, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# from yellowbrick.classifier import ROCAUC
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier

# import umap
import shap
import argparse
import datetime


def roc_auc_score_multiclass(actual_class, pred_proba, average = "micro"):

    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        # new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        # roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average) [commented on 13/03/2023]
        roc_auc = roc_auc_score(new_actual_class, pred_proba[:,per_class], average='weighted')
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def get_roc_curves(actual_class, pred_proba):

    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)

    roc_curve_dict = {}
    for _class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != _class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        curve = RocCurveDisplay.from_predictions(new_actual_class, pred_proba[:,_class])
        roc_curve_dict[_class] = curve

    return roc_curve_dict


def compute_3class_weighted_sensitivity(confusion_matrix, y_true):
    # sensitivity = recall = true positive rate = TP/(TP+FN) = TP/P
    support = np.zeros(3)
    for i in range(3):
        support[i] = np.sum(y_true == i)
    M = confusion_matrix
    c0_sensitivity = M[0,0]/(M[0,0]+M[0,1]+M[0,2])
    c1_sensitivity = M[1,1]/(M[1,1]+M[1,0]+M[1,2])
    c2_sensitivity = M[2,2]/(M[2,2]+M[2,0]+M[2,1])
    weighted_sensitivity = (c0_sensitivity*support[0] + c1_sensitivity*support[1] + c2_sensitivity*support[2]) / (np.sum(support)*1.0)
    return weighted_sensitivity


def compute_3class_weighted_specificity(confusion_matrix, y_true):
    # specificity = true negative rate = TN/(TN+FP) = TN/N
    support = np.zeros(3)
    for i in range(3):
        support[i] = np.sum(y_true == i)
    M = confusion_matrix
    c0_specificity = (M[1,1]+M[2,2])/(M[1,1]+M[2,2]+M[1,0]+M[2,0])
    c1_specificity = (M[0,0]+M[2,2])/(M[0,0]+M[2,2]+M[0,1]+M[2,1])
    c2_specificity = (M[0,0]+M[1,1])/(M[0,0]+M[1,1]+M[0,2]+M[1,2])
    weighted_specificity = (c0_specificity*support[0] + c1_specificity*support[1] + c2_specificity*support[2]) / (np.sum(support)*1.0)
    return weighted_specificity


def compute_class_weighted_metrics(y_true, ovr_metrics, metric_name="NPV"):
    
    n_classes = len(np.unique(y_true))
    support = np.zeros(n_classes)
    for i in range(3):
        support[i] = np.sum(y_true == i)

    weighted_metric = np.sum(np.multiply(ovr_metrics[metric_name], support)) / (np.sum(support)*1.0)
    return weighted_metric


def compute_one_versus_rest_metrics(confusion_matrix):
    
    metrics = {}
    
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    metrics["TPR"] = TP/(TP+FN)
    # Specificity or true negative rate
    metrics["TNR"] = TN/(TN+FP) 
    # Precision or positive predictive value
    metrics["PPV"] = TP/(TP+FP)
    # Negative predictive value
    metrics["NPV"] = TN/(TN+FN)
    # Fall out or false positive rate
    metrics["FPR"] = FP/(FP+TN)
    # False negative rate
    metrics["FNR"] = FN/(TP+FN)
    # False discovery rate
    metrics["FDR"] = FP/(TP+FP)

    # Overall accuracy
    metrics["ACC"] = (TP+TN)/(TP+FP+FN+TN)
    print("OVR Metrics:", metrics)

    return metrics


def append_results(d1, d2):
    for key in d1:
        if key in d2:
            d1[key].append(d2[key])
        else:
            raise ValueError("Key={} not found in d2".format(key))


def append_clinical_data(data, clinical_data):
    data = data.join(clinical_data, how="left")
    return data


def recode_columns(train, test):

    train['status_sample'] = train.index
    train["status_sample"] = train["status_sample"].str.split("_", expand=True)[1]

    test['status_sample'] = test.index
    test["status_sample"] = test["status_sample"].str.split("_", expand=True)[1]

    for col in ["sex", "status_sample"]:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(train[col])

        train[col + "_recoded"] = label_encoder.transform(train[col])
        train = train.drop([col], axis=1)

        test[col + "_recoded"] = label_encoder.transform(test[col])
        test = test.drop([col], axis=1)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder = encoder.fit(train[['indication for transplant']])

    indication_for_transplant_map = {0:'Alcohol', 
                                     1:'Autoimmune', 
                                     2:'HBV', 
                                     3:'HCV', 
                                     4:'NASH', 
                                     5:'NASH/HPS', 
                                     6:'PBC',
                                     7:'PSC', 
                                     8:'alpha 1 antitrypsine deficiency'}

    encoder_df = pd.DataFrame(encoder.transform(train[['indication for transplant']]).toarray())
    encoder_df = encoder_df.rename(indication_for_transplant_map, axis=1)
    data_wclinicals = train.copy()
    for col in encoder_df.columns:
        data_wclinicals[col] = encoder_df[col].to_numpy()
    train = data_wclinicals.copy()
    train = train.drop(["status", "sample", "indication for transplant"], axis=1)

    encoder_df = pd.DataFrame(encoder.transform(test[['indication for transplant']]).toarray())
    encoder_df = encoder_df.rename(indication_for_transplant_map, axis=1)
    data_wclinicals = test.copy()
    for col in encoder_df.columns:
        data_wclinicals[col] = encoder_df[col].to_numpy()
    test = data_wclinicals.copy()
    test = test.drop(["status", "sample", "indication for transplant"], axis=1)

    return train, test


def normalize_features(train, test, non_numeric_cols=None):

    # remove non-numeric columns
    if non_numeric_cols is not None:
        train_numeric = train.drop(non_numeric_cols, axis=1)
        test_numeric = test.drop(non_numeric_cols, axis=1)
    else:
        train_numeric = train
        test_numeric = test
    
    # scale
    scaler = StandardScaler()
    scaler = scaler.fit(train_numeric)

    scaled_train = scaler.transform(train_numeric)
    scaled_train_df = pd.DataFrame(scaled_train, index=train_numeric.index, columns=train_numeric.columns)

    scaled_test = scaler.transform(test_numeric)
    scaled_test_df = pd.DataFrame(scaled_test, index=test_numeric.index, columns=test_numeric.columns)

    # add numeric columns back
    for col in non_numeric_cols:
        scaled_train_df[col] = train[col].copy()
        scaled_test_df[col] = test[col].copy()

    return scaled_train_df, scaled_test_df


def read_data(data_dir, index):
    train_path = os.path.join(data_dir, "train.logCPM_shortlisted.t.stat_{}.tsv".format(index+1))
    train = pd.read_csv(train_path, sep="\t")
    train = train.transpose()

    test_path = os.path.join(data_dir, "test.logCPM_shortlisted.t.stat_{}.tsv".format(index+1))
    test = pd.read_csv(test_path, sep="\t")
    test = test.transpose()

    common_columns = list(set(train.columns).intersection(test.columns))
    print("Index {}: Found {} common features".format(index, len(common_columns)))

    train = train[common_columns]
    test = test[common_columns]

    return train, test


def read_and_split_data(data_dir, filename, log_transform=False):

    data = pd.read_csv(os.path.join(data_dir, filename), sep="\t")
    
    labels = list(data.index)
    labels = [s.split("_")[1] for s in labels]
    
    data = data.drop(["recipient_age", "Hgb", "ALP", "ALT", "AST", "Creatinine", "sex", "indication.for.transplant"], axis=1)

    if log_transform:
        data += 1.0
        data = np.log2(data)

    sss = StratifiedShuffleSplit(n_splits=101, test_size=0.3, random_state=0)
    return data, labels, sss


def read_clinical_data(filename):
    clinical_df = pd.read_csv(filename, sep="\t")
    mapping = {"Control LT": "ControlLT", "NASH LT": "NASHLT", "Rejection ": "TCMR"}
    clinical_df = clinical_df.replace({"status": mapping})
    clinical_df["sample_status"] = clinical_df["sample"].str.upper() + "_" + clinical_df["status"]
    clinical_df = clinical_df.set_index("sample_status")
    return clinical_df


def get_confidence_interval(values, ptile=95.0):
    alpha = 100.0-ptile

    # retrieve observation at lower percentile
    lower_p = alpha / 2.0
    lower = max(0.0, np.percentile(values, lower_p))

    # retrieve observation at upper percentile
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = min(1.0, np.percentile(values, upper_p))

    return (lower, upper)


def log_ovr_results(ovr_metrics, results_dir, run_name):
    avg_ovr_metrics = {}
    labels = ["Control LT", "NASH LT", "TCMR"]

    for key in ["ACC", "TPR", "TNR", "PPV", "NPV"]:
        
        arr = np.array(ovr_metrics[key])
        arr[np.isnan(arr)] = 0
        
        avg_ovr_metrics[key] = {}
        for i in range(3):
            avg_ovr_metrics[key][labels[i] + " vs Rest Mean"] = np.mean(arr[:,i])
            avg_ovr_metrics[key][labels[i] + " vs Rest Conf. Int."] = get_confidence_interval(arr[:,i])
        
    df_ovr_metrics = pd.DataFrame(avg_ovr_metrics)
    df_ovr_metrics = df_ovr_metrics.transpose()

    df_ovr_metrics.to_csv(os.path.join(results_dir, "ovr_metrics_{}.tsv".format(run_name)), sep="\t")

    print(df_ovr_metrics.round(2))


def save_shap_features(results, data, fig_dir, run_name, target_names, seed=51293):

    models = results["models"]
    accuracies = results["accuracies"]
    median_model = np.argsort(accuracies)[len(accuracies)//2]
    model_ = models[median_model]

    splits = results["split_indices"][median_model]
    train = data.iloc[splits["train"]]
    test = data.iloc[splits["test"]]
    X_train, X_test, _, _ = preprocess(train, test, features="non_clinical")

    explainer = shap.LinearExplainer(model_, X_train.values, nsamples=10, seed=seed)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    shap.summary_plot(shap_values, 
                    X_test.values,
                    max_display = 15,
                    plot_type='bar',
                    feature_names=X_train.columns, 
                    class_names=target_names, 
                    plot_size='auto', 
                    show=False)

    ax.set_xlabel("mean(|SHAP value|)\n(average impact on model output magnitude)")

    # Get the current figure and axes objects.
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
        item.set_color("black")
        
    ax.grid(False)

    for p in ["top", "bottom", "left", "right"]:
        ax.spines[p].set_visible(True)
        ax.spines[p].set_color("black")
        ax.spines[p].set_linewidth(0.5)

    plt.savefig(os.path.join(fig_dir, "shap_features_{}.pdf".format(run_name)))
    plt.show(block=False)


def visualize_results(results, target_names, results_dir, fig_dir, run_name):

    models = results["models"]
    confusion_matrices = results["confusion_matrices"]
    predictions = results["predictions"]
    accuracies = results["accuracies"]
    f1_scores = results["f1_scores"]
    precision_vals = results["precision_vals"]
    recall_vals = results["recall_vals"]
    auc_vals = results["auc_vals"]
    specificities = results["specificities"]
    npv_values = results["npv_values"]
    roc_curves = results["roc_curves"]

    results_df = pd.DataFrame.from_dict({'Accuracy': accuracies, 
                                         'F1-score': f1_scores, 
                                         'Precision/PPV': precision_vals, 
                                         'Recall/Sensitivity/TPR': recall_vals, 
                                         'Specificity/TNR': specificities,
                                         'NPV': npv_values}, 
                                        orient='columns')

    median_model = np.argsort(accuracies)[len(accuracies)//2]
    best_model_index = np.argmax(accuracies)

    d = {}
    median_model = np.argsort(accuracies)[len(accuracies)//2]
    for col in results_df.columns:
        d[col] = {"Mean": np.mean(results_df[col]), 
                  "Median Model": results_df[col][median_model], 
                  "Confidence Interval": get_confidence_interval(results_df[col].tolist())}
    results_summary_df = pd.DataFrame.from_records(d)
    results_summary_df = results_summary_df.transpose()
    results_summary_df.to_csv(os.path.join(results_dir, "results_summary_{}.tsv".format(run_name)), sep="\t")

    print(results_summary_df)

    conf_mats = np.array(confusion_matrices)
    print(conf_mats.shape)
    mean_conf_mat = np.mean(conf_mats, axis=0)
    print(mean_conf_mat)

    median_acc = accuracies[median_model]
    print(confusion_matrices[median_model])
    median_conf_mat = confusion_matrices[median_model]

    best_acc = accuracies[best_model_index]
    best_conf_mat = confusion_matrices[best_model_index]
    print("Best Model Index:", best_model_index)
    print("Best Accuracy:", best_acc)
    print("Best Conf. Mat.", best_conf_mat)

    # AUROC
    print("AUROC (Mean Across 101 Bootstraps):")
    auroc_dict = {}
    for tgn in target_names:
        auroc_dict[tgn] = []

    for auc_vals_iter in auc_vals:
        for index, tgn in enumerate(target_names):
            auroc_dict[tgn].append(auc_vals_iter[index])

    auroc_summary_df = pd.DataFrame.from_dict(auroc_dict, orient="columns")
    auroc_means_df = auroc_summary_df.mean(axis=0)
    print(auroc_means_df)

    # print(mean_conf_mat.astype(float).sum(axis=1)[:, np.newaxis])
    # mean_conf_mat_normed = mean_conf_mat / mean_conf_mat.astype(float).sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(3.5,3))
    sns.heatmap(mean_conf_mat, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual', weight = 'bold')
    plt.xlabel('Predicted', weight = 'bold')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(fig_dir, "avg_confusion_mat_regions_{}.pdf".format(run_name)))
    plt.show(block=False)

    fig, ax = plt.subplots(figsize=(3.5,3))
    sns.heatmap(median_conf_mat, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual', weight = 'bold')
    plt.xlabel('Predicted', weight = 'bold')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(fig_dir, "median_confusion_mat_regions_{}.pdf".format(run_name)))
    plt.show(block=False)

    fig, ax = plt.subplots(figsize=(3.5,3))
    sns.heatmap(best_conf_mat, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual', weight = 'bold')
    plt.xlabel('Predicted', weight = 'bold')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(fig_dir, "best_confusion_mat_regions_{}.pdf".format(run_name)))
    plt.show(block=False)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    for roc_index, _roc_curves in enumerate(roc_curves):
        for index, tgn in enumerate(target_names):
            roc_curve = _roc_curves[index]
            if roc_index == median_model:
                roc_curve.plot(axs[index], name=None, linewidth=1.5, linestyle = 'solid', color = 'red')
            else:
                roc_curve.plot(axs[index], name=None, linewidth=0.5, linestyle = 'dashed', color = 'gray')
        
    for index, ax in enumerate(axs):
        ax.set_title("{} vs Rest\nAUCROC={:.3f}".format(target_names[index], auroc_means_df[target_names[index]]))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend('', frameon=False)

    plt.savefig(os.path.join(fig_dir, "roc_curves_{}.pdf".format(run_name)))
    plt.show(block=False)


def preprocess(X_train, X_test, features="all"):

    # append clinical data
    clinical_df = read_clinical_data(os.path.join(data_dir, "clinical_info.tsv"))
    X_train = append_clinical_data(X_train, clinical_df)
    X_test = append_clinical_data(X_test, clinical_df)

    # z-tranform numeric features
    X_train, X_test = normalize_features(X_train, X_test, non_numeric_cols = ["sex", "status", "indication for transplant", "sample"])
        
    # encode categorcial features
    X_train, X_test = recode_columns(X_train, X_test)

    # separate labels from training data
    y_train = X_train["status_sample_recoded"].astype('int')
    X_train = X_train.drop(["status_sample_recoded"], axis=1)

    y_test = X_test["status_sample_recoded"].astype('int')
    X_test = X_test.drop(["status_sample_recoded"], axis=1)

    # replace missing values
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # filter features
    clinical_variables = ["recipient_age", "Hgb", "ALP", "ALT", "AST", "Creatinine", "sex_recoded", 
                          "Alcohol", "Autoimmune", "HBV", "HCV", "NASH", "NASH/HPS", "PBC", "PSC", 
                          "alpha 1 antitrypsine deficiency"]

    if features == "all":
        return X_train, X_test, y_train, y_test

    elif features == "non_clinical":
        # remove all clinical
        present_clinical_cols = []
        for c in clinical_variables:
            if c in X_train.columns:
                present_clinical_cols.append(c)

        X_train = X_train.drop(columns=present_clinical_cols)
        X_test = X_test.drop(columns=present_clinical_cols)
        return X_train, X_test, y_train, y_test

    elif features == "clinical_only":
        # remove all non-clinical
        non_clinical_cols = []
        for c in X_train.columns:
            if c not in clinical_variables:
                non_clinical_cols.append(c)

        X_train = X_train.drop(columns=non_clinical_cols)
        X_test = X_test.drop(columns=non_clinical_cols)
        return X_train, X_test, y_train, y_test

    else:
        raise ValueError("Unknown feature type requested")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cfMeDIP Classifier')
    parser.add_argument('--run-name', help='a name for the run', required=True, type=str, action="store", dest="run_name")
    args = parser.parse_args()

    seed = 51293
    data_dir = "../../../data"
    results_dir = "../../../results/new_results"
    fig_dir = os.path.join(results_dir, "figures")

    labels_index = {0: "ControlLT", 1: "NASHLT", 2: "TCMR"}

    ovr_metrics = {"TPR": [], "TNR": [], "PPV": [], "NPV": [],
                   "FPR": [], "FNR": [], "FDR": [],
                   "ACC": []}

    classifier = "logistic_l2"

    predictions = []
    accuracies = []
    f1_scores = []
    precision_vals = []
    recall_vals = []
    # feature_importances = pd.DataFrame(columns=["feature", "importance", "coeff", "rank"])
    models = []
    confusion_matrices = []
    auc_vals = []
    specificities = []
    npv_vals = []
    roc_curves = []
    split_indices = []

    data, labels, splits = read_and_split_data(data_dir, "features_raw_counts.tsv", log_transform=True)

    for _bootstrap_index, (train_index, test_index) in enumerate(splits.split(data, labels)):

        # read data and transpose
        # X_train, X_test = read_data(os.path.join(data_dir, "bootstrapped_files"), _bootstrap_index)
        # X_train, X_test = read_and_split_data(data_dir, "Final.logCPM.300dmrs.clinical.feb20.tsv", _bootstrap_index)
        # X_train, X_test = read_and_split_data(data_dir, "features_raw_counts.tsv", _bootstrap_index, log_transform=True)
        
        train, test = data.iloc[train_index], data.iloc[test_index]
        X_train, X_test, y_train, y_test = preprocess(train, test, features="non_clinical")

        if classifier == "logistic_l2":
            clf = LogisticRegression(n_jobs=-1, penalty="l2", solver="lbfgs", class_weight="balanced",
                                     multi_class="multinomial", max_iter=3000)
        elif classifier == "none":
            clf = LogisticRegressionCV(n_jobs=-1, penalty="l1", solver="saga", class_weight="balanced",
                                       multi_class="multinomial")
        elif classifier == "lasso":
            clf = linear_model.Lasso(alpha=0.1)

        clf.fit(X_train.values, y_train.values)

        test_predictions = clf.predict(X_test.values)
        acc = accuracy_score(y_test, test_predictions)
        precision, _, f1, support = precision_recall_fscore_support(y_test, test_predictions, average='weighted')
        cnf_matrix = metrics.confusion_matrix(y_test, test_predictions, normalize='true')

        specificity = compute_3class_weighted_specificity(cnf_matrix, y_test)
        recall = compute_3class_weighted_sensitivity(cnf_matrix, y_test)

        # tn, fp, fn, tp = cnf_matrix.ravel()
        # specificity = tn / (tn+fp)

        # false_positive_rate, true_positive_rate, thresholds = auc(y_test, clf.predict_proba(X_test.values)[:,1])
        test_prediction_probabilities = clf.predict_proba(X_test.values)
        # _auc = roc_auc_score_multiclass(y_test, test_predictions)  [commented on 13/03/2023]
        _auc = roc_auc_score_multiclass(y_test, test_prediction_probabilities)
        roc_curves_iter = get_roc_curves(y_test, test_prediction_probabilities)

        ovr_metrics_index = compute_one_versus_rest_metrics(cnf_matrix)
        npv = compute_class_weighted_metrics(y_test, ovr_metrics_index, metric_name="NPV")

        # if acc > 0.54 and acc < 0.92:
        predictions.append(test_predictions)
        accuracies.append(acc)
        f1_scores.append(f1)
        precision_vals.append(precision)
        recall_vals.append(recall)
        models.append(clf)
        confusion_matrices.append(cnf_matrix)
        auc_vals.append(_auc)
        specificities.append(specificity)
        npv_vals.append(npv)
        roc_curves.append(roc_curves_iter)
        append_results(ovr_metrics, ovr_metrics_index)
        split_indices.append({"train": train_index, "test": test_index})

        print("=============")
        print("Model {}".format(_bootstrap_index))
        print("=============")
        print("Train-Test Split")
        for i in range(3):
            print("\tClass = {}, Train Count = {}, Test Count = {}".format(labels_index[i], np.sum(y_train == i),
                                                                           np.sum(y_test == i)))
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Specificity:", specificity)
        print("NPV:", npv)
        print("Support:", support)
        print("AUC:", _auc)
        print("Confusion Matrix:")
        print(cnf_matrix)
        print("")


results = {"classifier": classifier, 
           "predictions": predictions,
           "accuracies": accuracies,
           "f1_scores": f1_scores,
           "precision_vals": precision_vals,
           "recall_vals": recall_vals,
           "models": models,
           "confusion_matrices": confusion_matrices,
           "auc_vals": auc_vals,
           "specificities": specificities,
           "npv_values": npv_vals,
           "ovr_metrics": ovr_metrics,
           "roc_curves": roc_curves,
           "split_indices": split_indices
           }

timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
np.save(os.path.join(results_dir, "cfmedip_classifier_results_{}_{}".format(args.run_name, timestamp)), results)

target_names = ["ControlLT", "NASHLT", "TCMR"]

_run_name = args.run_name + "_" + timestamp
visualize_results(results, target_names, results_dir, fig_dir, _run_name)
log_ovr_results(ovr_metrics, results_dir, _run_name)

# SHAP
save_shap_features(results, data, fig_dir, _run_name, target_names, seed=seed)

# clear
plt.show()