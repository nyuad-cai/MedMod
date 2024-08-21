from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics

# Binary classification metrics (e.g., for decompensation, in-hospital mortality)
def print_metrics_binary(y_true, predictions, verbose=1):
    """
    Prints and returns metrics for binary classification tasks.
    
    :param y_true: True binary labels.
    :param predictions: Predicted probabilities for the positive class.
    :param verbose: Verbosity level. If 1, prints the metrics.
    :return: Dictionary containing various classification metrics.
    """
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("Confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    precisions, recalls, _ = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print(f"Accuracy = {acc}")
        print(f"Precision class 0 = {prec0}")
        print(f"Precision class 1 = {prec1}")
        print(f"Recall class 0 = {rec0}")
        print(f"Recall class 1 = {rec1}")
        print(f"AUC of ROC = {auroc}")
        print(f"AUC of PRC = {auprc}")
        print(f"min(+P, Se) = {minpse}")

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}


# Multi-label classification metrics (e.g., for phenotyping)
def print_metrics_multilabel(y_true, predictions, verbose=1):
    """
    Prints and returns metrics for multi-label classification tasks.
    
    :param y_true: True binary labels for each class.
    :param predictions: Predicted probabilities for each class.
    :param verbose: Verbosity level. If 1, prints the metrics.
    :return: Dictionary containing AUC scores and average AUCs.
    """
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print(f"ave_auc_micro = {ave_auc_micro}")
        print(f"ave_auc_macro = {ave_auc_macro}")
        print(f"ave_auc_weighted = {ave_auc_weighted}")

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


# Mean Absolute Percentage Error (for regression tasks)
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes the Mean Absolute Percentage Error (MAPE).
    
    :param y_true: True values.
    :param y_pred: Predicted values.
    :return: MAPE value.
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


# Regression metrics (e.g., for length of stay prediction)
def print_metrics_regression(y_true, predictions, verbose=1):
    """
    Prints and returns metrics for regression tasks.
    
    :param y_true: True continuous labels.
    :param predictions: Predicted continuous labels.
    :param verbose: Verbosity level. If 1, prints the metrics.
    :return: Dictionary containing various regression metrics.
    """
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print(f"Mean absolute deviation (MAD) = {mad}")
        print(f"Mean squared error (MSE) = {mse}")
        print(f"Mean absolute percentage error (MAPE) = {mape}")
        print(f"Cohen kappa score = {kappa}")

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}


# Logarithmic binning and estimation for length of stay prediction
class LogBins:
    nbins = 10
    means = [0.611848, 2.587614, 6.977417, 16.465430, 37.053745,
             81.816438, 182.303159, 393.334856, 810.964040, 1715.702848]


def get_bin_log(x, nbins, one_hot=False):
    """
    Assigns a log-scaled bin to a value.
    
    :param x: Value to bin.
    :param nbins: Number of bins.
    :param one_hot: If True, returns a one-hot encoded bin; otherwise returns bin index.
    :return: Bin index or one-hot encoded bin.
    """
    binid = int(np.log(x + 1) / 8.0 * nbins)
    binid = max(0, min(binid, nbins - 1))

    if one_hot:
        ret = np.zeros((LogBins.nbins,))
        ret[binid] = 1
        return ret
    return binid


def get_estimate_log(prediction, nbins):
    """
    Estimates a value from a log-scaled bin prediction.
    
    :param prediction: Predicted bin probabilities.
    :param nbins: Number of bins.
    :return: Estimated value.
    """
    bin_id = np.argmax(prediction)
    return LogBins.means[bin_id]


def print_metrics_log_bins(y_true, predictions, verbose=1):
    """
    Prints and returns metrics for regression tasks using logarithmic bins.
    
    :param y_true: True continuous labels.
    :param predictions: Predicted continuous labels.
    :param verbose: Verbosity level. If 1, prints the metrics.
    :return: Dictionary containing various regression metrics.
    """
    y_true_bins = [get_bin_log(x, LogBins.nbins) for x in y_true]
    prediction_bins = [get_bin_log(x, LogBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("LogBins confusion matrix:")
        print(cf)
    return print_metrics_regression(y_true, predictions, verbose)


# Custom binning and estimation for length of stay prediction
class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    """
    Assigns a custom bin to a value.
    
    :param x: Value to bin.
    :param nbins: Number of bins.
    :param one_hot: If True, returns a one-hot encoded bin; otherwise returns bin index.
    :return: Bin index or one-hot encoded bin.
    """
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
        return None


def get_estimate_custom(prediction, nbins):
    """
    Estimates a value from a custom bin prediction.
    
    :param prediction: Predicted bin probabilities.
    :param nbins: Number of bins.
    :return: Estimated value.
    """
    bin_id = np.argmax(prediction)
    assert 0 <= bin_id < nbins
    return CustomBins.means[bin_id]


def print_metrics_custom_bins(y_true, predictions, verbose=1):
    """
    Prints and returns metrics for regression tasks using custom bins.
    
    :param y_true: True continuous labels.
    :param predictions: Predicted continuous labels.
    :param verbose: Verbosity level. If 1, prints the metrics.
    :return: Dictionary containing various regression metrics.
    """
    return print_metrics_regression(y_true, predictions, verbose)