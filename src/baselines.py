"""
Baseline anomaly detection models for UNSW-NB15.

Each baseline is trained only on normal training samples.
Thresholds are calibrated on normal validation reconstruction/anomaly scores.
Final metrics are reported once on the mixed test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .preprocessing import load_preprocessed_data
except ImportError:
    from preprocessing import load_preprocessed_data


def select_threshold_from_validation(val_scores, target_fpr = 0.05):
    """Select threshold using only normal validation scores."""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    return np.quantile(val_scores, 1.0 - target_fpr)


def compute_metrics(y_true, scores, threshold):
    """Compute binary classification metrics from anomaly scores."""
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division = 0)),
        "recall": float(recall_score(y_true, y_pred, zero_division = 0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division = 0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "test_fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def evaluate_isolation_forest(X_train, X_val, X_test, y_test, target_fpr = 0.05):
    """
    Isolation Forest baseline.

    sklearn's score_samples returns higher values for more normal samples,
    so we multiply by -1 to obtain anomaly scores where higher means more anomalous.
    """
    model = IsolationForest(
        n_estimators = 300,
        contamination = "auto",
        random_state = 42,
        n_jobs = -1,
    )
    model.fit(X_train)

    val_scores = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)

    threshold = select_threshold_from_validation(val_scores, target_fpr)
    metrics = compute_metrics(y_test, test_scores, threshold)
    metrics["model"] = "Isolation Forest"
    metrics["threshold_method"] = f"validation_normal_{int((1 - target_fpr) * 100)}th_percentile"

    return metrics


def evaluate_pca_reconstruction(X_train, X_val, X_test, y_test, target_fpr = 0.05, n_components = 0.95):
    """
    PCA reconstruction baseline.

    The anomaly score is the mean squared reconstruction error.
    n_components=0.95 keeps enough components to explain 95% of variance.
    """
    model = PCA(n_components = n_components, random_state = 42)
    model.fit(X_train)

    X_val_reconstructed = model.inverse_transform(model.transform(X_val))
    X_test_reconstructed = model.inverse_transform(model.transform(X_test))

    val_scores = np.mean((X_val - X_val_reconstructed) ** 2, axis = 1)
    test_scores = np.mean((X_test - X_test_reconstructed) ** 2, axis = 1)

    threshold = select_threshold_from_validation(val_scores, target_fpr)
    metrics = compute_metrics(y_test, test_scores, threshold)
    metrics["model"] = "PCA Reconstruction"
    metrics["threshold_method"] = f"validation_normal_{int((1 - target_fpr) * 100)}th_percentile"
    metrics["n_components"] = int(model.n_components_)

    return metrics


def run_baselines(data_dir = "../data/processed", results_dir = "../results", target_fpr = 0.05):
    """Run all baseline models and save a metrics table."""
    data = load_preprocessed_data(data_dir)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    results = [
        evaluate_isolation_forest(X_train, X_val, X_test, y_test, target_fpr),
        evaluate_pca_reconstruction(X_train, X_val, X_test, y_test, target_fpr),
    ]

    results_df = pd.DataFrame(results)

    output_path = Path(results_dir)
    output_path.mkdir(parents = True, exist_ok = True)

    csv_path = output_path / "baseline_metrics.csv"
    results_df.to_csv(csv_path, index = False)

    print("\nBaseline results:")
    print(results_df[[
        "model",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1_score",
        "test_fpr",
        "threshold",
    ]].to_string(index = False))

    print(f"\nSaved baseline metrics to: {csv_path}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Run anomaly detection baselines")
    parser.add_argument("--data_dir", type = str, default = "../data/processed")
    parser.add_argument("--results_dir", type = str, default = "../results")
    parser.add_argument("--target_fpr", type = float, default = 0.05)

    args = parser.parse_args()

    run_baselines(
        data_dir = args.data_dir,
        results_dir = args.results_dir,
        target_fpr = args.target_fpr,
    )
