"""
Evaluation functions for the trained autoencoder
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

try:
    from .preprocessing import load_preprocessed_data
    from .model import DeepAutoencoder, get_reconstruction_error
except ImportError:
    from preprocessing import load_preprocessed_data
    from model import DeepAutoencoder, get_reconstruction_error


def load_model(model_path, input_dim = 33, latent_dim = 8, device = None):
    """
    Load the trained autoencoder model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepAutoencoder(input_dim = input_dim, latent_dim = latent_dim)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()
    
    return model, device


def calculate_errors(model, val_loader, test_loader, device):
    """
    Calculate reconstruction errors for validation and test sets
    """
    
    val_errors = get_reconstruction_error(model, val_loader, device)
    test_errors = get_reconstruction_error(model, test_loader, device)
    
    return val_errors, test_errors


def find_optimal_threshold(y_val, val_errors):
    """
    Find optimal threshold using ROC curve (Youden's Index)
    """
    
    fpr, tpr, thresholds = roc_curve(y_val, val_errors)
    roc_auc = roc_auc_score(y_val, val_errors)
    
    # Youden's Index
    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold : {optimal_threshold:.6f}")
    print(f"TPR : {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    
    return {
        'optimal_threshold' : optimal_threshold,
        'tpr' : tpr,
        'fpr' : fpr,
        'thresholds' : thresholds,
        'optimal_idx' : optimal_idx,
        'roc_auc' : roc_auc
    }


def select_threshold_from_validation(val_errors, target_fpr = 0.05):
    """
    Select anomaly threshold using only normal validation errors.

    target_fpr=0.05 means the threshold is set at the 95th percentile
    of normal validation reconstruction errors.
    """
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")

    threshold = np.quantile(val_errors, 1.0 - target_fpr)

    print(f"Validation-calibrated threshold: {threshold:.6f}")
    print(f"Target validation FPR: {target_fpr:.2%}")

    return {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "calibration_method": "normal_validation_percentile",
    }


def calculate_metrics(y_test, test_errors, threshold):
    """
    Calculate performance metrics on test set
    """
    
    y_pred = (test_errors > threshold).astype(int)
    
    metrics = {
        'threshold' : threshold,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred),
        'recall' : recall_score(y_test, y_pred),
        'f1_score' : f1_score(y_test, y_pred),
        'roc_auc' : roc_auc_score(y_test, test_errors),
        'y_pred' : y_pred
    }
    
    print("Test set performance :")
    print(f"Threshold: {metrics['threshold']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    
    return metrics


def calculate_confusion_matrix(y_test, y_pred):
    """
    Calculate confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    result = {
        'confusion_matrix' : cm,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'tp' : tp
    }
    
    print(f"\nConfusion Matrix:")
    print(f"TN : {tn:6d} | FP : {fp:6d}")
    print(f"FN : {fn:6d} | TP : {tp:6d}")
    
    return result


def plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, save_path = None):
    """
    Plot ROC curve
    """
    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, linewidth = 2, label = f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Random')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], s = 100, c = 'red',
                marker = 'o', label = f'Selected (threshold = {optimal_threshold:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight = 'bold')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_confusion_matrix(cm, save_path = None):
    """
    Plot confusion matrix
    """
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels = ['Normal', 'Attack'],
                yticklabels = ['Normal', 'Attack'])
    plt.xlabel('Predicted', fontweight = 'bold')
    plt.ylabel('True', fontweight = 'bold')
    plt.title('Confusion Matrix', fontweight = 'bold')
    if save_path:
        Path(save_path).parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_error_distributions(val_normal_errors, test_normal_errors,
                            test_attack_errors, threshold, save_path = None):
    """
    Plot reconstruction error distributions
    """
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    # Validation set (normal only)
    axes[0].hist(val_normal_errors, bins = 50, alpha = 0.7, label = 'Normal',
                color = 'green', density = True)
    axes[0].axvline(threshold, color = 'black', linestyle = '--', linewidth = 2,
                    label = f'Threshold = {threshold:.3f}')
    axes[0].set_xlabel('Reconstruction Error (MSE)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Validation Set - Normal Only', fontweight = 'bold')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    # Test set
    axes[1].hist(test_normal_errors, bins = 50, alpha = 0.7, label = 'Normal',
                color = 'green', density = True)
    axes[1].hist(test_attack_errors, bins = 50, alpha = 0.7, label = 'Attack',
                color = 'red', density = True)
    axes[1].axvline(threshold, color = 'black', linestyle = '--', linewidth = 2,
                    label = f'Threshold = {threshold:.3f}')
    axes[1].set_xlabel('Reconstruction Error (MSE)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Test Set : Reconstruction Error Distribution', fontweight = 'bold')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.show()

def evaluate_autoencoder(data_dir = '../data/processed',
                        model_path = '../models/autoencoder.pth',
                        results_dir = '../results',
                        input_dim = 33,
                        latent_dim = 8,
                        batch_size = 256,
                        target_fpr = 0.05,
                        device = None):
    """
    Complete evaluation pipeline (combines all functions above, standalone use)
    """
    #Data loading
    data = load_preprocessed_data(data_dir)
    X_val = data['X_val']
    X_test = data['X_test']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"Val :  {X_val.shape}")
    print(f"Test : {X_test.shape}")
    
    #DataLoaders
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    #Model loading
    model, device = load_model(model_path, input_dim, latent_dim, device)
    
    #Errors
    val_errors, test_errors = calculate_errors(model, val_loader, test_loader, device)
    
    #Separate by label
    val_normal_errors = val_errors[y_val == 0]
    test_normal_errors = test_errors[y_test == 0]
    test_attack_errors = test_errors[y_test == 1]

    print(f"\nError statistics :")
    print(f"Val Normal  :  {val_normal_errors.mean():.6f}")
    print(f"Test Normal :  {test_normal_errors.mean():.6f}")
    print(f"Test Attack :  {test_attack_errors.mean():.6f}")
    print(f"Separation  :  {test_attack_errors.mean() / test_normal_errors.mean():.2f}x")

    # Threshold selected only from normal validation data.
    # This avoids tuning the operating point on the test set.
    threshold_info = select_threshold_from_validation(
        val_normal_errors,
        target_fpr = target_fpr,
    )

    # Metrics on the untouched mixed test set
    metrics = calculate_metrics(y_test, test_errors, threshold_info["threshold"])

    #Confusion matrix
    cm_info = calculate_confusion_matrix(y_test, metrics['y_pred'])

    #ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_errors)
    roc_auc = roc_auc_score(y_test, test_errors)
    selected_idx = np.argmin(np.abs(thresholds - threshold_info["threshold"]))

    plot_roc_curve(fpr, tpr, roc_auc,
                selected_idx, threshold_info["threshold"],
                f"{results_dir}/roc_curve.png")

    #Confusion matrix
    plot_confusion_matrix(cm_info['confusion_matrix'], f"{results_dir}/confusion_matrix.png")

    #Error distributions
    plot_error_distributions(val_normal_errors,
                            test_normal_errors, test_attack_errors,
                            threshold_info["threshold"],
                            f"{results_dir}/error_distributions.png")

    #Results
    return {
        'threshold' : threshold_info["threshold"],
        'threshold_method' : threshold_info["calibration_method"],
        'target_validation_fpr' : threshold_info["target_fpr"],
        'accuracy' : metrics['accuracy'],
        'precision' : metrics['precision'],
        'recall' : metrics['recall'],
        'f1_score' : metrics['f1_score'],
        'roc_auc' : metrics['roc_auc'],
        'confusion_matrix' : cm_info['confusion_matrix'].tolist(),
        'normal_mean_error' : test_normal_errors.mean(),
        'attack_mean_error' : test_attack_errors.mean(),
        'separation_ratio' : test_attack_errors.mean() / test_normal_errors.mean()
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Evaluate trained autoencoder')
    parser.add_argument('--data_dir', type = str, default = '../data/processed')
    parser.add_argument('--model_path', type = str, default = '../models/autoencoder.pth')
    parser.add_argument('--results_dir', type = str, default = '../results')
    parser.add_argument('--input_dim', type = int, default = 33)
    parser.add_argument('--latent_dim', type = int, default = 8)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--target_fpr', type = float, default = 0.05)
    parser.add_argument('--device', type = str, default = None)
    
    args = parser.parse_args()
    
    metrics = evaluate_autoencoder(
        data_dir = args.data_dir,
        model_path = args.model_path,
        results_dir = args.results_dir,
        input_dim = args.input_dim,
        latent_dim = args.latent_dim,
        batch_size = args.batch_size,
        target_fpr = args.target_fpr,
        device = args.device
    )
    
    print("\nKey Metrics :")
    print(f"Threshold Method : {metrics['threshold_method']}")
    print(f"Target Val FPR :  {metrics['target_validation_fpr']:.2%}")
    print(f"Accuracy :        {metrics['accuracy']:.4f}")
    print(f"Precision :       {metrics['precision']:.4f}")
    print(f"Recall :          {metrics['recall']:.4f}")
    print(f"F1-Score :        {metrics['f1_score']:.4f}")
    print(f"ROC-AUC :         {metrics['roc_auc']:.4f}")
    print(f"\nSeparation Ratio : {metrics['separation_ratio']:.2f}x")