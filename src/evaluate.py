"""
evaluate.py
===========
Standalone evaluation module for the Fake News Detection System.

Loads the persisted model + TF-IDF vectorizer (produced by ``train.py``),
recreates the exact same train/test split used during training (same
``random_state``, so the test set was never seen during fitting), and
produces a detailed evaluation report: classification metrics, a
confusion matrix plot, and an ROC curve — saved under ``models/reports/``.

This module is deliberately decoupled from ``train.py`` so that model
evaluation can be re-run at any time (e.g. after loading a model shipped
via CI/CD) without re-training.

Run directly with:

    python src/evaluate.py

Author: Your Name
"""

import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from utils import get_logger, ensure_dir, MODELS_DIR, MODEL_PATH, VECTORIZER_PATH
from train import load_dataset, split_data, get_decision_scores

logger = get_logger(__name__)

REPORTS_DIR = f"{MODELS_DIR}/reports"
LABELS = ["Fake", "Real"]


def load_artifacts():
    """Load the trained model and TF-IDF vectorizer from disk.

    Returns
    -------
    tuple
        ``(model, vectorizer)``

    Raises
    ------
    FileNotFoundError
        If either artifact is missing (i.e. ``train.py`` has not been run).
    """
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Model/vectorizer artifacts not found. Run `python src/train.py` first."
        ) from exc
    return model, vectorizer


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    """Plot and save a labeled confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    save_path : str
        Destination PNG path.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
    )
    plt.title("Confusion Matrix — Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved confusion matrix plot -> %s", save_path)


def plot_roc_curve(y_true, scores, save_path: str) -> float:
    """Plot and save the ROC curve, returning the AUC.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    scores : array-like
        Predicted probabilities / decision scores for the positive class.
    save_path : str
        Destination PNG path.

    Returns
    -------
    float
        Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved ROC curve plot -> %s", save_path)
    return roc_auc


def run_evaluation():
    """End-to-end evaluation pipeline: load artifacts, evaluate on the
    held-out test set, print a classification report, and save plots.

    Returns
    -------
    dict
        Summary containing the classification report and ROC-AUC.
    """
    ensure_dir(REPORTS_DIR)

    logger.info("Loading model and vectorizer...")
    model, vectorizer = load_artifacts()

    logger.info("Recreating train/test split (same random_state as training)...")
    df = load_dataset()
    X_train_text, X_test_text, y_train, y_test = split_data(df)
    X_test = vectorizer.transform(X_test_text)

    y_pred = model.predict(X_test)
    scores = get_decision_scores(model, X_test)

    report_text = classification_report(y_test, y_pred, target_names=LABELS)
    logger.info("Classification Report:\n%s", report_text)

    plot_confusion_matrix(y_test, y_pred, f"{REPORTS_DIR}/confusion_matrix.png")
    roc_auc_value = plot_roc_curve(y_test, scores, f"{REPORTS_DIR}/roc_curve.png")

    report_dict = classification_report(
        y_test, y_pred, target_names=LABELS, output_dict=True
    )
    report_dict["roc_auc"] = round(roc_auc_value, 4)

    with open(f"{REPORTS_DIR}/evaluation_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved evaluation_report.json -> %s", REPORTS_DIR)

    with open(f"{REPORTS_DIR}/classification_report.txt", "w") as f:
        f.write(report_text)

    return report_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
