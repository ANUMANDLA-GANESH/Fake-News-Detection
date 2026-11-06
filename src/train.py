"""
train.py
========
Feature engineering (TF-IDF), model training, comparison, and
hyperparameter tuning for the Fake News Detection System.

Pipeline
--------
1. Load ``dataset/news_cleaned.csv`` (produced by ``preprocess.py``).
2. Split into train/test sets (stratified on ``label``).
3. Fit a TF-IDF vectorizer (``max_features=10000``, ``ngram_range=(1,2)``)
   on the training text only (to avoid test-set leakage).
4. Train four candidate models: Logistic Regression, Multinomial Naive
   Bayes, Linear SVM, and Random Forest.
5. Compare them on accuracy, precision, recall, F1, and ROC-AUC.
6. Hyperparameter-tune the best-performing model with ``GridSearchCV``.
7. Persist the final tuned model (``models/model.pkl``), the fitted TF-IDF
   vectorizer (``models/tfidf.pkl``), and a metrics report
   (``models/metrics.json``) consumed by ``evaluate.py`` and the
   Streamlit app's "Model Performance" page.

Run directly with:

    python src/train.py

Author: Your Name
"""

import json
import time
import logging

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils import (
    get_logger,
    ensure_dir,
    format_seconds,
    CLEANED_DATA_PATH,
    MODELS_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
)

logger = get_logger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF configuration (fixed per project spec)
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------
def load_dataset(path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset produced by ``preprocess.py``.

    Parameters
    ----------
    path : str
        Path to ``news_cleaned.csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset does not exist yet (i.e. ``preprocess.py``
        has not been run).
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. "
            "Run `python src/preprocess.py` first."
        ) from exc

    df["content"] = df["content"].fillna("")
    df = df[df["content"].str.strip().str.len() > 0].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Stratified train/test split on the ``content`` / ``label`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with ``content`` and ``label`` columns.

    Returns
    -------
    tuple
        ``(X_train_text, X_test_text, y_train, y_test)``
    """
    X = df["content"]
    y = df["label"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_vectorizer() -> TfidfVectorizer:
    """Construct the project's TF-IDF vectorizer with fixed hyperparameters.

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer configured with
        ``max_features=10000, ngram_range=(1, 2)``.
    """
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def get_candidate_models() -> dict:
    """Return the four candidate models to compare.

    Returns
    -------
    dict[str, estimator]
        Mapping of human-readable model name to an unfitted sklearn
        estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def get_decision_scores(model, X) -> np.ndarray:
    """Return per-sample scores usable for ROC-AUC regardless of model type.

    Uses ``predict_proba`` when available (Logistic Regression, Naive
    Bayes, Random Forest); falls back to ``decision_function`` for models
    like ``LinearSVC`` which do not natively expose probabilities.

    Parameters
    ----------
    model : estimator
        A fitted sklearn classifier.
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray
        1D array of scores where higher = more likely class 1 (Real).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute the standard classification metrics for a fitted model.

    Parameters
    ----------
    model : estimator
        A fitted sklearn classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    dict
        Metrics: accuracy, precision, recall, f1, roc_auc, and the
        confusion matrix (as a nested list, JSON-serializable).
    """
    y_pred = model.predict(X_test)
    scores = get_decision_scores(model, X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, scores), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
_PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.1, 1, 5, 10],
        "solver": ["liblinear", "lbfgs"],
    },
    "Multinomial Naive Bayes": {
        "alpha": [0.01, 0.1, 0.5, 1.0],
    },
    "Linear SVM": {
        "C": [0.01, 0.1, 1, 10],
    },
    "Random Forest": {
        "n_estimators": [200, 300],
        "max_depth": [None, 50],
    },
}


def tune_best_model(model_name: str, X_train, y_train):
    """Run GridSearchCV over the best model's hyperparameter grid.

    Parameters
    ----------
    model_name : str
        Key into ``get_candidate_models()`` / ``_PARAM_GRIDS`` identifying
        which model to tune.
    X_train, y_train : array-like
        Training features/labels.

    Returns
    -------
    tuple
        ``(best_estimator, best_params, cv_best_f1)``
    """
    base_model = get_candidate_models()[model_name]
    param_grid = _PARAM_GRIDS[model_name]

    logger.info("Running GridSearchCV for '%s' with grid: %s", model_name, param_grid)
    search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info("Best params for '%s': %s (CV F1=%.4f)",
                model_name, search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_, search.best_score_


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_training_pipeline():
    """End-to-end training pipeline: load data, vectorize, train, compare,
    tune, and persist the final model + vectorizer + metrics report.

    Returns
    -------
    dict
        Full metrics report (also written to ``models/metrics.json``).
    """
    ensure_dir(MODELS_DIR)
    overall_start = time.time()

    logger.info("Loading cleaned dataset...")
    df = load_dataset()
    logger.info("Dataset loaded: %d rows", len(df))

    X_train_text, X_test_text, y_train, y_test = split_data(df)
    logger.info("Train size: %d | Test size: %d", len(X_train_text), len(X_test_text))

    logger.info("Fitting TF-IDF vectorizer (max_features=%d, ngram_range=%s)...",
                TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    logger.info("TF-IDF matrix shape: train=%s, test=%s", X_train.shape, X_test.shape)

    # -- Train & compare all candidate models -------------------------------
    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        start = time.time()
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        metrics = evaluate_model(model, X_test, y_test)
        metrics["train_time_seconds"] = round(elapsed, 2)
        results[name] = metrics
        fitted_models[name] = model

        logger.info(
            "%s -> acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f (%s)",
            name, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1_score"], metrics["roc_auc"],
            format_seconds(elapsed),
        )

    # -- Select best model by F1-score --------------------------------------
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    logger.info("Best baseline model: %s (F1=%.4f)",
                best_model_name, results[best_model_name]["f1_score"])

    # -- Hyperparameter tuning on the best model ----------------------------
    tuned_model, best_params, cv_best_f1 = tune_best_model(
        best_model_name, X_train, y_train
    )
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)
    tuned_metrics["best_params"] = best_params
    tuned_metrics["cv_f1_score"] = round(cv_best_f1, 4)

    logger.info(
        "Tuned %s -> acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f",
        best_model_name, tuned_metrics["accuracy"], tuned_metrics["precision"],
        tuned_metrics["recall"], tuned_metrics["f1_score"], tuned_metrics["roc_auc"],
    )

    # Linear SVM does not natively support predict_proba, which the
    # Streamlit app needs to show class probabilities. If it wins, wrap it
    # in a CalibratedClassifierCV to expose well-calibrated probabilities
    # without changing its decision boundary characteristics.
    final_model = tuned_model
    if best_model_name == "Linear SVM":
        logger.info("Calibrating Linear SVM to expose predict_proba()...")
        final_model = CalibratedClassifierCV(tuned_model, cv=3)
        final_model.fit(X_train, y_train)
        tuned_metrics = evaluate_model(final_model, X_test, y_test)
        tuned_metrics["best_params"] = best_params
        tuned_metrics["cv_f1_score"] = round(cv_best_f1, 4)

    # -- Persist artifacts ----------------------------------------------------
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Saved model -> %s", MODEL_PATH)
    logger.info("Saved vectorizer -> %s", VECTORIZER_PATH)

    report = {
        "best_model_name": best_model_name,
        "baseline_comparison": results,
        "tuned_model_metrics": tuned_metrics,
        "tfidf_config": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
        },
        "train_size": len(X_train_text),
        "test_size": len(X_test_text),
        "total_training_time_seconds": round(time.time() - overall_start, 2),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics report -> %s", METRICS_PATH)

    logger.info("Training pipeline complete in %s",
                format_seconds(report["total_training_time_seconds"]))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training_pipeline()
