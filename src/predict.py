"""
predict.py
==========
Inference module for the Fake News Detection System.

Loads the persisted TF-IDF vectorizer and trained model, and exposes
simple functions to classify a single article or a batch of articles as
Fake (0) or Real (1), along with confidence scores. This is the module
the Streamlit app (``app.py``) calls under the hood, and it can equally
be used as a standalone CLI or imported into other scripts/notebooks.

Run directly with:

    python src/predict.py --text "Some news article text here..."

Author: Your Name
"""

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

from utils import get_logger, MODEL_PATH, VECTORIZER_PATH
from preprocess import clean_text

logger = get_logger(__name__)

_MODEL = None
_VECTORIZER = None


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted before artifacts are loaded."""


def load_model_and_vectorizer(model_path: str = MODEL_PATH,
                               vectorizer_path: str = VECTORIZER_PATH):
    """Load (and cache) the trained model and TF-IDF vectorizer.

    Subsequent calls reuse the module-level cache rather than hitting disk
    again, which matters for a Streamlit app that reruns the script on
    every user interaction.

    Parameters
    ----------
    model_path : str
        Path to ``model.pkl``.
    vectorizer_path : str
        Path to ``tfidf.pkl``.

    Returns
    -------
    tuple
        ``(model, vectorizer)``

    Raises
    ------
    FileNotFoundError
        If the artifacts do not exist (i.e. ``train.py`` was never run).
    """
    global _MODEL, _VECTORIZER

    if _MODEL is not None and _VECTORIZER is not None:
        return _MODEL, _VECTORIZER

    try:
        _MODEL = joblib.load(model_path)
        _VECTORIZER = joblib.load(vectorizer_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not load model/vectorizer from '{model_path}' / "
            f"'{vectorizer_path}'. Run `python src/train.py` first."
        ) from exc

    logger.info("Model and vectorizer loaded successfully.")
    return _MODEL, _VECTORIZER


def _get_probabilities(model, X) -> np.ndarray:
    """Return class probabilities as a 2-column array [P(Fake), P(Real)].

    Falls back to a min-max-normalized decision function if the model
    does not natively support ``predict_proba`` (shouldn't happen for the
    shipped model, since ``train.py`` calibrates LinearSVC when it wins,
    but this keeps ``predict.py`` robust to future model swaps).

    Parameters
    ----------
    model : estimator
        A fitted sklearn-compatible classifier.
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, 2)`` array of probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    scores = model.decision_function(X)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return np.column_stack([1 - scores, scores])


def predict_single(text: str, title: str = "") -> dict:
    """Classify a single news article as Fake or Real.

    Parameters
    ----------
    text : str
        The article body text.
    title : str, optional
        The article title, if available. Concatenated with the body the
        same way it was during training, for consistency.

    Returns
    -------
    dict
        ``{"label": "Fake" | "Real", "label_code": 0 | 1,
           "confidence": float, "prob_fake": float, "prob_real": float}``

    Raises
    ------
    ValueError
        If ``text`` is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    model, vectorizer = load_model_and_vectorizer()

    cleaned = (clean_text(title) + " " + clean_text(text)).strip()
    if not cleaned:
        raise ValueError(
            "After cleaning, no usable text remained (input may have been "
            "only stopwords, numbers, or punctuation)."
        )

    X = vectorizer.transform([cleaned])
    probabilities = _get_probabilities(model, X)[0]
    prob_fake, prob_real = float(probabilities[0]), float(probabilities[1])
    label_code = int(prob_real >= prob_fake)

    return {
        "label": "Real" if label_code == 1 else "Fake",
        "label_code": label_code,
        "confidence": round(max(prob_fake, prob_real), 4),
        "prob_fake": round(prob_fake, 4),
        "prob_real": round(prob_real, 4),
    }


def predict_batch(df: pd.DataFrame, text_col: str = "text",
                   title_col: str = None) -> pd.DataFrame:
    """Classify a batch of articles supplied as a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a text column.
    text_col : str, optional
        Name of the column holding article body text. Defaults to
        ``"text"``.
    title_col : str, optional
        Name of the column holding article titles, if present. When
        omitted, only body text is used.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with four new columns appended:
        ``prediction``, ``confidence``, ``prob_fake``, ``prob_real``.

    Raises
    ------
    ValueError
        If ``text_col`` is not present in ``df``.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in the uploaded file.")

    model, vectorizer = load_model_and_vectorizer()

    titles = df[title_col].astype(str) if title_col and title_col in df.columns else ""
    texts = df[text_col].astype(str)

    combined_raw = (titles + " " + texts) if title_col else texts
    cleaned = combined_raw.apply(clean_text)

    # Guard against fully-empty rows after cleaning (would otherwise error
    # inside the vectorizer's `transform` on an all-empty vocabulary match).
    cleaned_safe = cleaned.replace("", "empty_article_placeholder")

    X = vectorizer.transform(cleaned_safe)
    probabilities = _get_probabilities(model, X)

    result_df = df.copy()
    result_df["prediction"] = np.where(probabilities[:, 1] >= probabilities[:, 0], "Real", "Fake")
    result_df["confidence"] = np.round(probabilities.max(axis=1), 4)
    result_df["prob_fake"] = np.round(probabilities[:, 0], 4)
    result_df["prob_real"] = np.round(probabilities[:, 1], 4)

    return result_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a news article as Fake or Real."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="The article body text to classify.",
    )
    parser.add_argument(
        "--title", type=str, default="",
        help="Optional article title.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()

    result = predict_single(args.text, args.title)
    print(f"\nPrediction: {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"  P(Fake) = {result['prob_fake'] * 100:.2f}%")
    print(f"  P(Real) = {result['prob_real'] * 100:.2f}%")
