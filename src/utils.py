"""
utils.py
========
Shared utilities used across the Fake News Detection System modules:
project path resolution, logging setup, and small reusable helpers.

Author: Your Name
"""

import os
import logging
import sys

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the repository root regardless of the current
# working directory the scripts are invoked from (VS Code, terminal, cron,
# Streamlit, etc.), by anchoring off this file's own location.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_FAKE_PATH = os.path.join(DATASET_DIR, "Fake.csv")
RAW_TRUE_PATH = os.path.join(DATASET_DIR, "True.csv")
CLEANED_DATA_PATH = os.path.join(DATASET_DIR, "news_cleaned.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Ensures handlers are not duplicated if called multiple times for the
    same module name (e.g. on module re-import within Streamlit's
    hot-reload).

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int, optional
        Logging level, defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Create a directory (including parents) if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def format_seconds(seconds: float) -> str:
    """Format a duration in seconds as a human-readable ``Xm Ys`` string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable duration, e.g. ``"2m 14s"``.
    """
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
