"""
preprocess.py
=============
Data loading and text-cleaning pipeline for the Fake News Detection System.

Responsibilities
-----------------
1. Load the raw ``Fake.csv`` / ``True.csv`` source files.
2. Merge them into a single labeled DataFrame (0 = Fake, 1 = Real).
3. Clean and normalize the raw text (lowercasing, punctuation/URL/HTML/
   number removal, tokenization, stopword removal, lemmatization).
4. Persist the cleaned dataset to ``dataset/news_cleaned.csv`` so that
   downstream modules (``train.py``, ``predict.py``) never need to repeat
   this expensive step.

This module can be run directly:

    python src/preprocess.py

or imported and reused, e.g. by ``predict.py`` to clean a single
user-submitted article at inference time with the exact same logic used
during training (this consistency is critical — training/inference skew
in text preprocessing is one of the most common bugs in NLP projects).

Author: Your Name
"""

import os
import re
import string
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
_REQUIRED_NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if they are not already present.

    Downloading only happens once per environment; subsequent calls are
    effectively no-ops since NLTK checks for local availability first.
    """
    for package, path in _REQUIRED_NLTK_PACKAGES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading missing NLTK package: %s", package)
            nltk.download(package, quiet=True)


ensure_nltk_resources()

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Compiled regex patterns (compiled once at import time for performance)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_NUMBER_PATTERN = re.compile(r"\d+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
_MULTISPACE_PATTERN = re.compile(r"\s+")

# --- Source-leakage patterns ------------------------------------------------
# In the raw dataset, ~99.8% of REAL articles open with a wire-service
# dateline like "WASHINGTON (Reuters) -" while <2% of FAKE articles ever
# mention "Reuters" at all. Left in place, the model learns to detect the
# publisher/dateline instead of any actual fake-vs-real language signal:
# it scores ~99% on the held-out test split (which has the same leak) but
# collapses to ~50/50 confidence on real-world text a user types in, since
# that text never contains a Reuters dateline. Strip these artifacts
# before they reach the vectorizer.
_DATELINE_PATTERN = re.compile(
    r"^[A-Za-z\s,.\-]{2,40}\((?:Reuters|AP|AFP)\)\s*-\s*"
)
_OUTLET_PATTERN = re.compile(
    r"\b(reuters|associated press|afp|21st century wire|21wire)\b", re.IGNORECASE
)
_TWITTER_HANDLE_PATTERN = re.compile(r"@\w+")
_PIC_TWITTER_PATTERN = re.compile(r"pic\.twitter\.com/\S+")


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------
def load_raw_data(dataset_dir: str) -> pd.DataFrame:
    """Load and merge the raw ``Fake.csv`` and ``True.csv`` files.

    Parameters
    ----------
    dataset_dir : str
        Directory containing ``Fake.csv`` and ``True.csv``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with an added ``label`` column
        (0 = Fake, 1 = Real), shuffled.

    Raises
    ------
    FileNotFoundError
        If either source CSV is missing.
    """
    fake_path = os.path.join(dataset_dir, "Fake.csv")
    true_path = os.path.join(dataset_dir, "True.csv")

    for path in (fake_path, true_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    logger.info("Loading raw datasets from %s", dataset_dir)
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("Loaded %d fake and %d real articles (%d total)",
                len(fake_df), len(true_df), len(df))
    return df


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Apply the full text-cleaning pipeline to a single string.

    Steps: lowercase -> remove URLs -> remove HTML tags -> remove numbers
    -> remove punctuation/special characters -> tokenize -> remove
    stopwords -> lemmatize -> rejoin.

    Parameters
    ----------
    text : str
        Raw input text (article title or body).

    Returns
    -------
    str
        Cleaned, lemmatized text as a single space-joined string. Returns
        an empty string for null/non-string input rather than raising, so
        this function is safe to use inside ``DataFrame.apply``.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Strip wire-service datelines / outlet names / twitter cruft FIRST,
    # before lowercasing destroys the "(Reuters)" capitalization pattern.
    text = _DATELINE_PATTERN.sub(" ", text)
    text = _OUTLET_PATTERN.sub(" ", text)
    text = _PIC_TWITTER_PATTERN.sub(" ", text)
    text = _TWITTER_HANDLE_PATTERN.sub(" ", text)

    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _HTML_PATTERN.sub(" ", text)
    text = _NUMBER_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    if not text:
        return ""

    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 1
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline over a raw merged DataFrame.

    Drops duplicates, drops rows with empty article bodies, cleans
    ``title`` and ``text``, builds a combined ``content`` field (what the
    model will actually vectorize), and returns a tidy DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame as returned by :func:`load_raw_data`, with
        columns ``title``, ``text``, ``subject``, ``date``, ``label``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ``title``, ``text``, ``label``,
        ``clean_title``, ``clean_text``, ``content``.
    """
    logger.info("Starting preprocessing on %d rows", len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    logger.info("Dropped %d duplicate rows", before - len(df))

    before = len(df)
    df = df[df["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Dropped %d rows with empty article text", before - len(df))

    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)

    logger.info("Cleaning titles...")
    df["clean_title"] = df["title"].apply(clean_text)

    logger.info("Cleaning article text (this may take a few minutes)...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Combined field used for TF-IDF: title carries strong signal, so we
    # weight it in lightly by concatenating it with the body.
    df["content"] = (df["clean_title"] + " " + df["clean_text"]).str.strip()

    before = len(df)
    df = df[df["content"].str.len() > 0].reset_index(drop=True)
    logger.info("Dropped %d rows with empty content after cleaning", before - len(df))

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df[["title", "text", "label", "clean_title", "clean_text", "content"]]


def run_preprocessing_pipeline(
    dataset_dir: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """End-to-end pipeline: load raw data, clean it, and save the result.

    Parameters
    ----------
    dataset_dir : str, optional
        Directory containing ``Fake.csv``/``True.csv``. Defaults to
        ``<project_root>/dataset``.
    output_path : str, optional
        Destination CSV path. Defaults to
        ``<project_root>/dataset/news_cleaned.csv``.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame that was written to disk.
    """
    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")
    output_path = output_path or os.path.join(dataset_dir, "news_cleaned.csv")

    raw_df = load_raw_data(dataset_dir)
    clean_df = preprocess_dataframe(raw_df)

    clean_df.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset to %s", output_path)

    return clean_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_preprocessing_pipeline()
