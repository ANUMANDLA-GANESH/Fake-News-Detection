"""
app.py
======
Streamlit web application for the Fake News Detection System.

Pages
-----
- **Detect**       : paste/type a single article, get an instant prediction
                      with confidence + class probabilities.
- **Batch Predict** : upload a CSV of articles, get predictions for all
                      rows, view + download the results.
- **History**       : session-local log of every prediction made this
                      session, downloadable as CSV.

Run with:

    streamlit run app.py

Author: Your Name
"""

import io
import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# src/*.py modules import each other with bare module names (e.g.
# `from utils import ...`) so that each script also works standalone via
# `python src/train.py`. Adding src/ to sys.path here keeps app.py
# consistent with that same import style rather than requiring a
# packaged `src.` namespace.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import METRICS_PATH, MODELS_DIR  # noqa: E402
from predict import predict_single, predict_batch, load_model_and_vectorizer  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling — dark-mode-friendly, uses CSS variables Streamlit exposes
# rather than hardcoded light-only colors, plus an accent palette distinct
# from Streamlit's default red.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --accent: #3fb6a8;
        --accent-dark: #2c8f84;
        --fake-color: #e35d5d;
        --real-color: #3fb6a8;
    }

    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.25rem;
    }
    .app-header h1 {
        font-size: 2rem;
        margin: 0;
    }
    .app-subtitle {
        opacity: 0.7;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 1rem;
    }
    .result-card.fake { border-left: 6px solid var(--fake-color); }
    .result-card.real { border-left: 6px solid var(--real-color); }

    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .result-label.fake { color: var(--fake-color); }
    .result-label.real { color: var(--real-color); }

    .metric-pill {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        background: rgba(63, 182, 168, 0.15);
        border: 1px solid var(--accent);
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }

    .stButton>button {
        background-color: var(--accent);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
    }
    .stButton>button:hover {
        background-color: var(--accent-dark);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts


def _log_prediction(source: str, title: str, snippet: str, result: dict) -> None:
    """Append a prediction result to the in-session history log.

    Parameters
    ----------
    source : str
        Where the prediction came from, e.g. ``"Single"`` or ``"Batch"``.
    title : str
        Article title (or empty string).
    snippet : str
        A short preview of the article text.
    result : dict
        The dict returned by ``predict_single``.
    """
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "title": title[:80],
        "snippet": snippet[:120],
        "prediction": result["label"],
        "confidence": result["confidence"],
        "prob_fake": result["prob_fake"],
        "prob_real": result["prob_real"],
    })


@st.cache_resource(show_spinner=False)
def _load_metrics() -> dict:
    """Load the metrics.json report produced by train.py (cached)."""
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading model...")
def _warm_model():
    """Pre-load the model/vectorizer once per app session (cached)."""
    return load_model_and_vectorizer()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🛰️ Fake News Detector")
st.sidebar.caption("AI/ML-powered news authenticity classifier")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Detect", "📂 Batch Predict", "🕓 History"],
    label_visibility="collapsed",
)

st.sidebar.divider()
metrics_preview = _load_metrics()
if metrics_preview:
    tuned = metrics_preview.get("tuned_model_metrics", {})
    st.sidebar.markdown("**Current model**")
    st.sidebar.write(metrics_preview.get("best_model_name", "N/A"))
    st.sidebar.metric("Test Accuracy", f"{tuned.get('accuracy', 0) * 100:.2f}%")
    st.sidebar.metric("Test F1-score", f"{tuned.get('f1_score', 0) * 100:.2f}%")
st.sidebar.divider()
st.sidebar.caption("Built with Python · scikit-learn · Streamlit")


# ---------------------------------------------------------------------------
# Page: Detect (single article)
# ---------------------------------------------------------------------------
def render_detect_page():
    st.markdown(
        '<div class="app-header"><h1>🔍 Detect Fake News</h1></div>'
        '<div class="app-subtitle">Paste a news article below and let the model '
        'classify it as Fake or Real, with a confidence score.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        title = st.text_input("Article title (optional)", placeholder="e.g. Fed Holds Rates Steady")
        text = st.text_area(
            "Article text",
            height=260,
            placeholder="Paste the full article body here...",
        )

        predict_clicked = st.button("🚀 Predict", width='content')

        if predict_clicked:
            if not text or not text.strip():
                st.error("⚠️ Please enter some article text before predicting.")
            else:
                try:
                    with st.spinner("Analyzing article..."):
                        result = predict_single(text, title)
                    _log_prediction("Single", title, text, result)

                    css_class = "real" if result["label"] == "Real" else "fake"
                    icon = "✅" if result["label"] == "Real" else "🚩"

                    st.markdown(
                        f"""
                        <div class="result-card {css_class}">
                            <div class="result-label {css_class}">{icon} {result['label']} News</div>
                            <div style="margin-top:0.5rem;">
                                <span class="metric-pill">Confidence: {result['confidence']*100:.2f}%</span>
                                <span class="metric-pill">P(Real): {result['prob_real']*100:.2f}%</span>
                                <span class="metric-pill">P(Fake): {result['prob_fake']*100:.2f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(result["prob_real"], text="P(Real)")
                    st.progress(result["prob_fake"], text="P(Fake)")

                except ValueError as ve:
                    st.error(f"⚠️ {ve}")
                except FileNotFoundError as fe:
                    st.error(f"⚠️ {fe}")
                except Exception as exc:  # noqa: BLE001 — surface unexpected errors to the user
                    st.error(f"⚠️ An unexpected error occurred: {exc}")

    with col2:
        st.markdown("#### 💡 Tips")
        st.info(
            "- Longer excerpts (a few sentences or more) generally give more "
            "reliable predictions than a single headline.\n"
            "- This model was trained on political news articles from "
            "2016–2017, so it performs best on similar topics/style.\n"
            "- Confidence close to 50% means the model is uncertain — treat "
            "the result cautiously."
        )


# ---------------------------------------------------------------------------
# Page: Batch Predict
# ---------------------------------------------------------------------------
def render_batch_page():
    st.markdown(
        '<div class="app-header"><h1>📂 Batch Prediction</h1></div>'
        '<div class="app-subtitle">Upload a CSV file with a "text" column '
        '(and optionally a "title" column) to classify many articles at once.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Could not read the CSV file: {exc}")
            return

        st.write(f"Loaded **{len(df)}** rows. Preview:")
        st.dataframe(df.head(5), width='stretch')

        cols = list(df.columns)
        text_col = st.selectbox(
            "Which column contains the article text?",
            options=cols,
            index=cols.index("text") if "text" in cols else 0,
        )
        title_col_options = ["(none)"] + cols
        title_col = st.selectbox(
            "Which column contains the article title? (optional)",
            options=title_col_options,
            index=title_col_options.index("title") if "title" in cols else 0,
        )
        title_col = None if title_col == "(none)" else title_col

        if st.button("🚀 Run Batch Prediction"):
            try:
                with st.spinner(f"Classifying {len(df)} articles..."):
                    result_df = predict_batch(df, text_col=text_col, title_col=title_col)

                st.success(f"✅ Classified {len(result_df)} articles.")
                st.dataframe(result_df, width='stretch')

                fake_count = (result_df["prediction"] == "Fake").sum()
                real_count = (result_df["prediction"] == "Real").sum()
                c1, c2 = st.columns(2)
                c1.metric("🚩 Predicted Fake", fake_count)
                c2.metric("✅ Predicted Real", real_count)

                for _, row in result_df.head(20).iterrows():
                    result = {
                        "label": row["prediction"], "confidence": row["confidence"],
                        "prob_fake": row["prob_fake"], "prob_real": row["prob_real"],
                    }
                    _log_prediction(
                        "Batch",
                        str(row[title_col]) if title_col else "",
                        str(row[text_col]),
                        result,
                    )

                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="fake_news_predictions.csv",
                    mime="text/csv",
                )

            except ValueError as ve:
                st.error(f"⚠️ {ve}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"⚠️ An unexpected error occurred during batch prediction: {exc}")
    else:
        st.info("👆 Upload a CSV file to get started. It must contain a text column.")


# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------
def render_history_page():
    st.markdown(
        '<div class="app-header"><h1>🕓 Prediction History</h1></div>'
        '<div class="app-subtitle">A session-local log of every prediction made '
        'in this browser session (cleared when the app restarts).</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No predictions yet this session. Try the **Detect** or **Batch Predict** pages.")
        return

    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        csv_buffer = io.StringIO()
        hist_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_buffer.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    with c2:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
_warm_model()

if page == "🔍 Detect":
    render_detect_page()
elif page == "📂 Batch Predict":
    render_batch_page()
elif page == "🕓 History":
    render_history_page()
