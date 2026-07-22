"""Streamlit UI for the fake news detector."""
import streamlit as st

from src.features import load_vectorizer
from src.models.logistic import load_model, predict_proba_text

MIN_LENGTH = 20


@st.cache_resource
def load_artifacts():
    """Load the fitted model + vectorizer once per session. Never retrains/refits."""
    try:
        model = load_model()
        vectorizer = load_vectorizer()
    except FileNotFoundError:
        return None, None
    return model, vectorizer


st.title("Fake News Detector")
st.caption("Paste an article's text to get a baseline REAL/FAKE prediction.")

model, vectorizer = load_artifacts()

if model is None or vectorizer is None:
    st.error(
        "No trained model/vectorizer found. Run `python -m src.models.logistic` "
        "first to train and save the baseline artifacts."
    )
else:
    text = st.text_area("Article text", height=250)

    if st.button("Analyze"):
        if len(text.strip()) < MIN_LENGTH:
            st.warning("Please paste a longer article.")
            st.stop()

        label, confidence = predict_proba_text(text, model, vectorizer)
        pct = confidence * 100

        if label == "REAL":
            st.success(f"Verdict: REAL ({pct:.1f}% confidence)")
        else:
            st.error(f"Verdict: FAKE ({pct:.1f}% confidence)")

        st.progress(confidence)
        st.caption(
            "This is a baseline model. Confidence reflects the model's certainty, "
            "not ground truth."
        )
