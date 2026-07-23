"""Streamlit UI for the fake news detector."""
import streamlit as st

from src.credibility import credibility_score
from src.explain import explain_prediction, explanation_to_html
from src.features import load_vectorizer
from src.models.logistic import load_model, predict_proba_text

MIN_LENGTH = 50


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
st.caption(
    "Paste an article's text to get a model verdict, the reasoning behind it, "
    "and an independent credibility signal."
)
st.caption("Portfolio demo trained on the ISOT Fake News dataset.")

model, vectorizer = load_artifacts()

if model is None or vectorizer is None:
    st.error(
        "No trained model/vectorizer found. Run `python -m src.models.logistic` "
        "first to train and save the baseline artifacts."
    )
    st.stop()

text = st.text_area("Article text", height=200)
url = st.text_input("Source URL (optional)")

if st.button("Analyze"):
    if len(text.strip()) < MIN_LENGTH:
        st.warning(f"Please paste a longer article (at least {MIN_LENGTH} characters).")
        st.stop()
    # Stashed so the "Explain" button below can reuse it across reruns.
    st.session_state["analyzed_text"] = text

if "analyzed_text" in st.session_state:
    analyzed_text = st.session_state["analyzed_text"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model verdict")
        label, confidence = predict_proba_text(analyzed_text, model, vectorizer)
        pct = confidence * 100
        if label == "REAL":
            st.success(f"Verdict: REAL ({pct:.1f}% confidence)")
        else:
            st.error(f"Verdict: FAKE ({pct:.1f}% confidence)")
        st.progress(confidence)

    with col2:
        st.subheader("Credibility score")
        result = credibility_score(analyzed_text, url or None)
        st.metric("Score (0-100)", result["final_score"])
        for reason in result["reasons"]:
            st.markdown(f"- {reason}")

    st.divider()

    if st.button("Explain this prediction"):
        # LIME re-runs the model on hundreds of perturbed samples through
        # spaCy preprocessing, so this is kept separate and only run on demand.
        with st.spinner("Analyzing word importance… this takes ~20 seconds"):
            explanation, _ = explain_prediction(analyzed_text)
        st.components.v1.html(explanation_to_html(explanation), height=350, scrolling=True)

st.divider()
st.caption(
    "This app serves the TF-IDF + Logistic Regression baseline. A fine-tuned "
    "DistilBERT model scored higher (99.87%) but wasn't shipped here for "
    "footprint reasons — see `reports/MODEL_COMPARISON.md` for the full "
    "comparison. The training data (ISOT) is stylistically easy to separate, "
    "so real-world performance on other sources will likely be lower."
)
