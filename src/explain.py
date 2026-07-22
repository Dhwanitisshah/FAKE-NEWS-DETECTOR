"""LIME explanations for the baseline (TF-IDF + Logistic Regression) pipeline."""
from typing import Callable

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import LABEL_FAKE, LABEL_REAL
from src.features import load_vectorizer
from src.models.logistic import load_model
from src.preprocess import preprocess_text

CLASS_NAMES = ["REAL", "FAKE"]  # index 0 = LABEL_REAL, index 1 = LABEL_FAKE


def make_predict_fn(
    model: LogisticRegression, vectorizer: TfidfVectorizer
) -> Callable[[list[str]], np.ndarray]:
    """Build a predict_proba fn over raw text, for LIME's perturbed samples.

    Applies the same preprocess_text -> vectorizer.transform pipeline used at
    training time, so LIME's samples get identical treatment to real predictions.
    """

    def predict_fn(raw_texts: list[str]) -> np.ndarray:
        cleaned = [preprocess_text(t) for t in raw_texts]
        vecs = vectorizer.transform(cleaned)
        probs = model.predict_proba(vecs)
        # model.classes_ is sorted ascending ([LABEL_REAL, LABEL_FAKE] = [0, 1]),
        # which already matches CLASS_NAMES order.
        assert list(model.classes_) == [LABEL_REAL, LABEL_FAKE]
        return probs

    return predict_fn


def explain_prediction(
    text: str, num_features: int = 10
) -> tuple[object, list[tuple[str, float]]]:
    """Return a LIME explanation for raw text, plus (word, weight) pairs.

    Positive weight pushes toward FAKE, negative weight pushes toward REAL.
    """
    model = load_model()
    vectorizer = load_vectorizer()
    predict_fn = make_predict_fn(model, vectorizer)

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    explanation = explainer.explain_instance(
        text, predict_fn, num_features=num_features, labels=(LABEL_FAKE,)
    )
    word_weights = explanation.as_list(label=LABEL_FAKE)

    return explanation, word_weights


def explanation_to_html(explanation) -> str:
    """Render a LIME explanation as inline HTML for the Streamlit app."""
    return explanation.as_html()


if __name__ == "__main__":
    examples = {
        "REAL-sounding": (
            "The Federal Reserve announced Wednesday it would hold interest rates "
            "steady, citing ongoing concerns about inflation and labor market data."
        ),
        "FAKE-sounding": (
            "SHOCKING: Scientists CONFIRM the moon is actually a hologram controlled "
            "by secret government agents, leaked documents reveal!!!"
        ),
    }

    for name, text in examples.items():
        _, word_weights = explain_prediction(text)
        print(f"\n{name}: {text[:80]}...")
        for word, weight in word_weights:
            direction = "FAKE" if weight > 0 else "REAL"
            print(f"  {str(word)!r:20} {weight:+.4f}  (-> {direction})")
