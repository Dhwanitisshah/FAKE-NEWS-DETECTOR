"""Baseline Logistic Regression classifier."""
import joblib
from sklearn.linear_model import LogisticRegression

from src.config import LABEL_FAKE, LABEL_REAL, MODELS_DIR, RANDOM_SEED
from src.features import TfidfVectorizer, build_features, load_data, save_vectorizer
from src.preprocess import preprocess_text

MODEL_PATH = MODELS_DIR / "logistic_model.pkl"


def train_baseline() -> tuple[LogisticRegression, "object", "object", TfidfVectorizer]:
    """Train the baseline classifier on TF-IDF features.

    Returns the fitted model along with the held-out test set and the
    fitted vectorizer, since the evaluator needs all three.
    """
    X, y = load_data()
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = build_features(X, y)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    model.fit(X_train_vec, y_train)

    return model, X_test_vec, y_test, vectorizer


def save_model(model: LogisticRegression) -> None:
    """Persist the fitted model to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model() -> LogisticRegression:
    """Load the fitted model for inference."""
    return joblib.load(MODEL_PATH)


def predict_proba_text(
    text: str, model: LogisticRegression, vectorizer: TfidfVectorizer
) -> tuple[str, float]:
    """Predict FAKE/REAL for a single raw string, with confidence.

    Applies the same cleaning/lemmatizing used at training time and
    transforms with the already-fitted vectorizer (never refit here).
    """
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    pred_class = model.classes_[probs.argmax()]
    confidence = float(probs.max())

    label = "FAKE" if pred_class == LABEL_FAKE else "REAL"
    return label, confidence


if __name__ == "__main__":
    model, X_test_vec, y_test, vectorizer = train_baseline()

    save_model(model)
    save_vectorizer(vectorizer)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {MODELS_DIR / 'tfidf_vectorizer.pkl'}")

    train_acc = model.score(X_test_vec, y_test)
    print(f"\nTest accuracy: {train_acc:.4f}")

    samples = [
        "The Federal Reserve announced Wednesday it would hold interest rates "
        "steady, citing ongoing concerns about inflation and labor market data.",
        "SHOCKING: Scientists CONFIRM the moon is actually a hologram controlled "
        "by secret government agents, leaked documents reveal!!!",
    ]

    print("\nSample predictions:")
    for text in samples:
        label, confidence = predict_proba_text(text, model, vectorizer)
        print(f"  [{label} ({confidence:.4f})] {text[:80]}...")
