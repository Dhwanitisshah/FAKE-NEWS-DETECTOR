"""Feature engineering: TF-IDF vectors and BERT embeddings."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


def load_data() -> tuple[pd.Series, pd.Series]:
    """Load preprocessed news, dropping any residual null/empty clean_content rows."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "preprocessed_news.csv", encoding="utf-8")

    df["clean_content"] = df["clean_content"].fillna("")
    df = df[df["clean_content"].str.strip().ne("")]

    X = df["clean_content"]
    y = df["label"].astype(int)
    return X, y


def build_features(
    X: pd.Series, y: pd.Series
) -> tuple:
    """Stratified train/test split, then TF-IDF vectorize.

    The vectorizer is fit on X_train only and used to transform both splits.
    Fitting on the full dataset (or on test data) would leak test-set
    vocabulary/IDF statistics into training, inflating evaluation metrics and
    producing a feature space that doesn't reflect a true holdout.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=5, sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """Persist the fitted vectorizer so inference reuses the exact training vocabulary/IDF."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def load_vectorizer() -> TfidfVectorizer:
    """Load the fitted vectorizer for inference. Never refit at inference time."""
    return joblib.load(VECTORIZER_PATH)


if __name__ == "__main__":
    X, y = load_data()
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = build_features(X, y)
    save_vectorizer(vectorizer)

    print(f"Train shape: {X_train_vec.shape}")
    print(f"Test shape: {X_test_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\nTrain label balance:")
    print(y_train.value_counts(normalize=True))
    print("\nTest label balance:")
    print(y_test.value_counts(normalize=True))
