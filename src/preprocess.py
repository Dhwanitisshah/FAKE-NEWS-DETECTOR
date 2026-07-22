"""Convert raw article text into clean, lemmatized tokens for modeling."""
import re
import time

import pandas as pd
import spacy

from src.config import CLEAN_CSV, PROCESSED_DATA_DIR

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError as e:
    raise OSError(
        "spaCy model 'en_core_web_sm' is not installed. "
        "Run: python -m spacy download en_core_web_sm"
    ) from e

nlp.max_length = 2_000_000

MIN_TOKEN_LEN = 2

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_RE = re.compile(r"&\w+;")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = HTML_ENTITY_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _lemmatize_doc(doc) -> str:
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token.lemma_) >= MIN_TOKEN_LEN
    )


def preprocess_series(texts) -> list:
    cleaned = [clean_text(t) for t in texts]
    docs = nlp.pipe(cleaned, batch_size=500, n_process=1)
    return [_lemmatize_doc(doc) for doc in docs]


def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    return _lemmatize_doc(doc)


if __name__ == "__main__":
    start = time.time()

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
    df["clean_content"] = preprocess_series(df["content"])

    empty_mask = df["clean_content"].str.strip().eq("")
    n_dropped = empty_mask.sum()
    df = df[~empty_mask]

    df = df[["content", "clean_content", "label"]]

    out_path = PROCESSED_DATA_DIR / "preprocessed_news.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Final shape: {df.shape}")
    print("Label balance:")
    print(df["label"].value_counts(normalize=True))
    print(f"Rows dropped as empty after cleaning: {n_dropped}")

    print("\nBefore/after examples:")
    for _, row in df.sample(2, random_state=42).iterrows():
        print("RAW: ", row["content"][:200])
        print("CLEAN:", row["clean_content"][:200])
        print()
