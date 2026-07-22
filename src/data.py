"""Load and clean the fake/real news dataset."""
import re

import pandas as pd

from src.config import CLEAN_CSV, FAKE_CSV, LABEL_FAKE, LABEL_REAL, RANDOM_SEED, TRUE_CSV

# Matches the Reuters wire-service dateline, e.g. "WASHINGTON (Reuters) - ".
# Almost every REAL article starts with this and no FAKE article does, so a
# model would learn to key off it instead of the actual writing. We strip it.
REUTERS_DATELINE = re.compile(r"^\s*[^()]{0,60}\(Reuters\)\s*-?\s*")


def strip_reuters_dateline(text: str) -> str:
    return REUTERS_DATELINE.sub("", text, count=1)


def load_and_clean_data() -> pd.DataFrame:
    fake = pd.read_csv(FAKE_CSV)
    true = pd.read_csv(TRUE_CSV)

    fake["label"] = LABEL_FAKE
    true["label"] = LABEL_REAL

    true["text"] = true["text"].apply(strip_reuters_dateline)

    df = pd.concat([fake, true], ignore_index=True)

    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    df["content"] = df["content"].str.strip()
    df = df[df["content"].astype(bool)]
    df = df.drop_duplicates(subset="content")

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df = df[["content", "label"]]
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)

    print(f"Final shape: {df.shape}")
    print("Label balance:")
    print(df["label"].value_counts(normalize=True))
    print("\nSample rows:")
    print(df.sample(2, random_state=RANDOM_SEED))

    return df


if __name__ == "__main__":
    load_and_clean_data()
