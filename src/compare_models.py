"""Records and visualizes the already-known three-model comparison.

Does not train or load any models — just the final numbers from evaluation runs.
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODELS_DIR

REPORTS_DIR = MODELS_DIR.parent / "reports"

RESULTS = [
    {
        "model": "Baseline (TF-IDF + LogReg)",
        "accuracy": 0.9861,
        "fake_recall": 0.9785,
        "false_negatives": 77,
        "false_positives": 32,
    },
    {
        "model": "BiLSTM",
        "accuracy": 0.9900,
        "fake_recall": 0.9872,
        "false_negatives": 46,
        "false_positives": 32,
    },
    {
        "model": "DistilBERT",
        "accuracy": 0.9987,
        "fake_recall": 0.9983,
        "false_negatives": 6,
        "false_positives": 4,
    },
]


def build_comparison_table() -> pd.DataFrame:
    """Build and print the comparison table, sorted by FAKE recall."""
    df = pd.DataFrame(RESULTS).sort_values("fake_recall").reset_index(drop=True)
    print(df.to_string(index=False))
    return df


def plot_error_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart of false negatives vs false positives per model."""
    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_fn = ax.bar(
        [i - width / 2 for i in x], df["false_negatives"], width, label="False Negatives"
    )
    bars_fp = ax.bar(
        [i + width / 2 for i in x], df["false_positives"], width, label="False Positives"
    )

    for bars in (bars_fn, bars_fp):
        ax.bar_label(bars, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Count")
    ax.set_title("False Negatives vs False Positives by Model")
    ax.legend()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = build_comparison_table()
    plot_error_comparison(df)
    print(f"\nSaved chart to {REPORTS_DIR / 'model_comparison.png'}")
