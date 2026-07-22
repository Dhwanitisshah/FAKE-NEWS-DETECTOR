"""Model-agnostic evaluation harness: metrics, confusion matrix, plain-English summary.

Every model in this project (Logistic Regression now, LSTM/BERT later) is judged
by this module. Accuracy alone is misleading here since the two classes differ
by source/style, so per-class precision/recall and the confusion matrix matter more.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import LABEL_FAKE, LABEL_REAL, MODELS_DIR

REPORTS_DIR = MODELS_DIR.parent / "reports"


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Compute accuracy, FAKE-class precision/recall/F1, and macro-F1.

    FAKE is the positive class, per the project's label convention. Prints a
    full classification_report for a human-readable breakdown of both classes.
    """
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label=LABEL_FAKE),
        "recall_fake": recall_score(y_true, y_pred, pos_label=LABEL_FAKE),
        "f1_fake": f1_score(y_true, y_pred, pos_label=LABEL_FAKE),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    print(f"\nClassification report ({model_name}):")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Render a labeled 2x2 confusion matrix and save it to reports/, headless."""
    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_REAL, LABEL_FAKE])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / f"confusion_{model_name}.png", bbox_inches="tight")
    plt.close(fig)


def explain_metrics_plainly(metrics: dict) -> str:
    """Translate the metrics dict into a plain-English summary for the README/demo."""
    return (
        f"On the held-out test set, the {metrics['model_name']} model was correct "
        f"{metrics['accuracy']:.1%} of the time overall. Focusing on fake news "
        f"specifically: recall of {metrics['recall_fake']:.2f} means we catch "
        f"{metrics['recall_fake']:.0%} of the fake articles in the data, while "
        f"precision of {metrics['precision_fake']:.2f} means that when we flag an "
        f"article as fake, we're right {metrics['precision_fake']:.0%} of the time. "
        f"The macro-averaged F1 score of {metrics['f1_macro']:.2f} summarizes "
        f"performance across both classes equally, so it isn't skewed by class size."
    )


if __name__ == "__main__":
    from src.models.logistic import train_baseline

    model, X_test_vec, y_test, _vectorizer = train_baseline()
    y_pred = model.predict(X_test_vec)

    results = evaluate_model(y_test, y_pred, model_name="logistic_regression")
    print(results)

    plot_confusion_matrix(y_test, y_pred, model_name="logistic_regression")
    print(f"\nSaved confusion matrix to {REPORTS_DIR / 'confusion_logistic_regression.png'}")

    print("\n" + explain_metrics_plainly(results))
