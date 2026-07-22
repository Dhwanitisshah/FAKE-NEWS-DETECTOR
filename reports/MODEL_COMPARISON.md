# Model Comparison

Three models — a TF-IDF + Logistic Regression baseline, a BiLSTM, and DistilBERT — evaluated on the same 7,820-row held-out test set.

## Results

| Model | Training data | Accuracy | FAKE recall | False negatives | False positives |
|---|---|---|---|---|---|
| DistilBERT | 8,000 rows (subset) | 0.9987 | 0.9983 | 6 | 4 |
| BiLSTM | 31,278 rows (full) | 0.9900 | 0.9872 | 46 | 32 |
| Baseline (TF-IDF + LogReg) | 31,278 rows (full) | 0.9861 | 0.9785 | 77 | 32 |

*Sorted by FAKE recall, best first.*

## What the numbers show

False negatives fall steadily across the three models — 77 → 46 → 6 — suggesting each step up in language understanding catches fakes the previous model missed. Notably, DistilBERT reaches the best result of the three despite training on only 8,000 rows, roughly a quarter of the full 31,278-row training set available to the baseline and BiLSTM.

## Honest caveat

This dataset is stylistically easy: real and fake articles come from systematically different sources, so style alone is a strong signal. These numbers are likely an upper bound on real-world performance, and none of the three models has been stress-tested on a "hard fake" — an article deliberately written to sound like legitimate reporting. That's exactly what the LIME analysis probes next.

## Decision

The deployed Streamlit app serves the TF-IDF + Logistic Regression baseline: it's a few hundred KB with no heavy dependencies, so it loads instantly and runs reliably on free hosting. DistilBERT is documented here as the most accurate model, but it is not shipped live — its ~1GB torch/transformers footprint risks slow cold starts and out-of-memory crashes on constrained free tiers, which would break a live portfolio demo. The ~1% accuracy gain over the baseline isn't worth that reliability risk for a demo. A local-only model picker (letting a user switch between baseline, BiLSTM, and DistilBERT when running the app locally) is a possible future stretch feature.
