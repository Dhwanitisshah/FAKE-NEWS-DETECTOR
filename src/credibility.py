"""Rule-based source credibility scorer.

Not machine learning: a small set of explainable heuristics that produce a
human-legible credibility signal alongside the ML model. Works on text alone;
a URL is optional and only adds a domain signal when present.
"""

import re
from urllib.parse import urlparse

# A handful of loud/clickbait phrases. Small and hardcoded on purpose --
# this is a legibility signal, not an exhaustive sensationalism detector.
# Unambiguous: always sensational regardless of case.
CLICKBAIT_PHRASES = [
    "you won't believe",
    "shocking truth",
    "miracle",
    "exposed",
]

# Ambiguous: neutral in normal news writing ("officials confirmed..."),
# only sensational when SHOUTED. Counted only on exact all-caps match.
SHOUTED_SENSATIONAL_WORDS = [
    "confirmed",
    "breaking",
    "secret",
    "shocking",
]

# Simplified heuristic domain lists -- NOT an authoritative reputation
# database. Just enough to demonstrate the domain signal; extend with care.
REPUTABLE_DOMAINS = {"reuters.com", "apnews.com", "bbc.com"}
JUNK_DOMAIN_PATTERNS = [".xyz", "clickhole", "newsflash24"]


def score_sensationalism(text: str) -> tuple[float, list[str]]:
    """Score raw text 0-100 (higher = more credible / less sensational)."""
    reasons: list[str] = []
    score = 100.0

    words = re.findall(r"[A-Za-z']+", text)
    caps_words = [w for w in words if len(w) > 2 and w.isupper()]
    if words:
        caps_ratio = len(caps_words) / len(words)
        if caps_words:
            score -= min(caps_ratio * 200, 40)
            reasons.append(f"{len(caps_words)} all-caps words")

    text_lower = text.lower()
    clickbait_hits = sum(text_lower.count(phrase) for phrase in CLICKBAIT_PHRASES)
    if clickbait_hits:
        score -= min(clickbait_hits * 10, 20)
        reasons.append(f"{clickbait_hits} clickbait term(s)")

    shouted_hits = sum(
        len(re.findall(rf"\b{re.escape(word.upper())}\b", text))
        for word in SHOUTED_SENSATIONAL_WORDS
    )
    if shouted_hits:
        score -= min(shouted_hits * 10, 20)
        reasons.append(f"{shouted_hits} shouted sensational term(s)")

    punct_runs = re.findall(r"(!{2,}|\?{2,})", text)
    if punct_runs:
        score -= min(len(punct_runs) * 8, 20)
        reasons.append(f"{len(punct_runs)} run(s) of excessive punctuation")

    return max(0.0, min(100.0, score)), reasons


def score_domain(url: str | None) -> tuple[float | None, str]:
    """Score an optional source URL 0-100 using a tiny heuristic domain list."""
    if not url:
        return None, "No source URL provided"

    domain = urlparse(url).netloc.lower().removeprefix("www.")

    if domain in REPUTABLE_DOMAINS:
        return 90.0, f"Domain '{domain}' is a known reputable source"
    if any(pattern in domain for pattern in JUNK_DOMAIN_PATTERNS):
        return 20.0, f"Domain '{domain}' matches a known low-quality pattern"
    return 60.0, f"Domain '{domain}' is unrecognized (neutral score)"


def credibility_score(text: str, url: str | None = None) -> dict:
    """Combine sub-scores into a final 0-100 credibility score with reasons."""
    sensationalism_score, sensationalism_reasons = score_sensationalism(text)
    domain_score, domain_reason = score_domain(url)

    if domain_score is None:
        final_score = sensationalism_score
    else:
        final_score = 0.7 * sensationalism_score + 0.3 * domain_score

    final_score = max(0, min(100, round(final_score)))

    return {
        "final_score": final_score,
        "sensationalism_score": round(sensationalism_score, 1),
        "domain_score": None if domain_score is None else round(domain_score, 1),
        "reasons": sensationalism_reasons + [domain_reason],
    }


if __name__ == "__main__":
    examples = [
        (
            "The central bank raised interest rates by a quarter point on Wednesday, "
            "citing persistent inflation pressures. Officials said further increases "
            "remain possible depending on upcoming economic data.",
            None,
        ),
        (
            "SHOCKING report EXPOSED!!! You won't believe what this secret memo "
            "confirmed about the miracle cure doctors don't want you to know???",
            None,
        ),
        (
            "Officials confirmed the trade agreement was signed after months of "
            "negotiation between the two countries.",
            "https://www.reuters.com/world/trade-deal-signed",
        ),
    ]

    for i, (text, url) in enumerate(examples, start=1):
        result = credibility_score(text, url)
        print(f"Example {i} (url={url!r})")
        print(f"  final_score: {result['final_score']}")
        print(f"  sensationalism_score: {result['sensationalism_score']}")
        print(f"  domain_score: {result['domain_score']}")
        print(f"  reasons: {result['reasons']}")
        print()
