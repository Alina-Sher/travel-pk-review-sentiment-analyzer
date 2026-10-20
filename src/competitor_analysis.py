"""Company-level comparison with explicit denominators and sample-size caveats."""
from __future__ import annotations

import pandas as pd

from .sentiment import NEGATIVE, NEUTRAL, POSITIVE, UNCLASSIFIED

MIN_RELIABLE_N = 30        # per-company classified reviews below this are flagged as a small sample
MIN_ANALYSIS_N = 30        # overall classified reviews below this trigger the dataset-level notice


def safe_pct(n, d):
    """Percentage rounded to 1 dp, or None when the denominator is zero."""
    return round(100.0 * n / d, 1) if d else None


def sentiment_counts(df: pd.DataFrame) -> dict:
    s = df["sentiment"] if not df.empty else pd.Series([], dtype="object")
    pos, neu, neg, unc = (int((s == x).sum()) for x in (POSITIVE, NEUTRAL, NEGATIVE, UNCLASSIFIED))
    classified = pos + neu + neg
    return {"total": int(len(df)), "classified": classified, "unclassified": unc,
            "positive": pos, "neutral": neu, "negative": neg,
            "positive_pct": safe_pct(pos, classified), "neutral_pct": safe_pct(neu, classified),
            "negative_pct": safe_pct(neg, classified)}


def company_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["company", "reviews", "classified", "positive", "neutral", "negative", "pos_pct", "neu_pct", "neg_pct",
            "avg_rating", "rated_reviews", "first_date", "last_date", "sources", "small_sample"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for company, g in df.groupby("company", sort=True):
        c = sentiment_counts(g)
        rated = g["rating"].dropna()
        rows.append({
            "company": company, "reviews": c["total"], "classified": c["classified"], "positive": c["positive"],
            "neutral": c["neutral"], "negative": c["negative"], "pos_pct": c["positive_pct"], "neu_pct": c["neutral_pct"],
            "neg_pct": c["negative_pct"], "avg_rating": round(float(rated.mean()), 2) if len(rated) else None,
            "rated_reviews": int(len(rated)), "first_date": g["date"].min(), "last_date": g["date"].max(),
            "sources": ", ".join(sorted(g["source"].unique())), "small_sample": c["classified"] < MIN_RELIABLE_N,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("reviews", ascending=False).reset_index(drop=True)


def sample_size_notes(summary: pd.DataFrame) -> list:
    notes = []
    if summary.empty:
        return notes
    small = summary[summary["small_sample"]]
    if len(small):
        notes.append("Small samples (fewer than %d classified reviews): %s. Percentages and ratings for these companies are unstable."
                     % (MIN_RELIABLE_N, "; ".join(f"{r.company} (n={r.classified})" for r in small.itertuples())))
    if len(summary) > 1 and summary["reviews"].min() > 0 and summary["reviews"].max() / summary["reviews"].min() >= 3:
        notes.append("Review volumes differ at least 3x between companies, so comparisons are not like-for-like.")
    if len(summary) > 1 and summary["sources"].nunique() > 1:
        notes.append("Companies are covered by different sources; platform differences can affect ratings and tone.")
    if len(summary) > 1 and summary["rated_reviews"].gt(0).any() and summary["rated_reviews"].eq(0).any():
        notes.append("Some companies have no star ratings, so average ratings are not comparable across all companies.")
    return notes


def dataset_warnings(df: pd.DataFrame) -> list:
    if df.empty:
        return ["No reviews in the current selection."]
    w = []
    if (df["provenance"] == "synthetic_demo").any():
        w.append("SYNTHETIC DEMONSTRATION DATA: these are generated examples, not real customer reviews. Do not cite any result as a finding.")
    c = sentiment_counts(df)
    if c["classified"] < MIN_ANALYSIS_N:
        w.append(f"Only {c['classified']} classified review(s): too few to support meaningful conclusions (guideline: at least {MIN_ANALYSIS_N}).")
    if c["unclassified"]:
        w.append(f"{c['unclassified']} review(s) are in Urdu script or otherwise unclassifiable and are excluded from sentiment percentages.")
    if df["rating"].isna().any():
        w.append(f"{int(df['rating'].isna().sum())} of {len(df)} review(s) have no star rating; averages use rated reviews only.")
    if df["date"].isna().any():
        w.append(f"{int(df['date'].isna().sum())} of {len(df)} review(s) have no date.")
    if df["source"].nunique() == 1:
        w.append("All reviews come from a single source, which may not represent all customers.")
    if int(df["near_duplicate"].sum()):
        w.append(f"{int(df['near_duplicate'].sum())} near-duplicate review(s) are present (flagged, not removed).")
    w.append("Sentiment is lexicon-based (English, partial Roman Urdu); expect some misclassification, especially sarcasm.")
    return w
