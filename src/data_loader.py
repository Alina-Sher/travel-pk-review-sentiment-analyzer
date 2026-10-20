"""Loading, cleaning, de-duplication, filtering and export. Synthetic data is always labelled and kept separate."""
from __future__ import annotations

import io
import re

import pandas as pd

from .sentiment import ANALYSIS_COLUMNS, analyze_dataframe
from .validation import CANONICAL, ValidationReport, empty_clean_frame, validate_and_clean

PROV_UPLOAD, PROV_LIVE, PROV_SYNTHETIC = "uploaded_csv", "live_collected", "synthetic_demo"
PROVENANCE_LABELS = {
    PROV_UPLOAD: "Uploaded CSV (user-provided)",
    PROV_LIVE: "Live-collected (public page, robots.txt-checked)",
    PROV_SYNTHETIC: "SYNTHETIC demonstration data (not real)",
}
OUTPUT_COLUMNS = CANONICAL + ["provenance"] + ANALYSIS_COLUMNS + ["near_duplicate"]


def empty_frame() -> pd.DataFrame:
    df = empty_clean_frame()
    df["provenance"] = pd.Series(dtype="object")
    df = analyze_dataframe(df)
    df["near_duplicate"] = pd.Series(dtype="bool")
    return df[OUTPUT_COLUMNS]


def _norm_key(company: str, text: str) -> str:
    t = re.sub(r"[^\w\s]", "", str(text).lower())
    return str(company).strip().lower() + "||" + re.sub(r"\s+", " ", t).strip()


def dedupe_exact(df: pd.DataFrame):
    """Remove exact duplicates (same company + same normalised text). Keeps the first occurrence."""
    if df.empty:
        return df, 0
    keys = [_norm_key(c, t) for c, t in zip(df["company"], df["review_text"])]
    dup = pd.Series(keys, index=df.index).duplicated(keep="first")
    return df[~dup].reset_index(drop=True), int(dup.sum())


def flag_near_duplicates(df: pd.DataFrame, threshold: float = 0.9, max_per_company: int = 2000):
    """Flag (never remove) reviews whose word sets overlap >= threshold (Jaccard) with an earlier review of the same company."""
    df = df.reset_index(drop=True).copy()
    df["near_duplicate"] = False
    skipped = []
    for company, idx in df.groupby("company").groups.items():
        idx = list(idx)
        if len(idx) > max_per_company:
            skipped.append(company)
            continue
        toks = [set(re.findall(r"\w+", df.at[i, "review_text"].lower())) for i in idx]
        for a in range(1, len(idx)):
            if len(toks[a]) < 5:
                continue
            for b in range(a):
                if len(toks[b]) < 5:
                    continue
                if len(toks[a] & toks[b]) / len(toks[a] | toks[b]) >= threshold:
                    df.at[idx[a], "near_duplicate"] = True
                    break
    return df, skipped


def prepare_reviews(raw: pd.DataFrame, provenance: str = PROV_UPLOAD, report: ValidationReport | None = None):
    """Validate, de-duplicate, tag provenance and run sentiment. Returns (analyzed_df, report)."""
    df, report = validate_and_clean(raw, report)
    if not report.ok or df.empty:
        report.stats["rows_out"] = 0
        return empty_frame(), report
    df, removed = dedupe_exact(df)
    report.stats["exact_duplicates_removed"] = removed
    if removed:
        report.warnings.append(f"Removed {removed} exact duplicate review(s).")
    df, skipped = flag_near_duplicates(df)
    near = int(df["near_duplicate"].sum())
    report.stats["near_duplicates_flagged"] = near
    if near:
        report.warnings.append(f"Flagged {near} near-duplicate review(s) (kept, not removed; see 'near_duplicate').")
    if skipped:
        report.warnings.append("Near-duplicate check skipped for very large company set(s): " + ", ".join(map(str, skipped)))
    df["provenance"] = provenance
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df["collected_at"] = df["collected_at"].fillna(now)
    df = analyze_dataframe(df)
    report.stats["rows_out"] = int(len(df))
    return df[OUTPUT_COLUMNS], report


def _read_any(obj) -> pd.DataFrame:
    if isinstance(obj, (bytes, bytearray)):
        obj = io.BytesIO(obj)
    if isinstance(obj, str) and "\n" in obj:
        return pd.read_csv(io.StringIO(obj), dtype=str, keep_default_na=False)
    if hasattr(obj, "seek"):
        obj.seek(0)
    try:
        return pd.read_csv(obj, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(obj, "seek"):
            obj.seek(0)
        return pd.read_csv(obj, dtype=str, keep_default_na=False, encoding="latin-1")


def load_csv(obj, provenance: str = PROV_UPLOAD):
    """Load a CSV (path, bytes, or file-like). Problems are returned in the report, never raised."""
    report = ValidationReport()
    try:
        raw = _read_any(obj)
    except pd.errors.EmptyDataError:
        report.errors.append("The CSV file is empty.")
        return empty_frame(), report
    except Exception as exc:  # malformed CSV
        report.errors.append(f"The file could not be read as CSV: {exc}")
        return empty_frame(), report
    return prepare_reviews(raw, provenance, report)


def combine_frames(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return empty_frame()
    merged = pd.concat(frames, ignore_index=True)
    merged, _ = dedupe_exact(merged)
    return merged


def filter_reviews(df, companies=None, sentiments=None, sources=None, rating_range=None, include_unrated=True,
                   date_range=None, include_undated=True, query=""):
    """Single filtering path used by every dashboard section so metrics stay consistent."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if companies is not None:
        mask &= df["company"].isin(companies)
    if sentiments is not None:
        mask &= df["sentiment"].isin(sentiments)
    if sources is not None:
        mask &= df["source"].isin(sources)
    if rating_range is not None:
        lo, hi = rating_range
        mask &= df["rating"].between(lo, hi) | (df["rating"].isna() & include_unrated)
    if date_range is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        in_range = (df["date"].dt.normalize() >= start) & (df["date"].dt.normalize() <= end)
        mask &= in_range | (df["date"].isna() & include_undated)
    if query and query.strip():
        q = re.escape(query.strip())
        mask &= df["review_text"].str.contains(q, case=False, regex=True) | df["company"].str.contains(q, case=False, regex=True)
    return df[mask].reset_index(drop=True)


def template_csv() -> str:
    return "company,review_text,rating,date,source,url,collected_at\n"


def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------- synthetic demo (NOT real data)
_POS_OPEN = ["Booked my flight in a few minutes and the process was easy.", "The agent was helpful and responded quickly.",
             "Great prices compared with other sites.", "Smooth hotel booking, the room matched the description.",
             "Our holiday package was well organized.", "Very convenient app and the payment went through without any problem."]
_POS_TAIL = ["Would recommend.", "Will book again.", "Thanks for the quick help!", "Happy with the experience.",
             "Good value for money.", "Totally satisfied."]
_NEG_OPEN = ["Refund request is still pending after weeks and nobody replies.", "Customer support was unresponsive and rude.",
             "Hidden charges were added at checkout, very misleading.", "The hotel room was not as described and the booking was a mess.",
             "Payment was deducted twice and the ticket was not issued.", "Cancelled my flight and the cancellation process was a nightmare.",
             "The app keeps crashing and shows errors during payment."]
_NEG_TAIL = ["Very disappointed.", "Would not recommend.", "Worst experience.", "Waste of money.", "Still waiting for a response.", "Avoid."]
_NEU_OPEN = ["Booked a flight for next month.", "Used the website to check hotel options.", "The booking was confirmed by email.",
             "Ordered a holiday package for two people.", "I changed my travel dates once."]
_NEU_TAIL = ["Details were as listed.", "Checked in on the day.", "Standard process.", "Received the confirmation.", "Nothing special."]
_DEMO_PROFILES = [("Demo Agency A (synthetic)", 14, 6, 10, 0), ("Demo Agency B (synthetic)", 16, 5, 7, 2), ("Demo Agency C (synthetic)", 8, 6, 14, 4)]


def synthetic_demo() -> pd.DataFrame:
    """Deterministic, clearly fictional records for demonstrating the UI. Company names are fictional on purpose."""
    rows = []
    for name, n_pos, n_neu, n_neg, off in _DEMO_PROFILES:
        for kind, n, opens, tails, ratings in (("pos", n_pos, _POS_OPEN, _POS_TAIL, [5, 4, 5, 4]),
                                               ("neu", n_neu, _NEU_OPEN, _NEU_TAIL, [3, 3, 4]),
                                               ("neg", n_neg, _NEG_OPEN, _NEG_TAIL, [1, 2, 1])):
            for i in range(n):
                text = f"{opens[(i + off) % len(opens)]} {tails[(i // len(opens) + off) % len(tails)]}"
                day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=(i * 11 + off * 5) % 330)
                rows.append({"company": name, "review_text": text, "rating": ratings[i % len(ratings)],
                             "date": day.strftime("%Y-%m-%d"), "source": "Synthetic demo (generated)", "url": "", "collected_at": ""})
    df, _ = prepare_reviews(pd.DataFrame(rows), PROV_SYNTHETIC)
    return df
