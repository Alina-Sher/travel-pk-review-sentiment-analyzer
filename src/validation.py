"""CSV validation and cleaning. Never invents values: unusable values become blank and are reported."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

REQUIRED = ["company", "review_text"]
OPTIONAL = ["rating", "date", "source", "url", "collected_at"]
CANONICAL = REQUIRED + OPTIONAL

COLUMN_ALIASES = {
    "company": ["company", "business", "brand", "agency", "company_name"],
    "review_text": ["review_text", "review", "text", "comment", "content", "body", "review_body"],
    "rating": ["rating", "stars", "star_rating", "score"],
    "date": ["date", "review_date", "created_at", "published", "published_date"],
    "source": ["source", "platform"],
    "url": ["url", "review_url", "link"],
    "collected_at": ["collected_at", "collection_timestamp", "scraped_at"],
}
DEFAULT_SOURCE = "Uploaded CSV (source not stated)"


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _key(name) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def normalize_columns(df: pd.DataFrame, report: ValidationReport):
    lookup = {a: canon for canon, aliases in COLUMN_ALIASES.items() for a in aliases}
    rename, seen = {}, set()
    for col in df.columns:
        canon = lookup.get(_key(col))
        if canon is None:
            continue
        if canon in seen:
            report.warnings.append(f"Column '{col}' duplicates '{canon}' and was ignored.")
            continue
        seen.add(canon)
        rename[col] = canon
    out = df[list(rename)].rename(columns=rename)
    ignored = [c for c in df.columns if _key(c) not in lookup]
    if ignored:
        report.warnings.append("Ignored unrecognized columns: " + ", ".join(map(str, ignored)))
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        report.errors.append(
            "Missing required column(s): " + ", ".join(missing)
            + ". Required: company, review_text. Optional: " + ", ".join(OPTIONAL) + "."
        )
    return out, report


def _blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def empty_clean_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "company": pd.Series(dtype="object"), "review_text": pd.Series(dtype="object"),
        "rating": pd.Series(dtype="float64"), "date": pd.Series(dtype="datetime64[ns]"),
        "source": pd.Series(dtype="object"), "url": pd.Series(dtype="object"),
        "collected_at": pd.Series(dtype="datetime64[ns]"),
    })


def _to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_convert(None).astype("datetime64[ns]")


def validate_and_clean(raw: pd.DataFrame, report: ValidationReport | None = None):
    report = report or ValidationReport()
    if raw is None or raw.shape[1] == 0:
        report.errors.append("The file has no columns.")
        return empty_clean_frame(), report
    df, report = normalize_columns(raw, report)
    report.stats["rows_in"] = int(len(raw))
    if not report.ok:
        return empty_clean_frame(), report
    df = df.copy()
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = ""
    df["company"] = df["company"].fillna("").astype(str).str.strip()
    df["review_text"] = df["review_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    bad_text = _blank(df["review_text"])
    bad_company = _blank(df["company"]) & ~bad_text
    report.stats["dropped_missing_text"] = int(bad_text.sum())
    report.stats["dropped_missing_company"] = int(bad_company.sum())
    if bad_text.sum():
        report.warnings.append(f"Dropped {int(bad_text.sum())} row(s) with empty review text.")
    if bad_company.sum():
        report.warnings.append(f"Dropped {int(bad_company.sum())} row(s) with no company name.")
    df = df[~bad_text & ~_blank(df["company"])].reset_index(drop=True)

    raw_rating = df["rating"]
    num = pd.to_numeric(raw_rating, errors="coerce")
    invalid = ~_blank(raw_rating) & (num.isna() | (num < 1) | (num > 5))
    report.stats["invalid_ratings"] = int(invalid.sum())
    if invalid.sum():
        report.warnings.append(f"{int(invalid.sum())} rating(s) were not numbers between 1 and 5 and were left blank.")
    df["rating"] = num.where(~invalid).astype("float64")

    raw_date = df["date"]
    parsed = _to_datetime(raw_date)
    bad_date = ~_blank(raw_date) & parsed.isna()
    report.stats["invalid_dates"] = int(bad_date.sum())
    if bad_date.sum():
        report.warnings.append(f"{int(bad_date.sum())} date(s) could not be parsed and were left blank (ISO format YYYY-MM-DD is recommended).")
    df["date"] = parsed

    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df.loc[df["source"] == "", "source"] = DEFAULT_SOURCE
    url = df["url"].fillna("").astype(str).str.strip()
    bad_url = (url != "") & ~url.str.match(r"^https?://", case=False)
    if bad_url.sum():
        report.warnings.append(f"{int(bad_url.sum())} URL(s) did not start with http(s) and were left blank.")
    df["url"] = url.where(~bad_url, "")
    df["collected_at"] = _to_datetime(df["collected_at"])
    if df.empty:
        report.warnings.append("No valid review rows remain after validation.")
    return df[CANONICAL], report
