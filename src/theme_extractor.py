"""Keyword-based theme tagging. Themes are CANDIDATES: reported only when review text actually matches."""
from __future__ import annotations

import re

import pandas as pd

from .sentiment import NEGATIVE, POSITIVE

PRAISE_THEMES = {
    "Customer service & support": r"\b(?:customer (?:service|support|care)|support (?:team|staff)|agents?|staff|representatives?|helpful|cooperative|responsive|responded|assist\w*|helped|helpline)\b",
    "Booking ease & convenience": r"\b(?:easy|easily|simple|convenient|convenience|smooth|smoothly|seamless|hassle[- ]?free|user[- ]friendly|booking process)\b",
    "Pricing & value": r"\b(?:price[sd]?|pricing|cheap\w*|affordable|discounts?|deals?|value for money|budget|offers?|sasta|sasti|paisa wasool|pocket[- ]friendly)\b",
    "Speed & timeliness": r"\b(?:quick\w*|fast\w*|prompt\w*|on time|instant\w*|immediately|within minutes|timely)\b",
    "Flight booking": r"\b(?:flights?|tickets?|airlines?|airfare|e-?tickets?|boarding)\b",
    "Hotel booking": r"\b(?:hotels?|rooms?|accommodation|resorts?|stay)\b",
    "Holiday & tour packages": r"\b(?:tours?|packages?|holidays?|trips?|umrah|honeymoon|itinerary|hunza|skardu|murree|northern areas)\b",
    "Refunds & cancellations": r"\b(?:refund\w*|cancel\w*)\b",
    "App & website": r"\b(?:app|application|website|site|interface|portal)\b",
}
COMPLAINT_THEMES = {
    "Refunds": r"\b(?:refund\w*|money back|reimburs\w*)\b",
    "Cancellations": r"\b(?:cancel\w*)\b",
    "Payment problems": r"\b(?:payments?|paid|charged|deducted|transactions?|double charge\w*|debit|credit card|jazzcash|easypaisa)\b",
    "Customer support": r"\b(?:customer (?:service|support|care)|support|agents?|representatives?|helpline|call cent(?:er|re)|unresponsive|no response|not respond\w*|ignored|ignoring|no reply|never repl\w*|whatsapp)\b",
    "Booking changes & rescheduling": r"\b(?:re-?schedul\w*|modif\w*|amend\w*|date change|name change|change[ds]? (?:my |the )?(?:date|dates|booking|name|flight)|changing)\b",
    "Hidden charges & fees": r"\b(?:hidden|extra (?:charges?|fees?|money)|additional (?:charges?|fees?)|service fees?|commission|overcharg\w*|fees?)\b",
    "Hotel problems": r"\b(?:hotels?|rooms?|accommodation|resorts?|check[- ]?in|reservation)\b",
    "Flight & ticketing problems": r"\b(?:flights?|tickets?|airlines?|pnr|boarding|e-?tickets?)\b",
    "App & website issues": r"\b(?:app|website|site|error|errors|glitch\w*|crash\w*|login|bug|otp)\b",
    "Slow response & delays": r"\b(?:slow\w*|waiting|waited|delay\w*|late|no reply|took (?:days|weeks|hours)|still waiting)\b",
}
_COMPILED = {
    "positive": {k: re.compile(v, re.I) for k, v in PRAISE_THEMES.items()},
    "negative": {k: re.compile(v, re.I) for k, v in COMPLAINT_THEMES.items()},
}
_LABEL = {"positive": POSITIVE, "negative": NEGATIVE}
SUMMARY_COLUMNS = ["theme", "count", "denominator", "pct"]
BY_COMPANY_COLUMNS = ["company", "theme", "count", "denominator", "pct"]


def _check(polarity: str):
    if polarity not in _COMPILED:
        raise ValueError("polarity must be 'positive' or 'negative'")


def _pct(count: int, denom: int):
    return round(100.0 * count / denom, 1) if denom else float("nan")


def tag_themes(text: str, polarity: str) -> list:
    _check(polarity)
    return [t for t, rx in _COMPILED[polarity].items() if rx.search(text or "")]


def eligible_reviews(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    _check(polarity)
    if df.empty:
        return df
    return df[df["sentiment"] == _LABEL[polarity]]


def theme_summary(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """Themes with >=1 supporting review among reviews of the matching sentiment. Denominator = those reviews."""
    sub = eligible_reviews(df, polarity)
    denom = int(len(sub))
    rows = []
    for theme, rx in _COMPILED[polarity].items():
        count = int(sum(1 for t in sub["review_text"] if rx.search(t))) if denom else 0
        if count:
            rows.append({"theme": theme, "count": count, "denominator": denom, "pct": _pct(count, denom)})
    out = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return out.sort_values(["count", "theme"], ascending=[False, True]).reset_index(drop=True)


def theme_by_company(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    frames = []
    for company in sorted(df["company"].unique()) if not df.empty else []:
        s = theme_summary(df[df["company"] == company], polarity)
        if not s.empty:
            s.insert(0, "company", company)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BY_COMPANY_COLUMNS)


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_LONGNUM = re.compile(r"\d{5,}")
_HANDLE = re.compile(r"(?<!\w)@\w+")


def anonymize(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    text = _PHONE.sub("[number]", text)
    text = _LONGNUM.sub("[number]", text)
    return _HANDLE.sub("[handle]", text)


def theme_excerpts(df: pd.DataFrame, polarity: str, theme: str, limit: int = 3, width: int = 110) -> list:
    """Short, anonymised, real excerpts around the matched keyword, with company/source attribution."""
    _check(polarity)
    rx = _COMPILED[polarity].get(theme)
    sub = eligible_reviews(df, polarity)
    if rx is None or sub.empty:
        return []
    sub = sub.assign(_mag=sub["sentiment_score"].abs()).sort_values("_mag", ascending=False)
    out = []
    for _, row in sub.iterrows():
        m = rx.search(row["review_text"])
        if not m:
            continue
        text = row["review_text"]
        a, b = max(0, m.start() - width), min(len(text), m.end() + width)
        snippet = ("..." if a > 0 else "") + text[a:b].strip() + ("..." if b < len(text) else "")
        out.append({"excerpt": anonymize(snippet), "company": row["company"], "source": row["source"]})
        if len(out) >= limit:
            break
    return out
