"""Lightweight, dependency-free lexicon sentiment for English and (partially) Roman Urdu.

Method (documented in README): word/phrase lexicon with weights; look-back negation (3 tokens, stops at
punctuation); post-positive negation for Roman Urdu ("acha nahi"); intensifiers/dampeners; "but" clause
re-weighting; "!" and ALL-CAPS emphasis; VADER-style squashing to a -1..1 score.
Star ratings are NOT used here. Urdu-script text is labelled "Unclassified" rather than guessed.
"""
from __future__ import annotations

import math
import re

import pandas as pd

POSITIVE, NEUTRAL, NEGATIVE, UNCLASSIFIED = "Positive", "Neutral", "Negative", "Unclassified"
POS_THRESHOLD, NEG_THRESHOLD = 0.2, -0.2

TOKEN_RE = re.compile(r"[A-Za-z0-9_'\-]+|[.,;:!?]")
URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
BOUNDARY = {".", ",", ";", ":", "!", "?"}

LEXICON = {
    # English positive
    "good": 2, "great": 2.5, "excellent": 3, "amazing": 3, "awesome": 3, "fantastic": 3, "wonderful": 3,
    "perfect": 3, "best": 2.5, "nice": 1.8, "smooth": 2, "smoothly": 2, "easy": 2, "easily": 2, "quick": 1.8,
    "quickly": 1.8, "fast": 1.8, "prompt": 2, "promptly": 2, "helpful": 2.5, "friendly": 2, "polite": 2,
    "professional": 2, "responsive": 2, "reliable": 2.5, "trustworthy": 2.5, "recommend": 2, "recommended": 2,
    "satisfied": 2.5, "happy": 2.5, "pleased": 2.5, "convenient": 2.2, "affordable": 2, "cheap": 1.2,
    "comfortable": 2, "clean": 1.5, "seamless": 2.5, "hassle-free": 2.5, "efficient": 2, "cooperative": 2,
    "supportive": 2, "thank": 1.5, "thanks": 1.5, "love": 2.5, "loved": 2.5, "superb": 3, "outstanding": 3,
    "brilliant": 3, "impressed": 2.5, "honest": 2, "safe": 1.5, "resolved": 2, "helped": 2, "pleasant": 2,
    "genuine": 1.5, "organized": 1.8, "organised": 1.8,
    # English negative
    "bad": -2, "worst": -3.2, "terrible": -3, "horrible": -3, "awful": -3, "poor": -2.2, "pathetic": -3,
    "disappointing": -2.5, "disappointed": -2.5, "disappointment": -2.5, "useless": -2.8, "scam": -3.5,
    "fraud": -3.5, "fake": -2.5, "cheat": -3, "cheated": -3.2, "rude": -2.5, "unprofessional": -2.8,
    "slow": -1.8, "delay": -1.8, "delayed": -2, "delays": -1.8, "cancelled": -1.2, "canceled": -1.2,
    "waste": -2.5, "rip-off": -3, "ripoff": -3, "overcharged": -2.8, "overcharge": -2.5, "hidden": -1.5,
    "unresponsive": -2.8, "ignored": -2.2, "ignoring": -2.2, "refused": -2.2, "refuse": -2, "stuck": -1.8,
    "failed": -2, "failure": -2, "problem": -1.5, "problems": -1.7, "issue": -1.2, "issues": -1.4,
    "complaint": -1.5, "frustrating": -2.5, "frustrated": -2.5, "annoying": -2, "unhappy": -2.5, "angry": -2.2,
    "lied": -3, "lie": -2.5, "misleading": -2.5, "mislead": -2.5, "stressful": -2, "nightmare": -3.2,
    "harassment": -3, "avoid": -2.2, "unreliable": -2.5, "incompetent": -3, "lousy": -2.5, "expensive": -1.5,
    "costly": -1.5, "confusing": -1.8, "error": -1.5, "errors": -1.7, "glitch": -1.5, "glitches": -1.7,
    "crash": -1.8, "crashes": -1.8, "crashing": -1.8, "wait": -0.6, "waiting": -0.6, "waited": -1.2,
    "unanswered": -2.2, "deducted": -1.6, "missing": -1.5, "lost": -1.8, "wrong": -1.8, "mess": -2.2,
    "chaos": -2.5, "dishonest": -3, "rude.": -2.5,
    # Roman Urdu (partial coverage; spelling varies a lot)
    "acha": 2, "achi": 2, "achha": 2, "achhi": 2, "zabardast": 3, "behtareen": 3, "shandar": 3, "kamal": 2.5,
    "shukriya": 1.5, "meherbani": 1.5, "sasta": 1.5, "sasti": 1.5, "behtar": 2, "umda": 2.5, "pasand": 1.5,
    "bura": -2, "buri": -2, "bekar": -2.5, "bakwas": -2.8, "ghatiya": -3, "dhoka": -3.5, "dhokha": -3.5,
    "loot": -2.8, "kharab": -2.5, "mehnga": -1.5, "mehngi": -1.5, "takleef": -2, "pareshan": -2,
    "pareshani": -2, "jhoot": -2.5, "jhota": -2.5, "ghalat": -2,
}

PHRASES = {
    "waste of money": -3.2, "waste of time": -3, "highly recommend": 3.2, "highly recommended": 3.2,
    "value for money": 2.5, "no response": -2.5, "not responding": -2.5, "never again": -3, "rip off": -3,
    "no problem": 1.5, "no issues": 1.5, "not refunded": -2.5, "no refund": -2.5, "still waiting": -2.2,
    "worst service": -3.5, "paisa wasool": 2.5, "worth it": 2, "thumbs up": 2, "pocket friendly": 2,
    "on time": 1.8, "well organized": 2.2, "well organised": 2.2, "no reply": -2.5, "never replied": -2.5,
    "would not recommend": -3, "wouldn't recommend": -3, "do not recommend": -3, "don't recommend": -3,
}

NEGATORS = {
    "not", "no", "never", "none", "nothing", "without", "hardly", "cannot", "dont", "don't", "didnt", "didn't",
    "doesnt", "doesn't", "isnt", "isn't", "wasnt", "wasn't", "arent", "aren't", "werent", "weren't", "cant",
    "can't", "wont", "won't", "couldnt", "couldn't", "wouldnt", "wouldn't", "havent", "haven't", "hasnt",
    "hasn't", "hadnt", "hadn't", "nahi", "nahin", "nhi", "nai", "mat",
}
POST_NEGATORS = {"nahi", "nahin", "nhi", "nai"}
BOOSTERS = {
    "very": 0.3, "really": 0.3, "extremely": 0.5, "super": 0.4, "so": 0.2, "highly": 0.35, "absolutely": 0.5,
    "totally": 0.3, "too": 0.2, "bahut": 0.3, "bohat": 0.3, "boht": 0.3, "bohot": 0.3, "buhat": 0.3, "bht": 0.25,
    "slightly": -0.3, "somewhat": -0.25, "kinda": -0.25, "fairly": -0.15, "bit": -0.2,
}
CONTRAST = {"but", "however", "lekin", "magar", "though"}
ROMAN_URDU_MARKERS = {
    "hai", "hain", "nahi", "nahin", "nhi", "bahut", "bohat", "boht", "acha", "achi", "bura", "buri", "kya", "ka",
    "ki", "ko", "se", "mein", "main", "aur", "tha", "thi", "ho", "kar", "karo", "ye", "yeh", "wo", "woh", "mera",
    "meri", "humara", "paisa", "paisay", "lekin", "magar", "bilkul", "sirf", "bhi", "abhi", "kuch", "koi",
}
NEGATION_SCALE = -0.74


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "Unknown"
    urdu = sum(1 for c in letters if URDU_RE.match(c))
    if urdu / len(letters) > 0.3:
        return "Urdu script"
    words = re.findall(r"[a-z']+", text.lower())
    hits = sum(1 for w in words if w in ROMAN_URDU_MARKERS)
    if words and hits >= 2 and hits / len(words) >= 0.15:
        return "Roman Urdu / mixed"
    return "English"


def _is_negated(low: list, start: int) -> bool:
    j, steps = start - 1, 0
    while j >= 0 and steps < 3 and low[j] not in BOUNDARY:
        if low[j] in NEGATORS:
            return True
        j -= 1
        steps += 1
    return False


def _booster(low: list, start: int) -> float:
    j, steps = start - 1, 0
    while j >= 0 and steps < 2 and low[j] not in BOUNDARY:
        if low[j] in BOOSTERS:
            return BOOSTERS[low[j]]
        j -= 1
        steps += 1
    return 0.0


def analyze_text(text) -> dict:
    """Return {label, score, language, matches}. Score in [-1, 1]; ratings are never consulted."""
    if not isinstance(text, str) or not text.strip():
        return {"label": UNCLASSIFIED, "score": 0.0, "language": "Unknown", "matches": 0}
    language = detect_language(text)
    if language == "Urdu script":
        return {"label": UNCLASSIFIED, "score": 0.0, "language": language, "matches": 0}

    raw = TOKEN_RE.findall(text.replace("\u2019", "'"))
    low = [t.lower() for t in raw]
    shouting = text.isupper()
    items, i = [], 0
    while i < len(low):
        took = False
        for n in (3, 2):
            if i + n <= len(low) and not any(t in BOUNDARY for t in low[i:i + n]):
                key = " ".join(low[i:i + n])
                if key in PHRASES:
                    base = PHRASES[key]
                    # phrases already encode their own polarity (e.g. "no problem"), so no look-back negation
                    value = base * (1 + _booster(low, i))
                    items.append((i, value))
                    i += n
                    took = True
                    break
        if took:
            continue
        w = low[i]
        if w in LEXICON:
            value = LEXICON[w] * (1 + _booster(low, i))
            post = i + 1 < len(low) and low[i + 1] in POST_NEGATORS
            if _is_negated(low, i) or post:
                value *= NEGATION_SCALE
            if not shouting and len(raw[i]) >= 3 and raw[i].isupper():
                value *= 1.15
            items.append((i, value))
        i += 1

    contrast_idx = max((k for k, t in enumerate(low) if t in CONTRAST), default=-1)
    total = 0.0
    for idx, value in items:
        if contrast_idx != -1 and idx < contrast_idx and any(j > contrast_idx for j, _ in items):
            value *= 0.5
        total += value
    if total != 0:
        total += math.copysign(0.29 * min(low.count("!"), 3), total)
    score = total / math.sqrt(total * total + 15)
    label = POSITIVE if score >= POS_THRESHOLD else NEGATIVE if score <= NEG_THRESHOLD else NEUTRAL
    return {"label": label, "score": round(score, 4), "language": language, "matches": len(items)}


ANALYSIS_COLUMNS = ["sentiment", "sentiment_score", "language", "lexicon_matches", "low_signal", "rating_mismatch"]


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment columns. rating_mismatch flags text/stars disagreement for manual checking only."""
    out = df.copy()
    results = [analyze_text(t) for t in out["review_text"]] if len(out) else []
    out["sentiment"] = pd.Series([r["label"] for r in results], index=out.index, dtype="object")
    out["sentiment_score"] = pd.Series([r["score"] for r in results], index=out.index, dtype="float64")
    out["language"] = pd.Series([r["language"] for r in results], index=out.index, dtype="object")
    out["lexicon_matches"] = pd.Series([r["matches"] for r in results], index=out.index, dtype="int64")
    out["low_signal"] = (out["lexicon_matches"] == 0) & (out["sentiment"] != UNCLASSIFIED)
    rating = out["rating"] if "rating" in out.columns else pd.Series(float("nan"), index=out.index)
    out["rating_mismatch"] = ((rating <= 2) & (out["sentiment"] == POSITIVE)) | ((rating >= 4) & (out["sentiment"] == NEGATIVE))
    return out
