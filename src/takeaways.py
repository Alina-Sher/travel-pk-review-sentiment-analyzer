"""Exactly three takeaways, computed from the selected data. Nothing here is hardcoded about any company."""
from __future__ import annotations

import pandas as pd

from .competitor_analysis import sentiment_counts
from .theme_extractor import theme_summary

MIN_FOCAL_CLASSIFIED = 10     # focal company needs this many classified reviews
MIN_POLARITY_REVIEWS = 5      # positive (or negative) focal reviews needed to rank themes
MIN_THEME_COUNT = 3           # a theme needs this many supporting reviews to be called recurring
MIN_COMPARATOR_DENOM = 10     # competitor pool needs this many reviews of the same polarity
MIN_GAP_PP = 10.0             # percentage-point gap needed to call a difference notable

FINDING, STATUS, DEMO = "Observed finding", "Status: insufficient evidence", "Demonstration only (synthetic data)"
BASE_LIMIT = ("Sentiment is lexicon-based and themes are keyword-matched, so some reviews are mis-tagged; the reviews are a "
              "provided sample, not a random sample of all customers; differences were not tested for statistical significance.")


def _item(number, kind, headline, explanation, statistic, companies, basis, limitation, recommendation=None, synthetic=False):
    if synthetic:
        kind = DEMO
        limitation = "SYNTHETIC DATA - not a real finding. " + limitation
    return {"number": number, "type": kind, "headline": headline, "explanation": explanation, "statistic": statistic,
            "companies": list(companies), "evidence_basis": basis, "limitation": limitation, "recommendation": recommendation}


def _pct(c, d):
    return round(100.0 * c / d, 1) if d else None


def _status(number, topic, why, focal, basis, synthetic):
    return _item(number, STATUS, f"{topic}: not enough evidence to conclude", why, "n/a", [focal] if focal else [],
                 basis, "No finding is stated because the sample does not support one. " + BASE_LIMIT, None, synthetic)


def _theme_row(summary: pd.DataFrame, theme: str):
    r = summary[summary["theme"] == theme]
    return int(r["count"].iloc[0]) if len(r) else 0


def _single_polarity(number, polarity, focal, focal_df, comp_df, f, c, synthetic):
    label = "praise" if polarity == "positive" else "complaint"
    word = "Positive" if polarity == "positive" else "Negative"
    denom = f[polarity]
    basis = f"{denom} {word}-classified {focal} review(s) out of {f['classified']} classified."
    topic = "Customer praise" if polarity == "positive" else "Customer complaints"
    if denom < MIN_POLARITY_REVIEWS:
        return _status(number, topic, f"Only {denom} {word}-classified review(s) for {focal}; at least {MIN_POLARITY_REVIEWS} are needed to rank {label} themes. "
                       + ("A lack of negative reviews in a sample does not prove customers are satisfied." if polarity == "negative" else ""),
                       focal, basis, synthetic)
    summ = theme_summary(focal_df, polarity)
    rows = summ[summ["count"] >= MIN_THEME_COUNT]
    if rows.empty:
        return _status(number, topic, f"No {label} theme reached {MIN_THEME_COUNT} supporting reviews among {denom} {word} review(s). "
                       + ("This does not show that customers are satisfied; it only reflects this sample." if polarity == "negative" else ""),
                       focal, basis, synthetic)
    top = rows.iloc[0]
    theme, count = top["theme"], int(top["count"])
    pct = _pct(count, denom)
    text = f"{count} of {denom} {word}-classified {focal} reviews ({pct}%) mention {theme.lower()}."
    if len(rows) > 1:
        sec = rows.iloc[1]
        text += f" Next most frequent: {sec['theme'].lower()} ({int(sec['count'])} of {denom})."
    companies = [focal]
    if c[polarity] >= MIN_COMPARATOR_DENOM:
        cs = theme_summary(comp_df, polarity)
        cc = _theme_row(cs, theme)
        text += f" Across the {comp_df['company'].nunique()} other company/companies, the same theme appears in {cc} of {c[polarity]} {word} reviews ({_pct(cc, c[polarity])}%)."
        companies += sorted(comp_df["company"].unique())
        basis += f" Comparison pool: {c[polarity]} {word} competitor review(s)."
    else:
        text += f" No competitor comparison: only {c[polarity]} {word} competitor review(s) (need {MIN_COMPARATOR_DENOM})."
    if polarity == "negative":
        text += " These are allegations made in individual reviews, not verified company-wide facts."
    head = f"Most-cited {label} for {focal}: {theme}"
    return _item(number, FINDING, head, text, f"{count}/{denom} ({pct}%)", companies, basis, BASE_LIMIT, None, synthetic)


def _best_gap(focal_df, comp_df, f, c):
    if f["classified"] < MIN_FOCAL_CLASSIFIED or c["classified"] < MIN_FOCAL_CLASSIFIED:
        return None
    best = None
    for pol in ("positive", "negative"):
        fd, cd = f[pol], c[pol]
        if fd < MIN_POLARITY_REVIEWS or cd < MIN_COMPARATOR_DENOM:
            continue
        fs, cs = theme_summary(focal_df, pol), theme_summary(comp_df, pol)
        for theme in sorted(set(fs["theme"]) | set(cs["theme"])):
            fc, cc = _theme_row(fs, theme), _theme_row(cs, theme)
            if fc < MIN_THEME_COUNT and cc < MIN_THEME_COUNT:
                continue
            gap = 100.0 * fc / fd - 100.0 * cc / cd
            if abs(gap) >= MIN_GAP_PP and (best is None or abs(gap) > abs(best["gap"])):
                best = {"pol": pol, "theme": theme, "fc": fc, "fd": fd, "cc": cc, "cd": cd, "gap": gap}
    return best


def generate_takeaways(df: pd.DataFrame, focal_company: str, is_synthetic: bool = False) -> list:
    """Always returns exactly three takeaway dicts (praise, complaints, competitive insight)."""
    syn = bool(is_synthetic)
    if df is None or df.empty or not focal_company or focal_company not in set(df["company"]):
        n = 0 if df is None else len(df)
        why = f"The current selection has {n} review(s) and none for '{focal_company}'." if focal_company else "No focus company is selected."
        return [_status(1, "Customer praise", why, focal_company, "No reviews for the focus company.", syn),
                _status(2, "Customer complaints", why + " The absence of reviews says nothing about customer satisfaction.", focal_company, "No reviews for the focus company.", syn),
                _status(3, "Competitive insight", why, focal_company, "No reviews for the focus company.", syn)]
    focal_df, comp_df = df[df["company"] == focal_company], df[df["company"] != focal_company]
    f, c = sentiment_counts(focal_df), sentiment_counts(comp_df)
    if f["classified"] < MIN_FOCAL_CLASSIFIED:
        basis = f"{f['classified']} classified review(s) for {focal_company}."
        why = f"{focal_company} has {f['classified']} classified review(s); at least {MIN_FOCAL_CLASSIFIED} are needed before themes are ranked."
        return [_status(1, "Customer praise", why, focal_company, basis, syn),
                _status(2, "Customer complaints", why + " A small sample cannot show that complaints are absent.", focal_company, basis, syn),
                _status(3, "Competitive insight", why, focal_company, basis, syn)]

    t1 = _single_polarity(1, "positive", focal_company, focal_df, comp_df, f, c, syn)
    t2 = _single_polarity(2, "negative", focal_company, focal_df, comp_df, f, c, syn)

    gap = _best_gap(focal_df, comp_df, f, c)
    others = sorted(comp_df["company"].unique())
    if gap:
        word = "Positive" if gap["pol"] == "positive" else "Negative"
        fp, cp = _pct(gap["fc"], gap["fd"]), _pct(gap["cc"], gap["cd"])
        direction = "more" if gap["gap"] > 0 else "less"
        text = (f"{gap['theme']} appears in {gap['fc']} of {gap['fd']} {word} {focal_company} reviews ({fp}%) versus "
                f"{gap['cc']} of {gap['cd']} {word} reviews for the other companies ({cp}%): {abs(gap['gap']):.1f} percentage points {direction} often.")
        if gap["pol"] == "negative" and gap["gap"] > 0:
            rec = f"Recommendation (not an observed finding): read the sampled {gap['theme'].lower()} complaints in the Review Explorer to judge whether they point to a recurring process problem."
        elif gap["pol"] == "negative":
            rec = f"Recommendation (not an observed finding): check whether {focal_company}'s lower share of {gap['theme'].lower()} complaints persists in a larger sample."
        elif gap["gap"] > 0:
            rec = f"Recommendation (not an observed finding): confirm this strength ({gap['theme'].lower()}) with more reviews before relying on it."
        else:
            rec = f"Recommendation (not an observed finding): look at how competitors' reviews describe {gap['theme'].lower()} to see whether {focal_company} offers or communicates something comparable."
        t3 = _item(3, FINDING, f"Biggest observed gap: {gap['theme']} ({word.lower()} reviews)", text,
                   f"{fp}% vs {cp}% ({gap['fc']}/{gap['fd']} vs {gap['cc']}/{gap['cd']})", [focal_company] + others,
                   f"{f['classified']} classified {focal_company} review(s) vs {c['classified']} for {len(others)} other company/companies.",
                   BASE_LIMIT + " Largest gap among themes with enough support; other gaps exist.", rec, syn)
    elif c["classified"] >= MIN_FOCAL_CLASSIFIED:
        fn, cn = f["negative_pct"], c["negative_pct"]
        text = (f"No theme differed by at least {MIN_GAP_PP:.0f} percentage points with adequate support. Overall, {f['negative']} of {f['classified']} "
                f"{focal_company} reviews are Negative ({fn}%) versus {c['negative']} of {c['classified']} for the others ({cn}%).")
        t3 = _item(3, FINDING, "No clear theme-level gap; overall negative share compared", text,
                   f"{fn}% vs {cn}% negative", [focal_company] + others,
                   f"{f['classified']} vs {c['classified']} classified reviews.",
                   BASE_LIMIT + " Small differences in negative share should not be read as one company being better.", None, syn)
    else:
        t3 = _status(3, "Competitive insight", f"Competitors have only {c['classified']} classified review(s); at least {MIN_FOCAL_CLASSIFIED} are needed for any comparison.",
                     focal_company, f"{f['classified']} {focal_company} vs {c['classified']} competitor classified review(s).", syn)
    return [t1, t2, t3]
