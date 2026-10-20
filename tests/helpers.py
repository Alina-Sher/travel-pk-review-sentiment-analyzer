import pandas as pd

from src.data_loader import PROV_UPLOAD, prepare_reviews


def make_df(rows, provenance=PROV_UPLOAD):
    """rows: list of (company, text) or (company, text, rating, date, source)."""
    recs = []
    for r in rows:
        r = list(r) + [""] * (5 - len(r))
        recs.append({"company": r[0], "review_text": r[1], "rating": r[2], "date": r[3], "source": r[4] or "Test source"})
    df, _ = prepare_reviews(pd.DataFrame(recs), provenance)
    return df


def many(company, text_template, n, **kw):
    return [(company, text_template.format(i=i), *kw.get("extra", ())) for i in range(n)]
