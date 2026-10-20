import io

import pandas as pd

from src.competitor_analysis import sentiment_counts
from src.data_loader import (PROV_SYNTHETIC, PROV_UPLOAD, combine_frames, filter_reviews, load_csv, synthetic_demo,
                             template_csv)
from tests.helpers import make_df


def csv(text):
    return io.BytesIO(text.encode("utf-8"))


def test_valid_csv_loads_and_analyzes():
    df, rep = load_csv(csv("company,review_text,rating,date\nA,Great service,5,2025-01-02\nA,Terrible rude staff,1,2025-01-03\n"))
    assert rep.ok and len(df) == 2
    assert set(df["sentiment"]) == {"Positive", "Negative"}
    assert (df["provenance"] == PROV_UPLOAD).all()


def test_missing_required_columns():
    df, rep = load_csv(csv("company,rating\nA,5\n"))
    assert not rep.ok and df.empty and "review_text" in rep.errors[0]


def test_empty_file_and_empty_data():
    df, rep = load_csv(csv(""))
    assert not rep.ok and df.empty
    df2, rep2 = load_csv(io.StringIO(template_csv()))
    assert df2.empty
    assert sentiment_counts(df2)["positive_pct"] is None  # no division by zero


def test_invalid_ratings_and_dates_become_blank_with_warning():
    df, rep = load_csv(csv("company,review_text,rating,date\nA,Good,9,2025-01-01\nA,Bad one,abc,notadate\nA,Fine okay,4,\n"))
    assert df["rating"].isna().sum() == 2 and df["rating"].dropna().iloc[0] == 4.0
    assert df["date"].isna().sum() == 2
    assert rep.stats["invalid_ratings"] == 2 and rep.stats["invalid_dates"] == 1
    assert any("rating" in w for w in rep.warnings)


def test_missing_rating_and_date_columns_are_ok():
    df, rep = load_csv(csv("company,review_text\nA,Good service\n"))
    assert rep.ok and df["rating"].isna().all() and df["date"].isna().all()


def test_blank_text_and_company_dropped():
    df, rep = load_csv(csv("company,review_text\nA,\nA,   \n,Good service\nA,Good service\n"))
    assert len(df) == 1 and rep.stats["dropped_missing_text"] == 2 and rep.stats["dropped_missing_company"] == 1


def test_exact_duplicates_removed_per_company_and_near_flagged():
    df, rep = load_csv(csv(
        "company,review_text\nA,Great service!\nA,great service\nB,Great service\n"
        "A,The agent was helpful and the booking process was easy today\nA,The agent was helpful and the booking process was easy today!!\n"))
    assert rep.stats["exact_duplicates_removed"] == 2
    assert len(df) == 3 and set(df["company"]) == {"A", "B"}
    df2, _ = load_csv(csv("company,review_text\nA,The agent was helpful and the booking process was easy today\n"
                          "A,The agent was helpful and the booking process was easy today ok\n"))
    assert df2["near_duplicate"].sum() == 1 and len(df2) == 2  # flagged, not removed


def test_sentiment_counts_and_percentages():
    df = make_df([("A", "Great and helpful"), ("A", "Excellent service"), ("A", "Terrible rude agent"), ("A", "Booked a flight for next month")])
    c = sentiment_counts(df)
    assert (c["positive"], c["neutral"], c["negative"], c["classified"]) == (2, 1, 1, 4)
    assert c["positive_pct"] == 50.0 and c["neutral_pct"] == 25.0


def test_unclassified_excluded_from_denominator():
    df = make_df([("A", "Great and helpful"), ("A", "یہ بہت اچھا ہے")])
    c = sentiment_counts(df)
    assert c["total"] == 2 and c["classified"] == 1 and c["positive_pct"] == 100.0 and c["unclassified"] == 1


def test_filters_and_denominators():
    df = make_df([("A", "Great and helpful", 5, "2025-01-01", "S1"), ("A", "Terrible rude agent", 1, "2025-03-01", "S2"),
                  ("B", "Good price", "", "", "S1"), ("B", "Awful delayed flight", 2, "2025-02-01", "S1")])
    assert len(filter_reviews(df, companies=["A"])) == 2
    assert len(filter_reviews(df, sources=["S1"])) == 3
    assert len(filter_reviews(df, rating_range=(4, 5), include_unrated=False)) == 1
    assert len(filter_reviews(df, rating_range=(4, 5), include_unrated=True)) == 2
    assert len(filter_reviews(df, date_range=("2025-01-01", "2025-02-15"), include_undated=False)) == 2
    assert len(filter_reviews(df, query="delayed")) == 1
    none = filter_reviews(df, companies=["Z"])
    assert none.empty and sentiment_counts(none)["negative_pct"] is None


def test_synthetic_data_labelled_and_separate():
    demo = synthetic_demo()
    assert (demo["provenance"] == PROV_SYNTHETIC).all()
    assert demo["company"].str.contains("synthetic").all()
    assert not demo["company"].str.lower().str.contains("travel.pk|bookme|sastaticket").any()
    real = make_df([("A", "Great and helpful")])
    merged = combine_frames([real])
    assert (merged["provenance"] == PROV_UPLOAD).all()  # demo never merged into real frames unless explicitly passed
