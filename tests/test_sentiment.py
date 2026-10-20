import pandas as pd

from src.sentiment import NEGATIVE, NEUTRAL, POSITIVE, UNCLASSIFIED, analyze_dataframe, analyze_text, detect_language


def label(t):
    return analyze_text(t)["label"]


def test_positive_examples():
    assert label("Great service, very helpful and quick!") == POSITIVE
    assert label("The hotel room was clean and staff were friendly") == POSITIVE
    assert label("Highly recommended, value for money") == POSITIVE


def test_negative_examples():
    assert label("Worst experience, waste of money") == NEGATIVE
    assert label("Refund still pending, no response from support") == NEGATIVE


def test_neutral_examples():
    assert label("I booked a flight for next month.") == NEUTRAL
    assert label("The booking was confirmed by email.") == NEUTRAL


def test_negation_and_intensifiers():
    assert label("not good at all") == NEGATIVE
    assert label("not bad") == POSITIVE
    assert label("No issues, no problem at all") == POSITIVE
    assert analyze_text("very good")["score"] > analyze_text("good")["score"]
    assert analyze_text("good!!!")["score"] > analyze_text("good")["score"]


def test_contrast_clause():
    assert label("Good price but customer support was rude") == NEGATIVE


def test_roman_urdu():
    assert label("bohat acha service") == POSITIVE
    assert label("acha nahi tha") == NEGATIVE
    assert detect_language("service bahut acha hai aur kaam bhi acha tha") == "Roman Urdu / mixed"


def test_empty_and_missing_text():
    for bad in ["", "   ", None, float("nan")]:
        r = analyze_text(bad)
        assert r["label"] == UNCLASSIFIED and r["score"] == 0.0


def test_urdu_script_not_guessed():
    assert label("یہ بہت اچھا ہے") == UNCLASSIFIED


def test_rating_is_not_used_for_sentiment():
    df = pd.DataFrame({"review_text": ["Great and helpful", "Terrible and rude"], "rating": [1.0, 5.0]})
    out = analyze_dataframe(df)
    assert list(out["sentiment"]) == [POSITIVE, NEGATIVE]
    assert list(out["rating_mismatch"]) == [True, True]  # flagged for manual checking only


def test_analyze_empty_dataframe():
    out = analyze_dataframe(pd.DataFrame({"review_text": pd.Series([], dtype="object"), "rating": pd.Series([], dtype="float64")}))
    assert len(out) == 0 and "sentiment" in out.columns
