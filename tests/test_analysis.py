import math

from src.competitor_analysis import company_summary, dataset_warnings, sample_size_notes
from src.review_collector import check_robots, collect_from_url, parse_jsonld_reviews
from src.theme_extractor import anonymize, tag_themes, theme_by_company, theme_excerpts, theme_summary
from tests.helpers import make_df


def test_theme_extraction_counts_and_denominator():
    df = make_df([("A", "Refund still pending, very disappointed"), ("A", "Refund never came, terrible"),
                  ("A", "Terrible rude support, no response"), ("A", "Great helpful agent")])
    s = theme_summary(df, "negative")
    refunds = s[s["theme"] == "Refunds"].iloc[0]
    assert refunds["count"] == 2 and refunds["denominator"] == 3 and refunds["pct"] == 66.7
    assert "Refunds" in tag_themes("my refund is late", "negative")
    assert not s[s["theme"] == "Hotel problems"].shape[0]  # themes without evidence are not reported


def test_theme_summary_empty_and_zero_denominator():
    df = make_df([("A", "Booked a flight for next month")])
    assert theme_summary(df, "positive").empty
    assert theme_summary(df.iloc[0:0], "negative").empty
    assert theme_by_company(df.iloc[0:0], "negative").empty


def test_excerpts_are_real_short_and_anonymized():
    df = make_df([("A", "Terrible refund experience, call me on 0300-1234567 or mail a.b@x.com about the refund")])
    ex = theme_excerpts(df, "negative", "Refunds")
    assert ex and ex[0]["company"] == "A" and "0300" not in ex[0]["excerpt"] and "@" not in ex[0]["excerpt"]
    assert "refund" in ex[0]["excerpt"].lower()
    assert anonymize("see https://x.com/abc now") == "see [link] now"


def test_company_comparison_with_sample_sizes():
    df = make_df([("A", "Great helpful agent", 5), ("A", "Terrible rude", 1), ("B", "Excellent service", ""), ("B", "Good price", "")])
    s = company_summary(df).set_index("company")
    assert s.loc["A", "reviews"] == 2 and s.loc["A", "avg_rating"] == 3.0 and s.loc["A", "pos_pct"] == 50.0
    assert s.loc["B", "avg_rating"] is None or math.isnan(s.loc["B", "avg_rating"])
    assert s.loc["B", "rated_reviews"] == 0 and bool(s.loc["A", "small_sample"])
    assert any("Small samples" in n for n in sample_size_notes(s.reset_index()))


def test_empty_summary_and_warnings():
    df = make_df([])
    assert company_summary(df).empty and dataset_warnings(df) == ["No reviews in the current selection."]


def test_small_dataset_warning_visible():
    df = make_df([("A", "Great helpful agent")])
    assert any("too few" in w for w in dataset_warnings(df))


class Resp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


def test_robots_disallow_and_unreachable_block_collection():
    deny = lambda url: Resp(200, "User-agent: *\nDisallow: /")
    assert check_robots("https://example.com/reviews", deny).allowed is False
    def boom(url):
        raise OSError("offline")
    assert check_robots("https://example.com/r", boom).allowed is None
    res = collect_from_url("https://example.com/r", "A", boom, delay_seconds=0)
    assert res.status == "blocked" and res.records.empty


def test_captcha_page_is_not_bypassed():
    def get(url):
        return Resp(404) if url.endswith("robots.txt") else Resp(200, "<html>Please solve the CAPTCHA</html>")
    res = collect_from_url("https://example.com/r", "A", get, delay_seconds=0)
    assert res.status == "blocked" and res.records.empty


def test_jsonld_parser_reads_only_embedded_reviews_without_author():
    html = """<script type="application/ld+json">{"@type":"LocalBusiness","review":[
      {"@type":"Review","author":{"name":"Secret Name"},"reviewBody":"Good service","reviewRating":{"ratingValue":"4"},"datePublished":"2025-02-01"},
      {"@type":"Review","reviewBody":""}]}</script>"""
    df = parse_jsonld_reviews(html, "A", "https://example.com/r")
    assert len(df) == 1 and "Secret Name" not in df.to_string() and df.iloc[0]["rating"] == "4"
    assert parse_jsonld_reviews("<html></html>", "A", "https://example.com/r").empty
