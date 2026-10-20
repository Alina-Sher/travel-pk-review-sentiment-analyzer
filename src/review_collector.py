"""Conservative public-page reader. It does NOT bypass logins, CAPTCHAs, paywalls or rate limits and
does not use private/unofficial APIs. It reads only schema.org JSON-LD 'Review' markup that a public page
already embeds, from a single URL the user supplies, and only if robots.txt allows it.

Reviewer names/avatars are never stored. Nothing here has been run against a live site in this build
(the development sandbox had no internet); parsing is unit-tested on local HTML only."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd

CANDIDATE_SOURCES = [
    ("Trustpilot company pages", "Check the site's terms; automated collection may be restricted. Export or copy reviews you are permitted to use into the CSV template."),
    ("Google Maps / Google Business reviews", "Use official, permitted access (e.g. an official API with your own credentials) or manual export. Not implemented here."),
    ("Google Play / Apple App Store reviews", "Relevant where the company has a booking app. Use permitted exports or manual copying into the CSV template."),
    ("User-provided CSV", "Fully supported. Use the template in the Data Management tab."),
]
BLOCK_MARKERS = ("captcha", "are you a robot", "access denied", "verify you are human", "cf-challenge")
HEADERS_NOTE = "TravelPkReviewAnalyzer/1.0 (assessment project)"


@dataclass
class AccessResult:
    allowed: bool | None
    status: str
    detail: str = ""


@dataclass
class CollectionResult:
    records: pd.DataFrame
    status: str
    messages: list = field(default_factory=list)


def _user_agent() -> str:
    contact = os.environ.get("COLLECTOR_CONTACT_EMAIL", "").strip()
    return HEADERS_NOTE + (f" contact: {contact}" if contact else "")


def _default_get(url, **kw):
    import requests
    return requests.get(url, timeout=10, headers={"User-Agent": _user_agent()}, **kw)


def check_robots(url: str, http_get=None) -> AccessResult:
    """allowed=True/False from robots.txt; allowed=None means it could not be determined (treated as NOT allowed)."""
    http_get = http_get or _default_get
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return AccessResult(False, "invalid_url", "Enter a full http(s) URL.")
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        resp = http_get(robots_url)
    except Exception as exc:
        return AccessResult(None, "robots_unreachable", f"Could not fetch robots.txt ({exc}). Not proceeding.")
    code = getattr(resp, "status_code", 0)
    if code in (401, 403):
        return AccessResult(False, "robots_denied", f"robots.txt returned HTTP {code}; treating as disallowed.")
    if code == 404:
        return AccessResult(True, "no_robots_txt", "No robots.txt found (HTTP 404). Site terms may still restrict automated access.")
    if code != 200:
        return AccessResult(None, "robots_error", f"robots.txt returned HTTP {code}. Not proceeding.")
    rp = RobotFileParser()
    rp.parse(resp.text.splitlines())
    if rp.can_fetch(_user_agent(), url):
        return AccessResult(True, "allowed", "robots.txt allows this URL. Site terms may still restrict automated access.")
    return AccessResult(False, "disallowed", "robots.txt disallows this URL.")


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def parse_jsonld_reviews(html: str, company: str, page_url: str) -> pd.DataFrame:
    """Extract schema.org Review nodes. Author identity is intentionally ignored."""
    from bs4 import BeautifulSoup
    rows = []
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (ValueError, TypeError):
            continue
        for node in _walk(data):
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if "Review" not in types:
                continue
            body = node.get("reviewBody") or node.get("description") or ""
            if not str(body).strip():
                continue
            rating = node.get("reviewRating")
            value = rating.get("ratingValue") if isinstance(rating, dict) else None
            rows.append({"company": company, "review_text": str(body), "rating": value if value is not None else "",
                         "date": node.get("datePublished", ""), "source": urlparse(page_url).netloc, "url": page_url,
                         "collected_at": pd.Timestamp.now(tz="UTC").isoformat()})
    return pd.DataFrame(rows, columns=["company", "review_text", "rating", "date", "source", "url", "collected_at"])


def collect_from_url(url: str, company: str, http_get=None, delay_seconds: float = 2.0) -> CollectionResult:
    """Single-page, robots-checked read. Returns an honest status; never fabricates records."""
    empty = pd.DataFrame(columns=["company", "review_text", "rating", "date", "source", "url", "collected_at"])
    if not company.strip():
        return CollectionResult(empty, "error", ["Enter the company name these reviews belong to."])
    access = check_robots(url, http_get)
    if access.allowed is not True:
        return CollectionResult(empty, "blocked", [access.detail, "Use the CSV upload instead."])
    http_get = http_get or _default_get
    time.sleep(delay_seconds)
    try:
        resp = http_get(url)
    except Exception as exc:
        return CollectionResult(empty, "error", [f"Request failed: {exc}"])
    if resp.status_code in (401, 403, 429) or any(m in resp.text.lower() for m in BLOCK_MARKERS):
        return CollectionResult(empty, "blocked", ["The page requires verification, login, or is rate-limiting. No attempt is made to bypass this.", "Use the CSV upload instead."])
    if resp.status_code != 200:
        return CollectionResult(empty, "error", [f"HTTP {resp.status_code}."])
    df = parse_jsonld_reviews(resp.text, company.strip(), url)
    if df.empty:
        return CollectionResult(df, "no_structured_reviews", ["The page loaded, but exposes no schema.org review data. Nothing was collected (this is not a complete dataset check)."])
    return CollectionResult(df, "ok", [f"Read {len(df)} review(s) from this single page. Pagination is not followed, so this is likely a partial sample."])
