# Travel.pk Customer Review Sentiment Analyzer

*Public review insights, competitor comparison, and 3 evidence-based takeaways.*

A Streamlit dashboard for the TechAbout assignment: analyze public customer reviews of Travel.pk and competitors, show common praise and complaints, compare sentiment, and produce exactly **3 takeaways**.

## Honest status (read first)

- **No reviews are bundled and none were collected automatically.** The app ships with no real review data. You supply genuine public reviews via CSV (template included). An optional single-page reader exists for pages that embed schema.org review markup; it was **not run against any live site** during development.
- **Competitors are candidates, not verified.** Bookme.pk and Sastaticket.pk are plausible Pakistani OTAs, but confirm they offer comparable services and have public reviews before including them.
- **Synthetic demo mode** (sidebar) generates fictional companies ("Demo Agency A (synthetic)") so you can see the UI. It is never mixed with real data and every output is labelled.
- Sentiment is **lexicon-based** (English, partial Roman Urdu). Urdu-script reviews are labelled *Unclassified* and excluded from percentages rather than guessed.

## Run on Windows

1. Install Python **3.10, 3.11 or 3.12** from python.org. Tick **"Add python.exe to PATH"**.
2. Extract this folder, open it, and double-click **`start.bat`**.
   It creates `.venv`, installs `requirements.txt`, and starts Streamlit.
3. Open the **Local URL** shown in the window (usually http://localhost:8501).

Manual alternative (Command Prompt in the project folder):

```
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Run tests: `python -m pytest`

## Using the app

1. **Data Management** tab -> download the CSV template, fill it with reviews you are permitted to use, upload it in the sidebar.
2. Use the sidebar filters (company, sentiment, source, rating, date, text search). **Every tab, including the takeaways, recalculates from the filtered data.**
3. Tabs: Overview, Company Comparison, Common Praise, Common Complaints, Review Explorer, Data Management, **3 Takeaways** (last tab).

### CSV columns

| Column | Required | Notes |
|---|---|---|
| company | yes | e.g. `Travel.pk` |
| review_text | yes | original text |
| rating | no | number 1-5; invalid values are blanked and reported |
| date | no | `YYYY-MM-DD` recommended |
| source | no | e.g. `Trustpilot` (defaults to "Uploaded CSV (source not stated)") |
| url | no | public review link; must start with http(s) |
| collected_at | no | defaults to load time |

Common alternative names (`review`, `text`, `stars`, `platform`, ...) are recognised.

## Method

- **Cleaning:** blank text/company rows dropped; exact duplicates (same company + normalised text) removed; near-duplicates (>=90% word overlap) **flagged, not removed**; every drop is reported.
- **Sentiment:** weighted word/phrase lexicon, 3-token negation, intensifiers, "but" re-weighting, `!`/CAPS emphasis, Roman Urdu terms and post-negation (`acha nahi`). Score >= 0.2 Positive, <= -0.2 Negative, else Neutral. **Star ratings are never used to set sentiment**; text/star disagreement is only flagged (`rating_mismatch`).
- **Themes:** keyword patterns (`src/theme_extractor.py`) matched against Positive (praise) or Negative (complaints) reviews. Reported only when >=1 review matches; denominators are the Positive/Negative review counts. A review can match several themes.
- **Takeaways** (`src/takeaways.py`): (1) top praise theme for the focus company with competitor comparison, (2) top complaint theme likewise, (3) the largest theme-level gap vs competitors (>=10 percentage points, adequate support), else an overall negative-share comparison. Needs >=10 classified focus reviews, >=5 of the relevant polarity, >=3 supporting reviews per theme, >=10 competitor reviews of that polarity. Otherwise it returns three labelled **status** statements instead of inventing findings. Recommendations are labelled "not an observed finding".

## Limitations

Keyword themes and lexicon sentiment misclassify sarcasm, mixed reviews and spelling variants; uploaded samples are not random samples of all customers; differences are not tested for statistical significance; reviews are individual allegations, not verified facts.

## Data sources and access

Respect each site's terms and robots.txt; do not bypass logins, CAPTCHAs or rate limits. Trustpilot, Google and app-store reviews usually need permitted export/official API access, so the supported path is the CSV upload. The optional reader (Data Management tab) checks robots.txt, waits between requests, stops at CAPTCHA/login/429 pages, ignores reviewer names, and reads one page only.

## GitHub

```
git init
git add .
git commit -m "Customer Review Sentiment Analyzer"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
`.gitignore` excludes `.venv`, caches, `.env`, and files in `data/`. Do not commit real review exports you lack rights to share.

## Streamlit Community Cloud

1. Push the repo to GitHub (above).
2. At share.streamlit.io choose **New app**, select the repo, branch `main`, main file `app.py`.
3. Deploy. Nothing needs secrets. Cloud storage is ephemeral, so users re-upload CSVs each session; the optional page reader may be blocked by sites and is not required.

(Deployment has not been performed by this project.)

## Layout

`app.py` (UI) | `src/validation.py`, `data_loader.py`, `sentiment.py`, `theme_extractor.py`, `competitor_analysis.py`, `takeaways.py`, `review_collector.py` | `tests/` | `templates/review_template.csv`
