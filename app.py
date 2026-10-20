"""Travel.pk Customer Review Sentiment Analyzer - Streamlit dashboard."""
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.competitor_analysis import company_summary, dataset_warnings, sample_size_notes, sentiment_counts
from src.data_loader import (PROV_LIVE, PROV_SYNTHETIC, PROVENANCE_LABELS, combine_frames, empty_frame, export_csv,
                             filter_reviews, load_csv, prepare_reviews, synthetic_demo, template_csv)
from src.review_collector import CANDIDATE_SOURCES, check_robots, collect_from_url
from src.takeaways import generate_takeaways
from src.theme_extractor import theme_by_company, theme_excerpts, theme_summary

NAVY, BLUE, MID, PALE = "#0B1F44", "#2F80ED", "#B8C4D9", "#EAF1FB"
SENT_COLORS = {"Positive": BLUE, "Neutral": MID, "Negative": NAVY}
MODE_REAL, MODE_DEMO = "Real data (uploaded CSV / live-collected)", "Synthetic demo (NOT real reviews)"

st.set_page_config(page_title="Travel.pk Customer Review Sentiment Analyzer", page_icon=None, layout="wide")
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: "Segoe UI", "Inter", Arial, sans-serif; }}
.hero {{ background: {NAVY}; color: #fff; padding: 1.4rem 1.6rem; border-radius: 10px; margin-bottom: 1rem; }}
.hero h1 {{ color: #fff; font-size: clamp(1.3rem, 3.2vw, 2rem); margin: 0 0 .3rem 0; padding: 0; }}
.hero p {{ color: #CFE0FA; margin: 0; font-size: clamp(.9rem, 2vw, 1.05rem); }}
.kpi {{ background: {PALE}; border-left: 4px solid {BLUE}; border-radius: 8px; padding: .7rem .9rem; margin-bottom: .6rem; }}
.kpi .l {{ color: #44546F; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .v {{ color: {NAVY}; font-size: 1.55rem; font-weight: 700; line-height: 1.2; }}
.kpi .s {{ color: #44546F; font-size: .8rem; }}
.badge {{ display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; background:{PALE}; color:{NAVY}; }}
.badge.demo {{ background:#FFF3CD; color:#664D03; }}
</style>
<div class="hero"><h1>Travel.pk Customer Review Sentiment Analyzer</h1>
<p>Public review insights, competitor comparison, and 3 evidence-based takeaways.</p></div>
""", unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def fmt_pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.1f}%"


def style_fig(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Segoe UI, Arial", color=NAVY), legend=dict(orientation="h", y=-0.25))
    return fig


@st.cache_data(show_spinner="Reading and analyzing file...")
def cached_upload(name, data):
    return load_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def cached_demo():
    return synthetic_demo()


# ------------------------------------------------------------------ sidebar: data
st.sidebar.header("1. Data")
mode = st.sidebar.radio("Data mode", [MODE_REAL, MODE_DEMO], help="Synthetic demo data is never mixed with real data.")
is_demo = mode == MODE_DEMO
reports = []
st.session_state.setdefault("live_frames", [])
if is_demo:
    base = cached_demo()
    st.sidebar.warning("Synthetic demonstration data: generated examples with fictional company names. Not real reviews.")
else:
    files = st.sidebar.file_uploader("Upload review CSV file(s)", type=["csv"], accept_multiple_files=True,
                                     help="Columns: company, review_text (required); rating, date, source, url, collected_at (optional). Template is in the Data Management tab.")
    frames = []
    for f in files or []:
        df_up, rep = cached_upload(f.name, f.getvalue())
        reports.append((f.name, rep))
        frames.append(df_up)
    base = combine_frames(frames + st.session_state["live_frames"])

if base.empty:
    st.info("**Getting started:** 1) Open the *Data Management* tab and download the CSV template. 2) Fill it with public reviews you are permitted to use "
            "(Travel.pk and competitors) and upload it in the sidebar. 3) Explore the tabs; the *3 Takeaways* tab updates automatically. "
            "Just want to see how it works? Choose **Synthetic demo** in the sidebar (clearly labelled, not real).")
    filtered = base
    focal = ""
else:
    st.sidebar.header("2. Filters")
    st.sidebar.caption("Filters apply to every tab, including the takeaways.")
    companies = sorted(base["company"].unique())
    sel_companies = st.sidebar.multiselect("Company", companies, default=companies)
    sel_sent = st.sidebar.multiselect("Sentiment", ["Positive", "Neutral", "Negative", "Unclassified"], default=["Positive", "Neutral", "Negative", "Unclassified"])
    sources = sorted(base["source"].unique())
    sel_sources = st.sidebar.multiselect("Source", sources, default=sources)
    rating_range, inc_unrated = None, True
    if base["rating"].notna().any():
        rating_range = st.sidebar.slider("Star rating", 1.0, 5.0, (1.0, 5.0), step=0.5)
        inc_unrated = st.sidebar.checkbox("Include reviews without a rating", value=True)
    date_range, inc_undated = None, True
    if base["date"].notna().any():
        dmin, dmax = base["date"].min().date(), base["date"].max().date()
        picked = st.sidebar.date_input("Review date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            date_range = picked
        inc_undated = st.sidebar.checkbox("Include reviews without a date", value=True)
    query = st.sidebar.text_input("Search text / company")
    filtered = filter_reviews(base, sel_companies, sel_sent, sel_sources, rating_range, inc_unrated, date_range, inc_undated, query)

    st.sidebar.header("3. Focus company")
    present = sorted(filtered["company"].unique()) if not filtered.empty else companies
    default_idx = next((i for i, c in enumerate(present) if "travel.pk" in c.lower()), 0)
    focal = st.sidebar.selectbox("Takeaways are written about", present, index=default_idx if present else None)

tabs = st.tabs(["Overview", "Company Comparison", "Common Praise", "Common Complaints", "Review Explorer", "Data Management", "3 Takeaways"])


def empty_notice():
    st.info("No reviews in the current selection. Upload data, load the synthetic demo, or relax the filters.")


# ------------------------------------------------------------------ Overview
with tabs[0]:
    if filtered.empty:
        empty_notice()
    else:
        origin = "demo" if (filtered["provenance"] == PROV_SYNTHETIC).any() else "real"
        label = "SYNTHETIC DEMO DATA" if origin == "demo" else "User-provided / collected data"
        st.markdown(f'<span class="badge {"demo" if origin == "demo" else ""}">{label}</span>', unsafe_allow_html=True)
        for w in dataset_warnings(filtered):
            (st.warning if ("SYNTHETIC" in w or "too few" in w) else st.caption)(w)
        c = sentiment_counts(filtered)
        rated = filtered["rating"].dropna()
        dates = filtered["date"].dropna()
        a, b, d = st.columns(3)
        with a:
            kpi("Valid reviews analyzed", f"{c['total']:,}", f"{c['classified']:,} classified, {c['unclassified']} unclassified")
        with b:
            kpi("Companies represented", filtered["company"].nunique())
        with d:
            kpi("Average star rating", f"{rated.mean():.2f}" if len(rated) else "n/a", f"from {len(rated)} rated review(s)")
        a, b, d, e = st.columns(4)
        with a:
            kpi("Positive", fmt_pct(c["positive_pct"]), f"{c['positive']} of {c['classified']} classified")
        with b:
            kpi("Neutral", fmt_pct(c["neutral_pct"]), f"{c['neutral']} of {c['classified']} classified")
        with d:
            kpi("Negative", fmt_pct(c["negative_pct"]), f"{c['negative']} of {c['classified']} classified")
        with e:
            kpi("Date range", f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}" if len(dates) else "n/a", f"{len(dates)} dated review(s)")
        fig = px.bar(pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"], "Reviews": [c["positive"], c["neutral"], c["negative"]]}),
                     x="Sentiment", y="Reviews", color="Sentiment", color_discrete_map=SENT_COLORS, text="Reviews", title="Sentiment of classified reviews (counts)")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        st.subheader("Data source coverage")
        cov = filtered.groupby(["provenance", "source", "company"]).size().reset_index(name="reviews")
        cov["provenance"] = cov["provenance"].map(PROVENANCE_LABELS)
        st.dataframe(cov, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ Comparison
with tabs[1]:
    if filtered.empty:
        empty_notice()
    else:
        summ = company_summary(filtered)
        for n in sample_size_notes(summ):
            st.warning(n)
        st.caption("This view reports observed differences within the supplied samples; it does not rank companies as better or worse.")
        plot = summ[summ["classified"] > 0].copy()
        plot["label"] = plot["company"] + " (n=" + plot["classified"].astype(str) + ")"
        if plot.empty:
            st.info("No classified reviews to compare.")
        else:
            fig = go.Figure()
            for s, col in (("positive", "pos_pct"), ("neutral", "neu_pct"), ("negative", "neg_pct")):
                fig.add_bar(name=s.title(), x=plot["label"], y=plot[col], marker_color=SENT_COLORS[s.title()],
                            customdata=plot[s], hovertemplate="%{x}<br>" + s.title() + ": %{y:.1f}% (%{customdata} reviews)<extra></extra>")
            fig.update_layout(barmode="stack", yaxis_title="% of classified reviews", title="Sentiment distribution by company")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            vol = px.bar(summ, x="company", y="reviews", text="reviews", title="Review volume by company", color_discrete_sequence=[BLUE])
            st.plotly_chart(style_fig(vol, 300), use_container_width=True)
        with c2:
            rt = summ[summ["avg_rating"].notna()].copy()
            if rt.empty:
                st.info("No star ratings available, so average ratings cannot be compared.")
            else:
                rt["text"] = rt["avg_rating"].astype(str) + " (n=" + rt["rated_reviews"].astype(str) + ")"
                fig = px.bar(rt, x="company", y="avg_rating", text="text", title="Average star rating (rated reviews only)", color_discrete_sequence=[NAVY])
                fig.update_yaxes(range=[0, 5])
                st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        st.dataframe(summ.drop(columns=["small_sample"]), use_container_width=True, hide_index=True)
        for pol, title in (("positive", "Praise themes by company"), ("negative", "Complaint themes by company")):
            st.subheader(title)
            tbc = theme_by_company(filtered, pol)
            if tbc.empty:
                st.caption("No supporting reviews found for any theme.")
            else:
                top = tbc.groupby("theme")["count"].sum().nlargest(6).index
                sub = tbc[tbc["theme"].isin(top)].copy()
                sub["share"] = sub["count"].astype(str) + "/" + sub["denominator"].astype(str)
                fig = px.bar(sub, x="theme", y="pct", color="company", barmode="group", hover_data=["share"],
                             labels={"pct": "% of that company's reviews of this sentiment"}, color_discrete_sequence=[NAVY, BLUE, MID, "#6C8EBF", "#1B3A73"])
                st.plotly_chart(style_fig(fig), use_container_width=True)
                st.dataframe(sub[["company", "theme", "count", "denominator", "pct"]], use_container_width=True, hide_index=True)


def theme_tab(polarity, noun):
    if filtered.empty:
        empty_notice()
        return
    options = ["All companies"] + sorted(filtered["company"].unique())
    choice = st.selectbox(f"Company ({noun})", options, key=f"co_{polarity}")
    sub = filtered if choice == options[0] else filtered[filtered["company"] == choice]
    summ = theme_summary(sub, polarity)
    word = "Positive" if polarity == "positive" else "Negative"
    denom = int((sub["sentiment"] == word).sum())
    st.caption(f"Candidate themes matched by keywords in {denom} {word}-classified review(s). A theme is shown only if at least one review supports it. "
               "A review can mention several themes, so percentages may sum to more than 100%.")
    if polarity == "negative":
        st.caption("Complaints are allegations made by individual reviewers and are not verified company-wide facts.")
    if summ.empty:
        st.info(f"No {noun} themes found: {denom} {word}-classified review(s) in this selection. A lack of evidence is not evidence of satisfaction or dissatisfaction.")
        return
    fig = px.bar(summ.head(10), x="pct", y="theme", orientation="h", text=summ.head(10).apply(lambda r: f"{r['count']}/{r['denominator']}", axis=1),
                 labels={"pct": f"% of {word} reviews", "theme": ""}, color_discrete_sequence=[BLUE if polarity == "positive" else NAVY])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.dataframe(summ, use_container_width=True, hide_index=True)
    for _, row in summ.head(6).iterrows():
        with st.expander(f"{row['theme']}: {row['count']} of {row['denominator']} ({row['pct']}%)"):
            for ex in theme_excerpts(sub, polarity, row["theme"]):
                st.markdown(f"> {ex['excerpt']}")
                st.caption(f"{ex['company']} - {ex['source']}")


with tabs[2]:
    theme_tab("positive", "praise")
with tabs[3]:
    theme_tab("negative", "complaint")

# ------------------------------------------------------------------ Explorer
with tabs[4]:
    if filtered.empty:
        empty_notice()
    else:
        st.caption(f"Showing {len(filtered):,} of {len(base):,} reviews after filters. Check the sentiment label against the original text; 'rating_mismatch' marks text/stars disagreement.")
        cols = ["company", "source", "date", "rating", "sentiment", "sentiment_score", "language", "review_text", "url", "provenance", "low_signal", "rating_mismatch", "near_duplicate"]
        st.dataframe(filtered[cols], use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("url"), "review_text": st.column_config.TextColumn("review_text", width="large"),
                                    "date": st.column_config.DateColumn("date")})

# ------------------------------------------------------------------ Data management
with tabs[5]:
    st.subheader("Template and downloads")
    st.download_button("Download CSV template", template_csv(), file_name="review_template.csv", mime="text/csv")
    st.caption("Required columns: company, review_text. Optional: rating (1-5), date (YYYY-MM-DD), source, url, collected_at. Leave unknown values blank; never guess them.")
    if not filtered.empty:
        st.download_button("Download cleaned + analyzed CSV (current selection)", export_csv(filtered), file_name="analyzed_reviews.csv", mime="text/csv")
        if (filtered["provenance"] == PROV_SYNTHETIC).any():
            st.warning("This export contains SYNTHETIC demonstration rows (provenance = synthetic_demo).")
    st.subheader("Validation messages")
    if is_demo:
        st.caption("Demo mode: no files uploaded.")
    elif not reports and not st.session_state["live_frames"]:
        st.caption("No files uploaded yet.")
    for name, rep in reports:
        with st.expander(f"{name}: {rep.stats.get('rows_in', 0)} row(s) read, {rep.stats.get('rows_out', 0)} kept", expanded=not rep.ok):
            for e in rep.errors:
                st.error(e)
            for w in rep.warnings:
                st.warning(w)
            if rep.ok and not rep.warnings:
                st.success("No validation problems found.")
            st.json(rep.stats)
    st.subheader("Data-source and collection status")
    st.caption("No live collection has been run automatically. Genuine review data must be uploaded, or read from a single public page below.")
    st.table(pd.DataFrame(CANDIDATE_SOURCES, columns=["Source", "Status / guidance"]))
    st.caption("Competitors such as Bookme.pk and Sastaticket.pk are candidates only: confirm they offer comparable services and have public reviews before including them.")
    with st.expander("Optional: read schema.org review markup from ONE public page"):
        st.caption("Respects robots.txt, waits between requests, never bypasses logins/CAPTCHAs, and does not store reviewer names. Many pages expose no such markup; "
                   "then nothing is collected. You are responsible for checking the site's terms.")
        url = st.text_input("Public page URL")
        comp = st.text_input("Company these reviews belong to")
        agree = st.checkbox("I have checked that the site's terms permit this")
        c1, c2 = st.columns(2)
        if c1.button("Check robots.txt"):
            res = check_robots(url) if url else None
            if res is None:
                st.error("Enter a URL first.")
            else:
                (st.success if res.allowed else st.error)(f"{res.status}: {res.detail}")
        if c2.button("Read page", disabled=not agree):
            with st.spinner("Checking robots.txt and reading the page..."):
                result = collect_from_url(url, comp)
            for m in result.messages:
                (st.success if result.status == "ok" else st.warning)(m)
            if result.status == "ok":
                live, rep = prepare_reviews(result.records, PROV_LIVE)
                st.session_state["live_frames"] = st.session_state["live_frames"] + [live]
                st.success(f"Added {len(live)} review(s). Reload the view by interacting with any control.")
    if st.session_state["live_frames"] and st.button("Clear live-collected reviews"):
        st.session_state["live_frames"] = []

# ------------------------------------------------------------------ Takeaways
with tabs[6]:
    st.subheader("3 takeaways")
    if base.empty:
        st.info("Upload data (or choose the synthetic demo) to generate takeaways.")
    else:
        if set(sel_sent) != {"Positive", "Neutral", "Negative", "Unclassified"}:
            st.warning("A sentiment filter is active, so these takeaways describe only the filtered reviews.")
        st.caption("Recomputed from the current selection. 'Observed finding' = measured from reviews. 'Recommendation' = suggestion, not a finding.")
        items = generate_takeaways(filtered, focal, is_synthetic=bool(is_demo))
        for t in items:
            with st.container(border=True):
                st.markdown(f"### {t['number']}. {t['headline']}")
                st.markdown(f"**{t['type']}**")
                st.write(t["explanation"])
                st.markdown(f"**Key statistic:** {t['statistic']}")
                st.markdown(f"**Companies:** {', '.join(t['companies']) if t['companies'] else 'n/a'}")
                st.markdown(f"**Evidence basis:** {t['evidence_basis']}")
                st.caption(f"Limitation: {t['limitation']}")
                if t["recommendation"]:
                    st.info(t["recommendation"])
