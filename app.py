"""
app.py - SmartHire: AI-Powered Resume Analyzer and Skill Gap Detector (Streamlit front-end)

Pages: Home | Resume Analyzer | Analysis Dashboard | Skill Gap Recommendations | Resume Improvement
All intelligence lives in the other modules; this file only handles UI and session state.
Privacy: uploaded PDFs are processed in memory and never saved to disk. Data lives only in the browser session.
"""
import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_model import SIMILARITY_CEILING
from pdf_extractor import MAX_FILE_MB, MAX_PAGES, PDFValidationError, extract_text_from_pdf
from recommendations import (BULLET_TEMPLATES, analyze_resume_structure, build_improvement_suggestions,
                             build_recommendations, build_report_markdown, build_roadmap)
from skill_gap import MIN_JD_WORDS, run_pipeline

# ----------------------------------------------------------------------------- design tokens
BG, CARD, TEXT, MUTED, SAGE, BORDER = "#F8F7F3", "#FFFFFF", "#252525", "#858585", "#A3AA91", "#E9E6DF"
SAND, CLAY = "#D8C9A1", "#C99C8C"
SERIF = "Playfair Display, Georgia, serif"
SANS = "Inter, -apple-system, Segoe UI, sans-serif"

PAGES = ["Home", "Resume Analyzer", "Analysis Dashboard", "Skill Gap Recommendations", "Resume Improvement"]

st.set_page_config(page_title="SmartHire", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; color:#252525; }
.stApp { background:#F8F7F3; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2.4rem; padding-bottom:4rem; max-width:1180px; }
h1,h2,h3,h4,.sh-serif { font-family:'Playfair Display', Georgia, serif !important; font-weight:500 !important; letter-spacing:-0.01em; color:#252525; }
section[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E9E6DF; }
.sh-brand { font-family:'Playfair Display',Georgia,serif; font-size:1.6rem; color:#252525; margin-bottom:.1rem; }
.sh-tag { color:#858585; font-size:.8rem; margin-bottom:1.4rem; }
.sh-eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; color:#8B9377; font-weight:600; margin-bottom:.6rem; }
.sh-hero h1 { font-size:3.1rem; line-height:1.12; margin:0 0 1rem 0; }
.sh-lead { color:#858585; font-size:1.08rem; line-height:1.65; max-width:640px; }
.sh-muted { color:#858585; font-size:.9rem; }
.sh-card { background:#FFFFFF; border:1px solid #E9E6DF; border-radius:14px; padding:1.3rem 1.4rem; box-shadow:0 1px 2px rgba(37,37,37,.04),0 6px 18px rgba(37,37,37,.03); height:100%; }
.sh-card h4 { margin:0 0 .5rem 0; font-size:1.12rem; }
.sh-card p { color:#858585; font-size:.92rem; line-height:1.55; margin:0; }
.sh-label { color:#858585; font-size:.76rem; text-transform:uppercase; letter-spacing:.1em; }
.sh-value { font-family:'Playfair Display',Georgia,serif; font-size:2.2rem; color:#252525; line-height:1.15; margin-top:.3rem; }
.sh-sub { color:#858585; font-size:.82rem; margin-top:.15rem; }
.sh-step { font-family:'Playfair Display',Georgia,serif; color:#A3AA91; font-size:1.7rem; }
.sh-note { background:#F1F2EB; border:1px solid #E1E4D6; border-radius:12px; padding:.9rem 1.1rem; color:#555; font-size:.88rem; line-height:1.55; }
.sh-gap { height:1.1rem; }
.sh-divider { border:none; border-top:1px solid #E9E6DF; margin:1.6rem 0; }
.pill { display:inline-block; padding:4px 12px; margin:3px 6px 3px 0; border-radius:999px; font-size:.82rem; border:1px solid; }
.pill-matched { background:#EEF0E8; border-color:#A3AA91; color:#4F5640; }
.pill-partial { background:#F6F1E4; border-color:#D8C9A1; color:#7A6A3A; }
.pill-missing { background:#F8EEEA; border-color:#D9B3A5; color:#8A4F3E; }
.pill-neutral { background:#FFFFFF; border-color:#E9E6DF; color:#858585; }
.sev { font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }
.sev-High { color:#8A4F3E; } .sev-Medium { color:#7A6A3A; } .sev-Low { color:#6B7358; }
div.stButton > button, div.stDownloadButton > button { border-radius:10px; border:1px solid #E9E6DF; background:#FFFFFF; color:#252525; font-weight:500; padding:.55rem 1.2rem; box-shadow:0 1px 2px rgba(0,0,0,.04); }
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#A3AA91; color:#252525; }
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] { background:#252525; color:#F8F7F3; border-color:#252525; }
div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover { background:#3a3a3a; color:#fff; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #E9E6DF; border-radius:12px; }
[data-testid="stPlotlyChart"], [data-testid="stImage"] { background:#FFFFFF; border:1px solid #E9E6DF; border-radius:14px; padding:.6rem; box-shadow:0 1px 2px rgba(37,37,37,.04); }
[data-testid="stDataFrame"] { border:1px solid #E9E6DF; border-radius:12px; overflow:hidden; }
textarea { background:#FFFFFF !important; border-radius:10px !important; font-size:.9rem !important; }
[data-testid="stFileUploader"] section { background:#FFFFFF; border:1px dashed #C9C6BC; border-radius:12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- state helpers
def init_state():
    defaults = {"nav": "Home", "resume_text": "", "resume_meta": None, "jd_text": "",
                "analysis": None, "uploader_n": 0, "upload_error": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    # Streamlit drops widget-bound keys on pages where the widget is not drawn; re-assigning keeps them alive.
    for k in ("resume_text", "jd_text"):
        st.session_state[k] = st.session_state[k]


def go_to(page: str):
    st.session_state["nav"] = page


def ingest_pdf(data: bytes, name: str, source: str):
    """Validate + extract in memory; store only the text (never the file)."""
    ex = extract_text_from_pdf(data)
    st.session_state.resume_text = ex.text
    st.session_state.resume_meta = {"name": name, "pages": ex.page_count, "words": ex.word_count,
                                    "backend": ex.backend, "source": source, "key": f"{name}:{len(data)}"}
    st.session_state.analysis = None
    st.session_state.upload_error = None


def remove_resume():
    st.session_state.resume_text, st.session_state.resume_meta = "", None
    st.session_state.analysis = None
    st.session_state.uploader_n += 1


def clear_all():
    remove_resume()
    st.session_state.jd_text = ""


# ----------------------------------------------------------------------------- small UI helpers
def esc(s) -> str:
    return html.escape(str(s))


def pills(items, kind: str) -> str:
    if not items:
        return '<span class="sh-muted">None</span>'
    return "".join(f'<span class="pill pill-{kind}">{esc(t)}</span>' for t in items)


def metric_card(label, value, sub=""):
    return f'<div class="sh-card"><div class="sh-label">{esc(label)}</div><div class="sh-value">{esc(value)}</div><div class="sh-sub">{esc(sub)}</div></div>'


def info_card(title, body):
    return f'<div class="sh-card"><h4>{esc(title)}</h4><p>{esc(body)}</p></div>'


def page_header(eyebrow, title, lead=""):
    st.markdown(f'<div class="sh-eyebrow">{esc(eyebrow)}</div><h1 style="margin:0 0 .4rem 0;font-size:2.3rem">{esc(title)}</h1>'
                + (f'<div class="sh-lead" style="margin-bottom:1.4rem">{esc(lead)}</div>' if lead else ""), unsafe_allow_html=True)


def gap():
    st.markdown('<div class="sh-gap"></div>', unsafe_allow_html=True)


def need_analysis() -> bool:
    if st.session_state.analysis is None:
        st.markdown(info_card("No analysis yet", "Upload a resume and add a job description on the Resume Analyzer page, then run the analysis."), unsafe_allow_html=True)
        gap()
        st.button("Go to Resume Analyzer", type="primary", on_click=go_to, args=("Resume Analyzer",))
        return True
    return False


def plotly_layout(fig, title="", height=320):
    fig.update_layout(title={"text": title, "font": {"family": SERIF, "size": 18, "color": TEXT}, "x": 0.02},
                      paper_bgcolor=CARD, plot_bgcolor=CARD, height=height, margin=dict(l=20, r=20, t=55, b=25),
                      font={"family": SANS, "color": TEXT, "size": 12}, legend=dict(orientation="h", y=-0.15))
    return fig


# ----------------------------------------------------------------------------- charts
def gauge_chart(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "%", "font": {"family": SERIF, "size": 46}},
        gauge={"axis": {"range": [0, 100], "tickcolor": BORDER, "tickfont": {"color": MUTED}},
               "bar": {"color": SAGE, "thickness": 0.3}, "bgcolor": CARD, "borderwidth": 0,
               "steps": [{"range": [0, 35], "color": "#F4F2EC"}, {"range": [35, 55], "color": "#EFEDE5"},
                         {"range": [55, 75], "color": "#EAE8DD"}, {"range": [75, 100], "color": "#E2E6D6"}]}))
    return plotly_layout(fig, "Overall match score", 300)


def category_chart(df):
    fig = go.Figure()
    for col, color in (("Matched", SAGE), ("Partial", SAND), ("Missing", CLAY)):
        fig.add_bar(y=df["Category"], x=df[col], name=col, orientation="h", marker_color=color)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed", gridcolor=BORDER)
    fig.update_xaxes(gridcolor=BORDER, dtick=1)
    return plotly_layout(fig, "Skills by category", 320)


def terms_chart(df):
    d = df.sort_values("Contribution to cosine")
    fig = go.Figure(go.Bar(x=d["Contribution to cosine"], y=d["Term"], orientation="h", marker_color=SAGE))
    fig.update_xaxes(gridcolor=BORDER)
    return plotly_layout(fig, "Top matching keywords", 320)


def components_figure(df):
    """Matplotlib chart: points earned vs maximum points for each score component."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = list(range(len(df)))
    ax.barh(y, df["Max points"], color=BORDER, height=0.5)
    ax.barh(y, df["Points"], color=SAGE, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Component"], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    for i, (p, m) in enumerate(zip(df["Points"], df["Max points"])):
        ax.text(m + 1, i, f"{p:.1f} / {m:.0f}", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(df["Max points"]) * 1.3)
    ax.set_title("Score breakdown (points)", fontsize=13, color=TEXT, loc="left", family="serif")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- PAGE: Home
def page_home():
    st.markdown('<div class="sh-hero"><div class="sh-eyebrow">SmartHire</div>'
                '<h1>Is your resume ready for the job you want?</h1>'
                '<div class="sh-lead">Upload your resume, add the job description, and see how well you match, '
                'which skills you are missing, and a clear plan to close the gap.</div></div>', unsafe_allow_html=True)
    gap()
    c1, _ = st.columns([1, 4])
    c1.button("Analyze my resume", type="primary", on_click=go_to, args=("Resume Analyzer",))
    gap(); gap()

    cols = st.columns(3)
    cols[0].markdown(info_card("Know your match", "See how well your resume fits a specific job, with a clear score and the reasons behind it."), unsafe_allow_html=True)
    cols[1].markdown(info_card("Find your skill gaps", "Spot the skills you already show, the ones you partly cover and the ones to add, ranked by importance."), unsafe_allow_html=True)
    cols[2].markdown(info_card("Get a learning plan", "Follow a step-by-step roadmap with topics and a small project for every gap."), unsafe_allow_html=True)
    gap()
    st.markdown("### How it works")
    steps = [("01", "Upload", "Add your resume as a PDF."), ("02", "Add the job", "Paste the description of the role you want."),
             ("03", "Review", "See your match score, strengths and skill gaps."), ("04", "Improve", "Follow the roadmap and resume tips.")]
    for col, (n, t, d) in zip(st.columns(4), steps):
        col.markdown(f'<div class="sh-card"><div class="sh-step">{n}</div><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    gap()
    st.markdown('<div class="sh-muted">Your resume is analysed in memory and never stored.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- PAGE: Resume Analyzer
def page_analyzer():
    page_header("Step 1 and 2", "Resume Analyzer", "Add your resume and the job description you are targeting.")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Your resume (PDF)")
        uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"], key=f"uploader_{st.session_state.uploader_n}",
                                    help=f"Max {MAX_FILE_MB} MB, up to {MAX_PAGES} pages. Scanned image PDFs are not supported.")
        if uploaded is not None:
            meta = st.session_state.resume_meta
            if not meta or meta["key"] != f"{uploaded.name}:{uploaded.size}":
                with st.spinner("Reading PDF..."):
                    try:
                        ingest_pdf(uploaded.getvalue(), uploaded.name, "upload")
                    except PDFValidationError as exc:
                        st.session_state.resume_text, st.session_state.resume_meta = "", None
                        st.session_state.upload_error = str(exc)
                    except Exception as exc:
                        st.session_state.resume_text, st.session_state.resume_meta = "", None
                        st.session_state.upload_error = f"Unexpected error while reading the PDF: {type(exc).__name__}"
        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)

        if st.session_state.resume_meta:
            st.button("Remove resume", on_click=remove_resume)

        meta = st.session_state.resume_meta
        if meta:
            st.success(f"Loaded **{meta['name']}** - {meta['pages']} page(s), {meta['words']} words.")
            st.text_area("Resume text (review and correct if needed)", key="resume_text", height=260)
        else:
            st.markdown('<div class="sh-muted">No resume loaded yet.</div>', unsafe_allow_html=True)

    with right:
        st.markdown("#### Job description")
        st.text_area("Paste or edit the job description", key="jd_text", height=330,
                     placeholder="Paste the full job description here, including requirements and preferred skills. "
                                 "Including headings such as 'Requirements' and 'Preferred qualifications' gives the most accurate results.")
        wc = len(st.session_state.jd_text.split())
        st.markdown(f'<div class="sh-muted">{wc} words (minimum {MIN_JD_WORDS})</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sh-divider">', unsafe_allow_html=True)
    ready = bool(st.session_state.resume_meta) and wc >= MIN_JD_WORDS
    if st.button("Run analysis", type="primary", disabled=not ready):
        status = st.status("Analyzing your resume", expanded=True)
        try:
            result = run_pipeline(st.session_state.resume_text, st.session_state.jd_text, on_step=lambda m: status.write(m))
            st.session_state.analysis = result
            status.update(label="Analysis complete", state="complete", expanded=False)
        except ValueError as exc:
            status.update(label="Input problem", state="error")
            st.error(str(exc))
        except Exception as exc:
            status.update(label="Something went wrong", state="error")
            st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
    if not ready:
        st.caption("Load a resume and enter a job description to enable the analysis.")
    if st.session_state.analysis is not None:
        a = st.session_state.analysis
        st.success(f"Done. Match score: {a.score.final}% ({a.score.label}).")
        st.button("View dashboard", type="primary", on_click=go_to, args=("Analysis Dashboard",), key="to_dash")


# ----------------------------------------------------------------------------- PAGE: Dashboard
DISPLAY_NAMES = {"TF-IDF similarity": "Keyword relevance", "Required-skill coverage": "Required skills", "Preferred-skill coverage": "Preferred skills"}


def page_dashboard():
    page_header("Results", "Analysis Dashboard", "How your resume compares with this job description.")
    if need_analysis():
        return
    a = st.session_state.analysis
    g, sim, sc = a.gap, a.similarity, a.score
    relevance = min(sim.cosine / SIMILARITY_CEILING, 1.0)

    cols = st.columns(4)
    cols[0].markdown(metric_card("Match score", f"{sc.final}%", sc.label), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Keyword relevance", f"{relevance:.0%}", "Wording similar to the job post"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Required skills", "n/a" if g.required_cov is None else f"{g.required_cov:.0%}", f"{g.n_required} required skills found in the job"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Skills matched", f"{len(g.matched)} / {len(a.jd_skills)}", f"{len(g.partial)} partial, {len(g.missing)} missing"), unsafe_allow_html=True)
    gap()

    l, r = st.columns([1, 1.25], gap="large")
    with l:
        st.plotly_chart(gauge_chart(sc.final), use_container_width=True, config={"displayModeBar": False})
    with r:
        if len(sc.components):
            fig = components_figure(sc.components.assign(Component=sc.components["Component"].map(DISPLAY_NAMES)))
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    gap()
    st.markdown("### Skill comparison")
    partial_labels = [s["skill"] + " (via " + ", ".join(s["via"]) + ")" for s in g.partial]
    missing_labels = [s["skill"] + " - " + s["priority"] for s in g.missing]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="sh-card"><h4>Matched ({len(g.matched)})</h4>{pills([s["skill"] for s in g.matched], "matched")}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="sh-card"><h4>Partially matched ({len(g.partial)})</h4>{pills(partial_labels, "partial")}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="sh-card"><h4>Missing ({len(g.missing)})</h4>{pills(missing_labels, "missing")}</div>', unsafe_allow_html=True)
    if g.extra:
        gap()
        st.markdown(f'<div class="sh-card"><h4>Additional skills in your resume</h4>{pills([s["skill"] for s in g.extra], "neutral")}</div>', unsafe_allow_html=True)

    gap()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if len(g.category_table):
            st.plotly_chart(category_chart(g.category_table), use_container_width=True, config={"displayModeBar": False})
    with c2:
        if len(sim.top_terms):
            st.plotly_chart(terms_chart(sim.top_terms), use_container_width=True, config={"displayModeBar": False})

    gap()
    st.markdown("### Strengths and weaknesses")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown('<div class="sh-card"><h4>Strengths</h4>' + "".join(f'<p style="margin-bottom:.6rem">{esc(s)}</p>' for s in a.insights["strengths"]) + "</div>", unsafe_allow_html=True)
    c2.markdown('<div class="sh-card"><h4>Weaknesses</h4>' + "".join(f'<p style="margin-bottom:.6rem">{esc(s)}</p>' for s in a.insights["weaknesses"]) + "</div>", unsafe_allow_html=True)
    gap()
    st.caption("The match score estimates how well your resume fits this job description. It does not predict hiring outcomes.")


# ----------------------------------------------------------------------------- PAGE: Recommendations
def page_recommendations():
    page_header("Next steps", "Skill Gap Recommendations", "Prioritised learning topics and a phased roadmap built from your gaps.")
    if need_analysis():
        return
    a = st.session_state.analysis
    recs = build_recommendations(a.gap)
    if not recs:
        st.markdown(info_card("No skill gaps detected", "Every skill found in this job description is present in your resume."), unsafe_allow_html=True)
        return
    roadmap = build_roadmap(recs)
    total_weeks = sum(p["total_weeks"] for p in roadmap)

    hi = sum(1 for r in recs if r["priority"] == "High")
    cols = st.columns(3)
    cols[0].markdown(metric_card("Skills to work on", len(recs), f"{hi} high priority"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Roadmap length", f"~{total_weeks} weeks", "Rough beginner estimate"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Phases", len(roadmap), "Prerequisites ordered first"), unsafe_allow_html=True)
    gap()

    st.markdown("### Priority overview")
    wanted = st.multiselect("Filter by priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    df = pd.DataFrame([{"Skill": r["skill"], "Status": r["status"], "Priority": r["priority"], "In job description": r["importance"].capitalize(),
                        "Category": r["category"], "Est. weeks": r["weeks"]} for r in recs if r["priority"] in wanted])
    st.dataframe(df, use_container_width=True, hide_index=True)

    gap()
    st.markdown("### Learning roadmap")
    for ph in roadmap:
        names = ", ".join(r["skill"] for r in ph["skills"])
        st.markdown(f'<div class="sh-card"><div class="sh-eyebrow">Phase {ph["phase"]} - {esc(ph["weeks"])}</div><h4>{esc(names)}</h4></div>', unsafe_allow_html=True)
        for r in ph["skills"]:
            with st.expander(f"{r['skill']}  |  {r['priority']} priority  |  about {r['weeks']} week(s)"):
                if r["status"] == "Partial":
                    st.markdown(f"You already show related skill(s): **{', '.join(r['via'])}**. Build on them.")
                st.markdown("**Topics to learn**\n" + "\n".join(f"- {t}" for t in r["topics"]))
                st.markdown(f"**Mini project:** {r['project']}")
                if not r["curated"]:
                    st.caption("General learning plan for this skill.")
        gap()
    st.caption("Time estimates assume a beginner studying roughly 8 to 10 hours a week and are only a planning guide.")

    suggestions = build_improvement_suggestions(a, analyze_resume_structure(a.resume_text))
    st.download_button("Download full report (Markdown)", data=build_report_markdown(a, recs, roadmap, suggestions),
                       file_name="smarthire_report.md", mime="text/markdown")


# ----------------------------------------------------------------------------- PAGE: Resume Improvement
def page_improvement():
    page_header("Polish", "Resume Improvement", "Evidence-based suggestions. SmartHire never invents qualifications for you.")
    if need_analysis():
        return
    a = st.session_state.analysis
    s = analyze_resume_structure(a.resume_text)
    sugg = build_improvement_suggestions(a, s)

    cols = st.columns(4)
    cols[0].markdown(metric_card("Word count", s["word_count"], "Ideal: one focused page"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Lines with numbers", f"{s['quantified_lines']} / {s['sentence_lines']}", "Quantified impact"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Action-verb lines", f"{s['action_verb_lines']} / {s['sentence_lines']}", "Strong openings"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Suggestions", len(sugg), "Sorted by severity"), unsafe_allow_html=True)
    gap()

    c1, c2 = st.columns(2, gap="large")
    c1.markdown('<div class="sh-card"><h4>Sections detected</h4>' + "".join(
        f'<span class="pill pill-{"matched" if ok else "missing"}">{esc(n)}</span>' for n, ok in s["sections"].items()) + "</div>", unsafe_allow_html=True)
    c2.markdown('<div class="sh-card"><h4>Contact and profile links</h4>' + "".join(
        f'<span class="pill pill-{"matched" if ok else "missing"}">{esc(n)}</span>' for n, ok in s["contact"].items()) + "</div>", unsafe_allow_html=True)

    gap()
    st.markdown("### Suggestions")
    if not sugg:
        st.markdown(info_card("Looking good", "No structural issues were detected."), unsafe_allow_html=True)
    for it in sugg:
        st.markdown(f'<div class="sh-card" style="margin-bottom:.8rem"><span class="sev sev-{it["severity"]}">{it["severity"]} - {esc(it["area"])}</span>'
                    f'<h4 style="margin-top:.3rem">{esc(it["title"])}</h4><p>{esc(it["detail"])}</p></div>', unsafe_allow_html=True)

    gap()
    st.markdown("### Bullet-point templates")
    st.markdown("Fill the brackets with **real** details only.")
    for t in BULLET_TEMPLATES:
        st.markdown(f"- {t}")
    st.markdown('<div class="sh-note"><b>Integrity note:</b> Only list skills you can discuss in an interview. For gaps, build a small project first, then add it to your resume with evidence.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- main
def main():
    init_state()
    with st.sidebar:
        st.markdown('<div class="sh-brand">SmartHire</div><div class="sh-tag">Resume analyzer and skill gap detector</div>', unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
        st.markdown('<hr class="sh-divider">', unsafe_allow_html=True)
        st.caption("Privacy: your PDF is processed in memory and never saved. Text is kept only in this browser session.")
        st.button("Clear my data", on_click=clear_all)

    {"Home": page_home, "Resume Analyzer": page_analyzer, "Analysis Dashboard": page_dashboard,
     "Skill Gap Recommendations": page_recommendations, "Resume Improvement": page_improvement}[st.session_state.nav]()


main()
