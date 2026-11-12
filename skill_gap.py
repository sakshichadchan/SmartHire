"""
skill_gap.py
------------
Step 5: compare resume skills with job-description skills and run the whole pipeline.

Outputs per skill:  MATCHED  (found in resume)
                    PARTIAL  (not found, but a related/transferable skill is)
                    MISSING  (nothing related found)
Missing/partial skills get a PRIORITY (High / Medium / Low) from a transparent scoring rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ml_model import ScoreBreakdown, SimilarityResult, compute_match_score, compute_similarity
from preprocessing import preprocess
from sample_data import SAMPLE_JOBS
from skill_extractor import SKILL_DB, extract_jd_skills, extract_skills, related_skills

PARTIAL_CREDIT = 0.5
MIN_JD_WORDS = 30
MIN_RESUME_WORDS = 50


# ------------------------------------------------------------------ priority rule
def priority_of(skill: str, category: str, importance: str, jd_mentions: int, partial: bool = False) -> str:
    """
    points = 4 if required else 2
           + 0.5 per extra mention in the JD (max +1)
           + 0.5 for core technical categories, -1 for soft skills
           - 1.5 if only a partial (related) match exists
    High >= 4,  Medium >= 2,  otherwise Low.
    """
    pts = 4.0 if importance == "required" else 2.0
    pts += 0.5 * min(max(jd_mentions - 1, 0), 2)
    if category in ("Programming Languages", "ML & Data Science", "Web & Backend Frameworks", "Databases"):
        pts += 0.5
    if category == "Soft Skills":
        pts -= 1.0
    if partial:
        pts -= 1.5
    return "High" if pts >= 4 else "Medium" if pts >= 2 else "Low"


@dataclass
class SkillGapResult:
    matched: list[dict] = field(default_factory=list)
    partial: list[dict] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)       # in resume but not asked for
    required_cov: float | None = None
    preferred_cov: float | None = None
    n_required: int = 0
    n_preferred: int = 0
    category_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _coverage(matched, partial, importance: str, total: int):
    if total == 0:
        return None
    m = sum(1 for s in matched if s["importance"] == importance)
    p = sum(1 for s in partial if s["importance"] == importance)
    return (m + PARTIAL_CREDIT * p) / total


def analyze_skill_gap(resume_skills: dict, jd_skills: dict) -> SkillGapResult:
    res = SkillGapResult()
    for skill, info in jd_skills.items():
        item = {"skill": skill, "category": info["category"], "importance": info["importance"],
                "jd_mentions": info["count"], "resume_mentions": resume_skills.get(skill, {}).get("count", 0)}
        if skill in resume_skills:
            res.matched.append(item)
            continue
        via = sorted(r for r in related_skills(skill) if r in resume_skills)
        if via:
            item["via"] = via
            item["priority"] = priority_of(skill, item["category"], item["importance"], item["jd_mentions"], partial=True)
            res.partial.append(item)
        else:
            item["priority"] = priority_of(skill, item["category"], item["importance"], item["jd_mentions"])
            res.missing.append(item)

    rank = {"High": 0, "Medium": 1, "Low": 2}
    res.partial.sort(key=lambda d: (rank[d["priority"]], -d["jd_mentions"], d["skill"]))
    res.missing.sort(key=lambda d: (rank[d["priority"]], -d["jd_mentions"], d["skill"]))
    res.matched.sort(key=lambda d: (d["importance"] != "required", d["skill"]))

    res.extra = sorted(
        [{"skill": s, "category": i["category"], "resume_mentions": i["count"]} for s, i in resume_skills.items() if s not in jd_skills],
        key=lambda d: (d["category"], d["skill"]),
    )
    res.n_required = sum(1 for i in jd_skills.values() if i["importance"] == "required")
    res.n_preferred = sum(1 for i in jd_skills.values() if i["importance"] == "preferred")
    res.required_cov = _coverage(res.matched, res.partial, "required", res.n_required)
    res.preferred_cov = _coverage(res.matched, res.partial, "preferred", res.n_preferred)

    # per-category table (only categories the JD asks about)
    rows = []
    for cat in SKILL_DB:
        m = sum(1 for s in res.matched if s["category"] == cat)
        p = sum(1 for s in res.partial if s["category"] == cat)
        x = sum(1 for s in res.missing if s["category"] == cat)
        if m + p + x:
            rows.append({"Category": cat, "Matched": m, "Partial": p, "Missing": x,
                         "Coverage %": round(100 * (m + PARTIAL_CREDIT * p) / (m + p + x))})
    res.category_table = pd.DataFrame(rows, columns=["Category", "Matched", "Partial", "Missing", "Coverage %"])
    return res


# ------------------------------------------------------------------ strengths / weaknesses
def generate_insights(gap: SkillGapResult, sim: SimilarityResult, score: ScoreBreakdown) -> dict[str, list[str]]:
    """Plain-language findings, each derived directly from computed numbers (nothing invented)."""
    strengths, weaknesses = [], []
    req_matched = [s["skill"] for s in gap.matched if s["importance"] == "required"]

    if gap.n_required:
        pct = round(100 * len(req_matched) / gap.n_required)
        if req_matched:
            strengths.append(f"{len(req_matched)} of {gap.n_required} required skills are explicitly present ({pct}%): {', '.join(req_matched[:6])}{'...' if len(req_matched) > 6 else ''}.")
    for _, row in gap.category_table.iterrows():
        if row["Coverage %"] == 100 and row["Matched"] >= 1:
            strengths.append(f"Full coverage in {row['Category']} ({row['Matched']} skill(s) requested, all found).")
    if sim.cosine >= 0.25:
        strengths.append("Your resume wording overlaps well with the job description.")
    if len(gap.extra) >= 3:
        strengths.append(f"{len(gap.extra)} additional skills beyond the job description, e.g. {', '.join(s['skill'] for s in gap.extra[:5])}.")
    if not strengths:
        strengths.append("No strong overlaps were detected yet; see the recommendations to build a focused skill set.")

    high_missing = [s["skill"] for s in gap.missing if s["priority"] == "High"]
    if high_missing:
        weaknesses.append(f"{len(high_missing)} high-priority skill(s) not found in the resume: {', '.join(high_missing[:6])}{'...' if len(high_missing) > 6 else ''}.")
    for _, row in gap.category_table.iterrows():
        if row["Coverage %"] == 0:
            weaknesses.append(f"No coverage in {row['Category']} ({row['Missing']} skill(s) requested).")
    if gap.partial:
        weaknesses.append("Only related (not exact) skills found for: " + ", ".join(s["skill"] for s in gap.partial[:6]) + ".")
    if sim.cosine < 0.12:
        weaknesses.append("Your resume wording differs a lot from the job description; consider using the job's terminology where it is truthful.")
    if gap.n_required == 0:
        weaknesses.append("No required skills were detected in the job description, so coverage could not be assessed. Try a more detailed description.")
    if not weaknesses:
        weaknesses.append("No major gaps detected against this job description.")
    return {"strengths": strengths, "weaknesses": weaknesses}


# ------------------------------------------------------------------ full pipeline
@dataclass
class AnalysisResult:
    similarity: SimilarityResult
    score: ScoreBreakdown
    gap: SkillGapResult
    resume_skills: dict
    jd_skills: dict
    insights: dict
    resume_text: str
    jd_text: str


def run_pipeline(resume_text: str, jd_text: str, on_step: Callable[[str], None] | None = None) -> AnalysisResult:
    """End-to-end analysis. `on_step` lets the UI show progress messages."""
    say = on_step or (lambda _m: None)
    if len(resume_text.split()) < MIN_RESUME_WORDS:
        raise ValueError(f"Resume text is too short (minimum {MIN_RESUME_WORDS} words).")
    if len(jd_text.split()) < MIN_JD_WORDS:
        raise ValueError(f"Job description is too short (minimum {MIN_JD_WORDS} words). Paste the full description.")

    say("Reading your resume and the job description")
    preprocess(resume_text), preprocess(jd_text)

    say("Measuring how closely they match")
    sim = compute_similarity(resume_text, jd_text, reference_docs=list(SAMPLE_JOBS.values()))

    say("Finding skills")
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_jd_skills(jd_text)

    say("Comparing your skills with the job")
    gap = analyze_skill_gap(resume_skills, jd_skills)

    say("Calculating your match score")
    score = compute_match_score(sim.cosine, gap.required_cov, gap.preferred_cov)
    insights = generate_insights(gap, sim, score)
    return AnalysisResult(sim, score, gap, resume_skills, jd_skills, insights, resume_text, jd_text)
