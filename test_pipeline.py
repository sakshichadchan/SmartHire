"""Run with:  python -m pytest -q   (from the project root)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ml_model import compute_match_score, compute_similarity
from pdf_extractor import PDFValidationError, extract_text_from_pdf
from preprocessing import clean_text, preprocess
from recommendations import build_recommendations, build_roadmap
from sample_data import SAMPLE_JOBS, load_sample_resume_bytes
from skill_extractor import extract_jd_skills, extract_skills
from skill_gap import run_pipeline


def test_preprocessing_keeps_tech_tokens_and_removes_stopwords():
    assert "cpp" in clean_text("I know C++ and Node.js").split()
    p = preprocess("The students are building machine learning models")
    assert "the" not in p.filtered and "are" not in p.filtered
    assert p.processed


def test_skill_extraction_boundaries():
    found = extract_skills("Worked with MySQL, JavaScript, Node.js and C++. Able to excel in teams.")
    assert "MySQL" in found and "JavaScript" in found and "C++" in found
    assert "SQL" not in found          # 'mysql' must not trigger 'sql'
    assert "Java" not in found         # 'javascript' must not trigger 'java'
    assert "Excel" not in found        # the verb 'excel' is not the tool


def test_required_vs_preferred_detection():
    jd = extract_jd_skills("Requirements:\n- Python and SQL\n\nPreferred qualifications:\n- Docker\n- Knowledge of AWS is a plus")
    assert jd["Python"]["importance"] == "required"
    assert jd["Docker"]["importance"] == "preferred" and jd["AWS"]["importance"] == "preferred"


def test_cosine_is_higher_for_related_text():
    resume = "Python developer with pandas numpy scikit-learn machine learning projects and SQL databases " * 3
    related = "Looking for python developer with pandas scikit-learn machine learning and SQL " * 3
    unrelated = "Hospital nurse caring for patients administering medicine and supporting doctors " * 3
    refs = list(SAMPLE_JOBS.values())
    assert compute_similarity(resume, related, refs).cosine > compute_similarity(resume, unrelated, refs).cosine


def test_score_formula_and_weight_redistribution():
    full = compute_match_score(0.5, 1.0, 1.0)
    assert full.final == 100.0
    no_pref = compute_match_score(0.25, 0.5, None)       # weights 0.25 / 0.60 re-normalised
    expected = 100 * (0.25 / 0.85 * 0.5 + 0.60 / 0.85 * 0.5)
    assert abs(no_pref.final - round(expected, 1)) < 0.11


def test_pdf_validation_errors():
    for bad in (b"", b"not a pdf", b"%PDF-1.4 garbage"):
        with pytest.raises(PDFValidationError):
            extract_text_from_pdf(bad)


def test_end_to_end_on_sample_data():
    text = extract_text_from_pdf(load_sample_resume_bytes()).text
    res = run_pipeline(text, SAMPLE_JOBS["Machine Learning Intern"])
    assert 0 <= res.score.final <= 100
    assert {"Python", "Pandas", "Scikit-learn"} <= {m["skill"] for m in res.gap.matched}
    assert "TensorFlow" in {m["skill"] for m in res.gap.missing}
    recs = build_recommendations(res.gap)
    assert recs and build_roadmap(recs)
    # a different job must give a different score (no hard-coded results)
    other = run_pipeline(text, SAMPLE_JOBS["DevOps / Cloud Engineer (Entry Level)"])
    assert other.score.final < res.score.final


def test_short_inputs_rejected():
    with pytest.raises(ValueError):
        run_pipeline("too short", SAMPLE_JOBS["Data Analyst"])
