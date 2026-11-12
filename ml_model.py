"""
ml_model.py
-----------
Step 4: the numerical / ML part of SmartHire.

(a) TF-IDF vectorisation  -> turns resume and job description into numeric vectors.
(b) Cosine similarity     -> measures how close the two vectors point in the same direction.
(c) Weighted match score  -> combines similarity with skill coverage into one transparent number.

IMPORTANT (honesty note): this is UNSUPERVISED text similarity. There is no training data, no labels,
and no classifier. The score is a *document-match indicator*, NOT a probability of being hired.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import build_stem_lookup, preprocess

# ------------------------------------------------------------------ score formula constants
# Raw cosine similarity between a resume and a job post is rarely above ~0.5, because the two documents
# are written in different styles. We therefore rescale:  S_norm = min(cosine / 0.50, 1).
SIMILARITY_CEILING = 0.50

# Base weights (they are re-normalised if a component does not exist, e.g. JD with no preferred skills).
BASE_WEIGHTS = {
    "TF-IDF similarity": 0.25,
    "Required-skill coverage": 0.60,
    "Preferred-skill coverage": 0.15,
}

FORMULA_TEXT = (
    "Match Score = 100 x [ 0.25 x S_norm  +  0.60 x C_required  +  0.15 x C_preferred ]\n"
    "  S_norm      = min( cosine_similarity / 0.50 , 1 )\n"
    "  C_required  = ( matched + 0.5 x partial ) / total required skills\n"
    "  C_preferred = ( matched + 0.5 x partial ) / total preferred skills\n"
    "If the job description has no preferred (or no required) skills, the remaining weights are re-scaled to sum to 1."
)


# ------------------------------------------------------------------ (a)+(b) TF-IDF and cosine
@dataclass
class SimilarityResult:
    cosine: float
    top_terms: pd.DataFrame        # terms contributing most to the similarity
    vocab_size: int
    n_documents: int               # documents used to learn IDF
    resume_tokens: int
    jd_tokens: int


def compute_similarity(resume_text: str, jd_text: str, reference_docs: list[str] | None = None) -> SimilarityResult:
    """
    Fit a TF-IDF vectoriser and return cosine(resume, job description).

    The vectoriser is fitted on [resume, JD] plus a few *reference* job descriptions. With only two
    documents, IDF would be almost meaningless (every word appears in 1 or 2 documents), so the reference
    corpus gives common words ("team", "develop") a lower weight and distinctive words a higher weight.
    """
    r = preprocess(resume_text)
    j = preprocess(jd_text)
    if not r.processed or not j.processed:
        raise ValueError("Resume or job description has no usable text after preprocessing.")

    refs = []
    for doc in reference_docs or []:
        p = preprocess(doc).processed
        if p and p != j.processed:          # do not include the JD twice
            refs.append(p)

    docs = [r.processed, j.processed] + refs
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams, so 'machin learn' counts as one feature
        sublinear_tf=True,      # use 1 + log(tf) so repeating a word 20 times does not dominate
        token_pattern=r"\S+",   # text is already tokenised by preprocessing.py
        lowercase=False,
    )
    X = vectorizer.fit_transform(docs)                          # rows are L2-normalised
    cosine = float(cosine_similarity(X[0], X[1])[0, 0])

    # Because rows are unit length, cosine = sum over terms of (w_resume * w_jd).
    # So each term's product is exactly its contribution to the similarity -> explainable.
    contrib = np.asarray(X[0].multiply(X[1]).todense()).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(contrib)[::-1]
    lookup = build_stem_lookup(resume_text + "\n" + jd_text)

    rows = []
    for idx in order[:12]:
        if contrib[idx] <= 0:
            break
        readable = " ".join(lookup.get(tok, tok) for tok in terms[idx].split())
        rows.append({"Term": readable, "Contribution to cosine": round(float(contrib[idx]), 4)})
    top = pd.DataFrame(rows, columns=["Term", "Contribution to cosine"])

    return SimilarityResult(cosine, top, len(terms), len(docs), len(r.filtered), len(j.filtered))


# ------------------------------------------------------------------ (c) weighted score
@dataclass
class ScoreBreakdown:
    final: float
    label: str
    components: pd.DataFrame       # one row per component: value, weights, points
    notes: list[str]


def score_label(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 55:
        return "Good match"
    if score >= 35:
        return "Moderate match"
    return "Low match"


def compute_match_score(cosine: float, required_cov: float | None, preferred_cov: float | None) -> ScoreBreakdown:
    """
    required_cov / preferred_cov are fractions in [0,1], or None if the JD has no such skills.
    Implements the formula shown in FORMULA_TEXT.
    """
    s_norm = min(cosine / SIMILARITY_CEILING, 1.0)
    values = {
        "TF-IDF similarity": s_norm,
        "Required-skill coverage": required_cov,
        "Preferred-skill coverage": preferred_cov,
    }
    notes = []
    available = {k: v for k, v in values.items() if v is not None}
    for k, v in values.items():
        if v is None:
            notes.append(f"No {k.split()[0].lower()} skills were detected in the job description, so that weight was redistributed.")
    total_w = sum(BASE_WEIGHTS[k] for k in available)

    rows, final = [], 0.0
    for name, value in available.items():
        w_eff = BASE_WEIGHTS[name] / total_w
        points = 100 * w_eff * value
        final += points
        rows.append({
            "Component": name,
            "Value (0-1)": round(value, 3),
            "Base weight": BASE_WEIGHTS[name],
            "Effective weight": round(w_eff, 3),
            "Points": round(points, 1),
            "Max points": round(100 * w_eff, 1),
        })
    if required_cov is None and preferred_cov is None:
        notes.append("No dictionary skills were found in the job description; the score relies on text similarity only. Add more concrete skills to the JD.")
    final = round(final, 1)
    return ScoreBreakdown(final, score_label(final), pd.DataFrame(rows), notes)
