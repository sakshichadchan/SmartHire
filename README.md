---
title: SmartHire Resume Analyzer
emoji: 📄
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# SmartHire - AI-Powered Resume Analyzer and Skill Gap Detector

A web app that compares a **PDF resume** with a **job description** and reports a match score, matched / partially matched / missing skills,
strengths and weaknesses, a prioritised learning roadmap and honest resume-improvement tips.

Built with **Python, Scikit-learn (TF-IDF, cosine similarity), NLTK, regex, Pandas, NumPy, PyMuPDF, Streamlit, Plotly and Matplotlib**.
No paid API is needed: the whole ML pipeline runs locally.

> **What this is (and is not).** SmartHire uses *unsupervised* text similarity plus *rule-based* skill extraction.
> There is no trained classifier and no labelled hiring data. The match score measures **document and skill overlap**; it is **not a probability of being hired**.

---

## 1. Architecture

```
 PDF upload ──► pdf_extractor.py ──► raw text
                                        │
 Job description (typed / sample) ──────┤
                                        ▼
                               preprocessing.py
              lowercase → clean → tokenize → stop-words → stemming
                          │                           │
                          ▼                           ▼
                    ml_model.py                skill_extractor.py
        TF-IDF vectors → cosine similarity   dictionary + regex skill finding
                          │                   (required vs preferred in JD)
                          └──────────┬────────────────┘
                                     ▼
                               skill_gap.py
              matched / partial / missing + priority + insights
                                     ▼
                  ml_model.compute_match_score  (weighted formula)
                                     ▼
                           recommendations.py
        learning topics + roadmap + resume suggestions + report
                                     ▼
                      app.py (Streamlit, 5 pages)
```

## 2. ML / NLP workflow

1. **Extraction** - PyMuPDF reads the PDF in memory (pdfplumber is the fallback). Validation: header check, size ≤ 5 MB, ≤ 10 pages, not password-protected, enough text (scanned images are rejected).
2. **Pre-processing** - lowercase; remove URLs/e-mails/phone numbers; protect tokens like `C++`, `C#`, `.NET`, `Node.js`; remove punctuation; regex tokenizer; stop-word removal (NLTK list + resume boilerplate words such as "experience", "required"); Porter stemming.
3. **TF-IDF** - `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`. It is fitted on the resume, the job description and the six sample job descriptions (a small reference corpus so that IDF is meaningful).
4. **Cosine similarity** - because TF-IDF rows are L2-normalised, cosine = sum of (resume weight × job weight) per term, so the app can show which terms drive the similarity.
5. **Skill extraction** - a dictionary of ~100 skills in 8 categories (languages, frameworks, ML/data, databases, cloud/DevOps, tools, CS fundamentals, soft skills) with aliases, matched by boundary-safe regex (`sql` ≠ `mysql`, `java` ≠ `javascript`, `c` ≠ `c++`).
6. **Required vs preferred** - the JD is read line by line; headings such as *Preferred qualifications* and phrases such as *is a plus* mark skills as preferred.
7. **Gap analysis** - each JD skill becomes **Matched** (found), **Partial** (a related skill is found, e.g. PyTorch ↔ TensorFlow, React ↔ JavaScript) or **Missing**.
8. **Priority** - High / Medium / Low from a transparent point rule (see `priority_of` in `skill_gap.py`).
9. **Score, recommendations, suggestions** - see below.

## 3. The score formula

```
Match Score = 100 × [ 0.25 × S_norm  +  0.60 × C_required  +  0.15 × C_preferred ]

S_norm       = min( cosine_similarity / 0.50 , 1 )
C_required   = ( matched + 0.5 × partial ) / total required skills
C_preferred  = ( matched + 0.5 × partial ) / total preferred skills
```

* Required skills carry the most weight because they are the real filters in most job posts.
* Cosine between two different documents rarely exceeds ~0.5, so it is divided by 0.5 (the ceiling) and capped at 1. This constant is a documented design choice, not something learned.
* If the JD contains no preferred (or required) skills, the other weights are re-scaled so they still sum to 1.
* Bands: ≥75 Strong, 55-75 Good, 35-55 Moderate, <35 Low.

## 4. Project structure

```
smarthire/
├── app.py               Streamlit UI (5 pages)
├── pdf_extractor.py     PDF validation + text extraction
├── preprocessing.py     cleaning, tokenizing, stop-words, stemming
├── skill_extractor.py   skill dictionary, regex extraction, required/preferred detection
├── ml_model.py          TF-IDF, cosine similarity, weighted score
├── skill_gap.py         matched/partial/missing, priority, insights, run_pipeline()
├── recommendations.py   learning DB, roadmap, resume checks, report
├── sample_data.py       sample job descriptions (+ IDF reference corpus)
├── sample_data/sample_resume.pdf   fictional sample resume
├── tests/test_pipeline.py          unit + end-to-end tests
├── .streamlit/config.toml          theme
├── requirements.txt
└── README.md
```

## 5. Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
python -m pytest -q                # optional: run the tests (pip install pytest)
```

To test quickly, upload `sample_data/sample_resume.pdf` and paste any job description from `sample_data.py`.

## 6. Deploy on Hugging Face Spaces

1. Create an account at huggingface.co, then **New Space** → name `smarthire` → SDK **Streamlit** → hardware **CPU basic (free)** → Public.
2. Upload every file in this folder (keep the folder layout, including `sample_data/` and `.streamlit/`), or push with git:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/smarthire
   cp -r <this-project>/* <this-project>/.streamlit smarthire/   # copy all files
   cd smarthire && git add . && git commit -m "SmartHire" && git push
   ```
3. The YAML block at the top of this README tells Spaces to run `app.py` with Streamlit. The Space builds automatically (2-4 minutes) and gives you a public URL.
4. Put the Space URL and the GitHub link in your resume. If the build fails, open the **Logs** tab; the usual cause is a typo in `requirements.txt`.

NLTK stop-words are downloaded on first run; if that is not possible the app silently uses a built-in list.

## 7. Algorithms in simple language (viva notes)

**TF-IDF.** *Term Frequency* counts how often a word appears in a document; *Inverse Document Frequency* lowers the weight of words that appear in many documents ("team") and raises the weight of rare, distinctive ones ("pytorch"). Multiplying them gives each word an importance score, so a document becomes a vector of numbers. `sublinear_tf` uses 1+log(tf) so repeating a word many times does not dominate.

**Cosine similarity.** Treat each document as an arrow in word-space. Cosine of the angle between two arrows = how similar their direction is (1 = same topic, 0 = nothing in common). It ignores document length, which is why it suits resumes vs job posts.

**Stop-words and stemming.** Stop-words ("the", "is") carry no meaning for matching, so they are removed. Stemming cuts words to a root (*developing*, *developed* → *develop*) so different forms match.

**Regex skill extraction.** Each skill has aliases (e.g. `scikit-learn`, `sklearn`). Regex with word-boundary-style rules finds them without false hits such as `java` inside `javascript`.

**Partial match.** Skills are grouped into families (TensorFlow/PyTorch/Keras; React/Angular/Vue; AWS/Azure/GCP ...). If a required skill is missing but a family member is present, it counts as half a skill because the knowledge is transferable.

**Why not a classifier?** Training one needs thousands of labelled (resume, hired/not-hired) pairs, which are private and biased. Similarity + coverage is explainable, needs no labels, and every number can be traced.

### Likely viva questions

| Question | Short answer |
|---|---|
| Is this supervised or unsupervised? | Unsupervised / rule-based. No labels, no training. |
| Does the score predict hiring? | No. It measures overlap between two documents. |
| Why fit TF-IDF with extra documents? | With 2 documents IDF is meaningless; the reference job posts give realistic word rarity. |
| Why divide cosine by 0.5? | Real resume-vs-JD cosine seldom exceeds 0.5; scaling makes the component use the 0-1 range. It is a documented heuristic. |
| Why bigrams? | "machine learning" as one feature is more informative than two separate words. |
| How do you avoid false skill matches? | Boundary-safe regex plus overrides (e.g. the verb "excel" vs the tool Excel). |
| What are the limitations? | Skills outside the dictionary are invisible; "A or B" alternatives are not parsed; synonyms beyond the alias list are missed; scanned PDFs are unsupported; English only. |
| How would you improve it? | Sentence-embedding similarity (e.g. SBERT), a larger/ESCO skill taxonomy, NER-based skill extraction, labelled data for a supervised ranker. |
| How is privacy handled? | The PDF is read from memory and never saved; only extracted text lives in the browser session; a "Clear my data" button resets it. |

## 8. Limitations (be upfront about these)

* The skill dictionary is finite and hand-written; extend `SKILL_DB` in `skill_extractor.py` to add skills.
* Required/preferred detection depends on headings and phrases in the JD; unstructured JDs default to "required".
* Related-skill groups and learning-time estimates are judgement-based heuristics.
* Resume structure checks (sections, quantified lines) are simple heuristics and can misjudge unusual layouts.
* Sample job descriptions and the sample resume are fictional and were written for this project.

## 9. Ideas for extension

Sentence-BERT embeddings as a second similarity signal; multiple resumes ranked against one job; export of the report as PDF; ESCO/O*NET skill taxonomy; unit tests in CI with GitHub Actions.
