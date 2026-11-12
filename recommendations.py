"""
recommendations.py
------------------
Step 6: turn the skill-gap result into (1) learning recommendations + roadmap and
(2) resume-improvement suggestions.

INTEGRITY RULE: nothing here invents qualifications. Suggestions tell the user to add a skill ONLY if they
genuinely have it; otherwise the skill is presented as a learning goal.
"""
from __future__ import annotations

import re
from datetime import date

from skill_gap import AnalysisResult, SkillGapResult


def _e(weeks: int, topics: list[str], project: str) -> dict:
    return {"weeks": weeks, "topics": topics, "project": project}


# Curated learning content per skill (topics -> mini project -> estimated weeks for a beginner at ~8-10 h/week).
LEARNING_DB: dict[str, dict] = {
    "Python": _e(3, ["Syntax, data types, functions", "Lists/dicts, file handling, OOP", "Virtual environments and pip"], "Command-line expense tracker"),
    "Java": _e(4, ["OOP: classes, inheritance, interfaces", "Collections and exceptions", "Maven/Gradle basics"], "Library management console app"),
    "JavaScript": _e(3, ["ES6 syntax, DOM, events", "Async: promises, fetch", "Modules and npm"], "Interactive to-do app with API calls"),
    "TypeScript": _e(2, ["Static types, interfaces, generics", "Configuring tsconfig", "Typing React props"], "Convert a small JS project to TypeScript"),
    "C++": _e(4, ["Pointers, references, STL", "OOP in C++", "Memory management"], "Contact manager using STL containers"),
    "SQL": _e(2, ["SELECT, WHERE, GROUP BY, HAVING", "JOINs and subqueries", "Window functions, indexes"], "Analyse a public sales dataset using only SQL"),
    "MySQL": _e(2, ["Install and connect (CLI + Python connector)", "Schema design and normalisation", "Joins and indexes"], "Student database with 4 related tables"),
    "PostgreSQL": _e(2, ["Install, psql basics", "Data types, constraints, JSONB", "Query planning with EXPLAIN"], "Blog backend schema + queries"),
    "MongoDB": _e(2, ["Documents and collections", "CRUD and aggregation pipeline", "Indexes and schema design"], "Notes API stored in MongoDB"),
    "Redis": _e(1, ["Key-value basics, TTL", "Caching pattern", "Pub/Sub overview"], "Cache API responses with Redis"),
    "Pandas": _e(2, ["DataFrames, indexing, filtering", "groupby, merge, pivot", "Cleaning missing data"], "Clean and analyse a Kaggle CSV"),
    "NumPy": _e(1, ["Arrays and broadcasting", "Vectorised operations", "Linear algebra basics"], "Implement linear regression with NumPy only"),
    "Scikit-learn": _e(3, ["Train/test split, pipelines", "Regression, classification, clustering", "Metrics and cross-validation"], "Churn prediction with model comparison"),
    "Machine Learning": _e(6, ["Supervised vs unsupervised learning", "Bias-variance, overfitting, regularisation", "Evaluation metrics", "Feature engineering"], "End-to-end prediction project on a public dataset"),
    "Deep Learning": _e(5, ["Neural-network fundamentals, backpropagation", "CNNs and RNNs", "Training tricks: dropout, batch norm"], "Image classifier on a small dataset"),
    "TensorFlow": _e(3, ["Tensors, Keras API", "Building and training models", "Saving and serving models"], "MNIST / fashion-MNIST classifier"),
    "PyTorch": _e(3, ["Tensors and autograd", "nn.Module and training loops", "DataLoaders"], "CNN on CIFAR-10 subset"),
    "Keras": _e(2, ["Sequential and Functional API", "Callbacks", "Transfer learning"], "Transfer-learning image classifier"),
    "NLP": _e(4, ["Text preprocessing, TF-IDF, bag-of-words", "Word embeddings", "Text classification and sentiment"], "Spam or sentiment classifier"),
    "NLTK": _e(1, ["Tokenization, stop-words, stemming", "POS tagging", "Sentiment with VADER"], "Text-cleaning toolkit"),
    "spaCy": _e(1, ["Pipelines, tokens, entities", "Named-entity recognition", "Custom components"], "Entity extractor for news articles"),
    "Hugging Face": _e(2, ["Transformers pipelines", "Tokenizers and fine-tuning", "Model hub and Spaces"], "Fine-tune a small text classifier"),
    "Matplotlib": _e(1, ["Line/bar/scatter/hist", "Subplots and styling", "Saving figures"], "Plot dashboard for a CSV"),
    "Seaborn": _e(1, ["Distribution and relational plots", "Heatmaps for correlation", "Themes"], "EDA report on a Kaggle dataset"),
    "Plotly": _e(1, ["Plotly Express", "Interactive charts", "Embedding in Streamlit"], "Interactive sales dashboard"),
    "Data Visualization": _e(2, ["Choosing the right chart", "Matplotlib/Seaborn/Plotly", "Storytelling with data"], "Visual story from a public dataset"),
    "Statistics": _e(3, ["Descriptive statistics, distributions", "Hypothesis testing, p-values, confidence intervals", "Correlation vs causation, regression"], "Run and interpret an A/B test on sample data"),
    "Power BI": _e(2, ["Importing and modelling data", "DAX basics", "Building a report"], "Sales dashboard in Power BI"),
    "Tableau": _e(2, ["Connecting data", "Calculated fields", "Dashboards and stories"], "Public-data dashboard in Tableau Public"),
    "Excel": _e(1, ["Formulas and lookups", "Pivot tables", "Charts and conditional formatting"], "Monthly budget analysis workbook"),
    "Streamlit": _e(1, ["Widgets and layout", "Caching", "Deploy on Hugging Face Spaces"], "Turn a notebook into a web app"),
    "Docker": _e(2, ["Images vs containers", "Writing a Dockerfile", "docker-compose"], "Containerise a Flask/Streamlit app"),
    "Kubernetes": _e(3, ["Pods, deployments, services", "kubectl basics", "Config maps and scaling"], "Deploy a containerised app on minikube"),
    "AWS": _e(3, ["IAM, EC2, S3 basics", "Deploying a web app", "Cost awareness and free tier"], "Host a static site and API on AWS free tier"),
    "Azure": _e(3, ["Core services overview", "App Service and storage", "Azure fundamentals (AZ-900) topics"], "Deploy a small web app on Azure"),
    "GCP": _e(3, ["Core services overview", "Cloud Run / Compute Engine", "Cloud Storage"], "Deploy a container on Cloud Run"),
    "CI/CD": _e(2, ["Pipelines concept", "GitHub Actions workflows", "Automated tests on push"], "Add a test + lint workflow to a repo"),
    "GitHub Actions": _e(1, ["Workflow YAML", "Matrix builds", "Secrets"], "Run tests automatically on every push"),
    "Jenkins": _e(2, ["Jobs and pipelines", "Jenkinsfile", "Integrations"], "Pipeline that builds and tests a project"),
    "Terraform": _e(2, ["HCL syntax", "Providers, state", "Modules"], "Provision a small cloud resource set as code"),
    "Linux": _e(2, ["Shell navigation and permissions", "Processes, packages", "Text tools: grep, sed, awk"], "Write a log-analysis script"),
    "Bash/Shell": _e(1, ["Variables, loops, conditionals", "Pipes and redirection", "Cron jobs"], "Automated backup script"),
    "Git": _e(1, ["Commit, branch, merge", "Pull requests", "Resolving conflicts"], "Contribute to a small open-source repo"),
    "GitHub": _e(1, ["Repos, issues, pull requests", "README writing", "GitHub Pages"], "Publish a well-documented project"),
    "React": _e(4, ["Components, props, state", "Hooks", "Routing and API calls"], "Weather app using a public API"),
    "Angular": _e(4, ["Components, modules", "Services and DI", "RxJS basics"], "CRUD dashboard"),
    "Vue.js": _e(3, ["Reactivity, components", "Router and state", "API calls"], "Task board"),
    "Node.js": _e(3, ["Event loop, modules", "Express routing", "Connecting to a database"], "REST API for a notes app"),
    "Express.js": _e(2, ["Routing, middleware", "Error handling", "Auth basics"], "Authenticated REST API"),
    "Django": _e(4, ["MVT architecture", "ORM and migrations", "Forms, auth, admin"], "Blog with login and comments"),
    "Flask": _e(2, ["Routes and templates", "SQLAlchemy", "Blueprints"], "Expense tracker web app"),
    "FastAPI": _e(2, ["Path/query params", "Pydantic models", "Async endpoints and docs"], "ML model served as an API"),
    "Spring Boot": _e(4, ["Dependency injection", "REST controllers", "JPA and databases"], "Student management REST API"),
    "REST APIs": _e(2, ["HTTP verbs, status codes", "Designing resources", "Auth and versioning"], "CRUD API with documentation"),
    "Microservices": _e(3, ["Service boundaries", "Communication patterns", "Containers and gateways"], "Split a monolith into two services"),
    "HTML": _e(1, ["Semantic tags", "Forms", "Accessibility"], "Personal portfolio page"),
    "CSS": _e(2, ["Box model, flexbox, grid", "Responsive design", "Variables and animations"], "Responsive landing page"),
    "Data Structures": _e(5, ["Arrays, linked lists, stacks, queues", "Trees, graphs, hash tables", "Complexity analysis"], "Solve 50 curated problems and keep notes in a repo"),
    "Algorithms": _e(5, ["Sorting and searching", "Recursion, DP, greedy", "Graph algorithms"], "Algorithm visualiser"),
    "OOP": _e(2, ["Classes, inheritance, polymorphism", "Encapsulation", "SOLID basics"], "Parking-lot system design in code"),
    "Unit Testing": _e(1, ["pytest / JUnit basics", "Fixtures and mocking", "Coverage"], "Add tests to an existing project"),
    "Agile": _e(1, ["Scrum roles and ceremonies", "User stories", "Kanban boards"], "Run a mini-sprint for a team project"),
    "Computer Networks": _e(3, ["OSI/TCP-IP models", "HTTP, DNS, TCP vs UDP", "Subnetting"], "Packet analysis with Wireshark"),
    "Operating Systems": _e(3, ["Processes and threads", "Memory management", "Scheduling and deadlocks"], "Simulate CPU scheduling algorithms"),
    "DBMS": _e(2, ["ER modelling and normalisation", "Transactions and ACID", "Indexes"], "Design a schema for a real-world scenario"),
    "Data Analysis": _e(2, ["Exploratory data analysis", "Descriptive statistics", "Presenting findings"], "EDA notebook with written conclusions"),
    "Data Cleaning": _e(1, ["Missing values, outliers", "Type conversion, deduplication", "Validation"], "Clean a messy public dataset"),
    "Feature Engineering": _e(2, ["Encoding categorical data", "Scaling, binning", "Feature selection"], "Improve a baseline model through features"),
    "Model Evaluation": _e(1, ["Cross-validation", "Precision/recall/F1/ROC-AUC", "Hyper-parameter search"], "Compare models with proper CV"),
    "R": _e(3, ["Syntax and data frames", "tidyverse", "ggplot2"], "EDA report in R Markdown"),
}

CATEGORY_FALLBACK = {
    "Soft Skills": _e(2, ["Practise through team projects and clubs", "Write short project reports / present demos", "Ask peers for feedback"], "Lead a small team project or give a tech talk"),
}
GENERIC = _e(2, ["Official documentation / getting-started guide", "A structured beginner course", "Build one small project using it"], "Small project that uses this skill end to end")

# prerequisites: learn the value(s) first if they are also in the user's gap list
PREREQUISITES = {
    "Pandas": ["Python", "NumPy"], "NumPy": ["Python"], "Scikit-learn": ["Python", "Pandas", "Machine Learning"],
    "TensorFlow": ["Python", "Deep Learning"], "PyTorch": ["Python", "Deep Learning"], "Keras": ["Python"],
    "Deep Learning": ["Machine Learning"], "Machine Learning": ["Python", "Statistics"], "NLP": ["Python", "Machine Learning"],
    "NLTK": ["Python"], "spaCy": ["Python"], "Hugging Face": ["Python", "Deep Learning"], "Matplotlib": ["Python"],
    "Seaborn": ["Python", "Matplotlib"], "Plotly": ["Python"], "Streamlit": ["Python"], "Django": ["Python"], "Flask": ["Python"],
    "FastAPI": ["Python"], "React": ["JavaScript", "HTML", "CSS"], "Angular": ["JavaScript", "TypeScript"], "Vue.js": ["JavaScript"],
    "Node.js": ["JavaScript"], "Express.js": ["Node.js"], "TypeScript": ["JavaScript"], "Spring Boot": ["Java", "OOP"],
    "Kubernetes": ["Docker"], "CI/CD": ["Git"], "GitHub Actions": ["Git", "CI/CD"], "Jenkins": ["CI/CD"], "Terraform": ["Linux"],
    "MySQL": ["SQL"], "PostgreSQL": ["SQL"], "REST APIs": ["Python"], "Microservices": ["REST APIs", "Docker"], "Data Visualization": ["Python"],
}

PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}


# ---------------------------------------------------------------- learning recommendations
def build_recommendations(gap: SkillGapResult) -> list[dict]:
    """One recommendation per missing/partial skill, sorted by priority."""
    recs = []
    for status, items in (("Missing", gap.missing), ("Partial", gap.partial)):
        for it in items:
            info = LEARNING_DB.get(it["skill"]) or CATEGORY_FALLBACK.get(it["category"]) or GENERIC
            recs.append({
                "skill": it["skill"], "category": it["category"], "status": status,
                "importance": it["importance"], "priority": it["priority"],
                "via": it.get("via", []), "weeks": info["weeks"] if status == "Missing" else max(1, info["weeks"] - 1),
                "topics": info["topics"], "project": info["project"],
                "curated": it["skill"] in LEARNING_DB,
            })
    recs.sort(key=lambda r: (PRIORITY_RANK[r["priority"]], r["status"] != "Missing", r["skill"]))
    return recs


def build_roadmap(recs: list[dict], max_weeks_per_phase: int = 4) -> list[dict]:
    """
    Order skills by priority, but pull prerequisites (that are also gaps) in front of the skills that need them
    (depth-first topological ordering), then pack them into phases of at most `max_weeks_per_phase` weeks.
    """
    by_name = {r["skill"]: r for r in recs}
    ordered, seen = [], set()

    def visit(name: str):
        if name in seen or name not in by_name:
            return
        seen.add(name)
        for pre in PREREQUISITES.get(name, []):
            visit(pre)
        ordered.append(by_name[name])

    for r in recs:
        visit(r["skill"])

    phases, current, cur_weeks, week_cursor = [], [], 0, 0
    for r in ordered:
        if current and cur_weeks + r["weeks"] > max_weeks_per_phase:
            phases.append((current, cur_weeks))
            current, cur_weeks = [], 0
        current.append(r)
        cur_weeks += r["weeks"]
    if current:
        phases.append((current, cur_weeks))

    roadmap = []
    for i, (items, weeks) in enumerate(phases, 1):
        start, end = week_cursor + 1, week_cursor + weeks
        week_cursor = end
        roadmap.append({"phase": i, "weeks": f"Week {start}" if start == end else f"Weeks {start}-{end}",
                        "skills": items, "total_weeks": weeks})
    return roadmap


# ---------------------------------------------------------------- resume structure analysis
SECTION_PATTERNS = {
    "Summary / Objective": r"summary|objective|profile|about me",
    "Education": r"education|academic",
    "Skills": r"skills|technologies|tech stack",
    "Projects": r"projects?",
    "Experience / Internship": r"experience|internships?|employment|work history",
    "Certifications": r"certifications?|courses",
    "Achievements": r"achievements?|awards?|honou?rs|accomplishments",
}
ACTION_VERBS = set("""built developed designed implemented created led managed analysed analyzed automated optimised optimized
deployed improved reduced increased launched trained evaluated collaborated presented engineered integrated tested wrote
conducted applied performed organised organized mentored delivered streamlined achieved compared visualised visualized""".split())
WEAK_PHRASES = ["responsible for", "worked on", "helped with", "duties included", "hard working", "hard-working", "team player", "various tasks"]


def analyze_resume_structure(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sections = {}
    for name, pat in SECTION_PATTERNS.items():
        # a heading = a SHORT line that is not a sentence and contains the section keyword
        sections[name] = any(len(l) <= 45 and not l.endswith(".") and re.search(rf"\b(?:{pat})\b", l, re.I) for l in lines)
    contact = {
        "Email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)),
        "Phone": bool(re.search(r"\+?\d[\d\s\-()]{8,}\d", text)),
        "LinkedIn": bool(re.search(r"linkedin\.com", text, re.I)),
        "GitHub": bool(re.search(r"github\.com", text, re.I)),
    }
    sentence_lines = [l for l in lines if len(l.split()) >= 6]
    quantified = [l for l in sentence_lines if re.search(r"\d+\s*(%|\+|x\b|k\b)|\b\d{2,}\b", l)]
    def starts_with_verb(line: str) -> bool:
        # check the line start AND the text after a "Project title - " prefix
        for chunk in (line, line.split(" - ", 1)[-1]):
            first = re.sub(r"^[\-\*\s]+", "", chunk).split(" ")[0].lower().strip(",.:")
            if first in ACTION_VERBS:
                return True
        return False

    verb_lines = sum(1 for l in sentence_lines if starts_with_verb(l))
    weak = sorted({p for p in WEAK_PHRASES if p in text.lower()})
    return {"sections": sections, "contact": contact, "word_count": len(text.split()),
            "sentence_lines": len(sentence_lines), "quantified_lines": len(quantified),
            "action_verb_lines": verb_lines, "weak_phrases": weak}


def build_improvement_suggestions(result: AnalysisResult, structure: dict) -> list[dict]:
    """Honest, evidence-based suggestions. Each carries a severity: High / Medium / Low."""
    gap, out = result.gap, []

    def add(sev, area, title, detail):
        out.append({"severity": sev, "area": area, "title": title, "detail": detail})

    # --- skills & keywords (never invent)
    for it in gap.partial[:5]:
        add("Medium", "Keywords", f"Be explicit about {it['skill']} - only if true",
            f"Your resume shows related skill(s) ({', '.join(it['via'])}). If you have genuinely used {it['skill']} in a project, course or internship, "
            f"name it explicitly and say where. If not, treat it as a learning goal rather than adding it.")
    for it in [m for m in gap.matched if m["importance"] == "required" and m["resume_mentions"] == 1][:4]:
        add("Low", "Evidence", f"Back up {it['skill']} with context",
            f"{it['skill']} appears only once. Mention it inside a project or experience bullet showing what you built with it, not just in the skills list.")
    high_missing = [m["skill"] for m in gap.missing if m["priority"] == "High"]
    if high_missing:
        add("High", "Skill gap", "Do not list skills you do not have",
            f"High-priority gaps: {', '.join(high_missing[:6])}. Close them through learning plus a small project (see the roadmap), then add them with evidence.")

    # --- structure
    missing_sections = [n for n, ok in structure["sections"].items() if not ok and n in ("Summary / Objective", "Education", "Skills", "Projects")]
    if missing_sections:
        add("Medium", "Structure", "Add standard section headings",
            "These headings were not detected: " + ", ".join(missing_sections) + ". ATS and recruiters scan for clear headings.")
    if not structure["sections"]["Experience / Internship"]:
        add("Low", "Structure", "No experience/internship section detected",
            "If you have internships, freelance work, open-source or club roles, add them. If not, strong projects can take that space.")
    if not structure["contact"]["GitHub"]:
        add("Medium", "Profile", "Add a GitHub link", "For technical roles, a GitHub profile with documented projects is strong evidence.")
    if not structure["contact"]["LinkedIn"]:
        add("Low", "Profile", "Add a LinkedIn link", "Recruiters commonly look for one.")
    if not (structure["contact"]["Email"] and structure["contact"]["Phone"]):
        add("High", "Profile", "Contact details incomplete", "An email address and a phone number should both be visible.")

    # --- writing quality
    sl, ql = max(structure["sentence_lines"], 1), structure["quantified_lines"]
    if ql / sl < 0.25:
        add("High", "Impact", "Add measurable results",
            f"Only {ql} of {structure['sentence_lines']} descriptive lines contain numbers. Add real figures you can verify (dataset size, accuracy, users, time saved).")
    if structure["action_verb_lines"] / sl < 0.4:
        add("Medium", "Writing", "Start bullets with strong action verbs", "e.g. Built, Implemented, Automated, Evaluated, Deployed.")
    if structure["weak_phrases"]:
        add("Low", "Writing", "Replace weak phrases", "Found: " + ", ".join(structure["weak_phrases"]) + ". Say what you did and what changed.")
    wc = structure["word_count"]
    if wc < 200:
        add("Medium", "Length", "Resume looks thin", f"Only {wc} words. Add project detail: problem, approach, tools, result.")
    elif wc > 900:
        add("Medium", "Length", "Resume looks long", f"{wc} words. For a student resume, keep to one page and cut the least relevant items.")
    if result.similarity.cosine < 0.1:
        add("Medium", "Relevance", "Tailor wording to the role", "Overall wording differs a lot from the job post. Re-order projects so the most relevant one comes first and reuse the job's terminology where truthful.")

    rank = {"High": 0, "Medium": 1, "Low": 2}
    out.sort(key=lambda d: rank[d["severity"]])
    return out


BULLET_TEMPLATES = [
    "Built [what you made] using [tools you actually used] to [problem solved], achieving [real, verifiable result].",
    "Analysed [dataset, size] with [library] and found [insight]; presented findings to [audience].",
    "Implemented [feature/model] that improved [metric] from [A] to [B] (measured on [test set / users]).",
    "Automated [manual task] with [language/tool], saving about [time] per [period].",
]


# ---------------------------------------------------------------- downloadable report
def build_report_markdown(result: AnalysisResult, recs: list[dict], roadmap: list[dict], suggestions: list[dict]) -> str:
    g = result.gap
    L = [f"# SmartHire Analysis Report", f"_Generated {date.today().isoformat()}. Match score is a text/skill similarity indicator, not a hiring probability._", "",
         f"**Match score:** {result.score.final}% ({result.score.label})  ",
         f"**TF-IDF cosine similarity:** {result.similarity.cosine:.3f}  ",
         f"**Required-skill coverage:** {'n/a' if g.required_cov is None else f'{g.required_cov:.0%}'}", "",
         "## Matched skills", ", ".join(s["skill"] for s in g.matched) or "None", "",
         "## Partially matched skills", ", ".join(f"{s['skill']} (via {', '.join(s['via'])})" for s in g.partial) or "None", "",
         "## Missing skills", ", ".join(f"{s['skill']} [{s['priority']}]" for s in g.missing) or "None", "",
         "## Strengths"] + [f"- {s}" for s in result.insights["strengths"]] + ["", "## Weaknesses"] + [f"- {s}" for s in result.insights["weaknesses"]] + ["", "## Learning roadmap"]
    for ph in roadmap:
        L.append(f"### Phase {ph['phase']} ({ph['weeks']})")
        for r in ph["skills"]:
            L.append(f"- **{r['skill']}** [{r['priority']}]: {'; '.join(r['topics'])}. Project: {r['project']}.")
    L += ["", "## Resume improvement suggestions"] + [f"- [{s['severity']}] **{s['title']}** - {s['detail']}" for s in suggestions]
    return "\n".join(L)
