"""
skill_extractor.py
------------------
Step 3: rule-based skill extraction using a structured technical skill dictionary.

* SKILL_DB   : category -> canonical skill -> list of aliases (lower-case).
* extract_skills()     : finds skills in any text (resume) and counts mentions.
* extract_jd_skills()  : same, but also labels each skill REQUIRED or PREFERRED by looking at
                         section headings ("Preferred qualifications") and phrases ("is a plus").
* related_skills()     : transferable-skill groups used to detect PARTIAL matches
                         (e.g. job needs PyTorch, resume has TensorFlow).

This is a dictionary + regex approach (a classic information-extraction technique),
not a trained model. Its limits are documented in the README.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- skill database
SKILL_DB: dict[str, dict[str, list[str]]] = {
    "Programming Languages": {
        "Python": ["python", "python3"],
        "Java": ["java"],
        "JavaScript": ["javascript", "ecmascript", "es6"],
        "TypeScript": ["typescript"],
        "C": ["c programming", "c language", "embedded c"],
        "C++": ["c++", "cpp"],
        "C#": ["c#", "csharp"],
        "Go": ["golang", "go language"],
        "R": ["r programming", "r language", "rstudio"],
        "Kotlin": ["kotlin"],
        "PHP": ["php"],
        "Swift": ["swift"],
        "Rust": ["rust"],
        "MATLAB": ["matlab"],
        "Scala": ["scala"],
        "Bash/Shell": ["bash", "shell scripting", "shell script", "shell scripts"],
    },
    "Web & Backend Frameworks": {
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3"],
        "React": ["react", "reactjs", "react.js"],
        "Angular": ["angular", "angularjs"],
        "Vue.js": ["vue", "vuejs", "vue.js"],
        "Node.js": ["node.js", "nodejs", "node js"],
        "Express.js": ["express.js", "expressjs"],
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi"],
        "Spring Boot": ["spring boot", "springboot", "spring framework"],
        "Bootstrap": ["bootstrap"],
        "Tailwind CSS": ["tailwind", "tailwind css"],
        "REST APIs": ["rest api", "rest apis", "restful api", "restful apis", "restful", "rest endpoints"],
        "Microservices": ["microservices", "microservice"],
    },
    "ML & Data Science": {
        "Machine Learning": ["machine learning", "ml"],
        "Deep Learning": ["deep learning", "neural networks", "neural network"],
        "NLP": ["nlp", "natural language processing", "text mining"],
        "Computer Vision": ["computer vision", "opencv", "image processing"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn", "scikitlearn"],
        "TensorFlow": ["tensorflow"],
        "Keras": ["keras"],
        "PyTorch": ["pytorch"],
        "SciPy": ["scipy"],
        "NLTK": ["nltk"],
        "spaCy": ["spacy"],
        "Hugging Face": ["hugging face", "huggingface"],
        "Matplotlib": ["matplotlib"],
        "Seaborn": ["seaborn"],
        "Plotly": ["plotly"],
        "Statistics": ["statistics", "statistical analysis", "statistical modeling", "statistical modelling"],
        "Data Visualization": ["data visualization", "data visualisation", "data visualizations"],
        "Data Analysis": ["data analysis", "data analytics"],
        "Data Cleaning": ["data cleaning", "data preprocessing", "data pre-processing", "data wrangling", "data cleansing"],
        "Feature Engineering": ["feature engineering"],
        "Model Evaluation": ["model evaluation", "cross-validation", "cross validation", "hyperparameter tuning"],
    },
    "Databases": {
        "SQL": ["sql"],
        "MySQL": ["mysql"],
        "PostgreSQL": ["postgresql", "postgres"],
        "MongoDB": ["mongodb", "mongo db"],
        "SQLite": ["sqlite"],
        "Oracle DB": ["oracle db", "oracle database"],
        "SQL Server": ["sql server", "mssql"],
        "Redis": ["redis"],
        "NoSQL": ["nosql"],
        "Firebase": ["firebase"],
        "DBMS": ["dbms", "rdbms", "relational database", "relational databases"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure", "microsoft azure"],
        "GCP": ["gcp", "google cloud"],
        "Docker": ["docker", "containerization", "containerisation"],
        "Kubernetes": ["kubernetes", "k8s"],
        "CI/CD": ["ci/cd", "ci cd", "cicd", "continuous integration"],
        "Jenkins": ["jenkins"],
        "GitHub Actions": ["github actions"],
        "Terraform": ["terraform"],
        "Linux": ["linux", "unix"],
    },
    "Tools & Platforms": {
        "Git": ["git"],
        "GitHub": ["github"],
        "GitLab": ["gitlab"],
        "Jira": ["jira"],
        "Postman": ["postman"],
        "Jupyter": ["jupyter", "jupyter notebook", "google colab"],
        "Streamlit": ["streamlit"],
        "Power BI": ["power bi", "powerbi"],
        "Tableau": ["tableau"],
        "Excel": ["microsoft excel", "ms excel", "advanced excel", "excel spreadsheets"],  # bare 'Excel' via override
        "Agile": ["agile", "scrum"],
        "Unit Testing": ["unit testing", "unit tests", "pytest", "junit", "unittest"],
        "Figma": ["figma"],
    },
    "CS Fundamentals": {
        "Data Structures": ["data structures", "data structure"],
        "Algorithms": ["algorithms", "algorithm design"],
        "OOP": ["oop", "oops", "object oriented", "object-oriented"],
        "Operating Systems": ["operating systems", "operating system"],
        "Computer Networks": ["computer networks", "computer networking", "networking"],
        "System Design": ["system design"],
        "Design Patterns": ["design patterns"],
    },
    "Soft Skills": {
        "Communication": ["communication", "communicate", "communicator", "presentation skills"],
        "Teamwork": ["teamwork", "team player", "collaboration", "collaborative", "collaborated", "cross-functional", "work in a team"],
        "Leadership": ["leadership", "team lead", "led a team", "mentoring"],
        "Problem Solving": ["problem solving", "problem-solving", "troubleshooting"],
        "Analytical Thinking": ["analytical", "analytical thinking", "analytical skills"],
        "Time Management": ["time management", "meeting deadlines"],
        "Adaptability": ["adaptability", "adaptable", "fast learner", "quick learner"],
        "Critical Thinking": ["critical thinking"],
        "Attention to Detail": ["attention to detail", "detail-oriented", "detail oriented"],
    },
}

# Raw-regex overrides for words that are also normal English ("excel", "go").
REGEX_OVERRIDES = {
    "Excel": r"(?:microsoft excel|ms excel|advanced excel|excel spreadsheets?|(?<!to )(?<!will )(?<!can )(?<!and )excel(?!\s+(?:in|at|as)\b)(?!ling|lent|s\b))",
}

# canonical skill -> category
SKILL_CATEGORY = {skill: cat for cat, skills in SKILL_DB.items() for skill in skills}

# ---------------------------------------------------------------- regex compilation
_LEFT = r"(?<![A-Za-z0-9_+#.])"       # not glued to a previous word char (so 'sql' != 'mysql', 'js' != 'node.js')
_RIGHT = r"(?![A-Za-z0-9_+#])"        # not glued to a following char (so 'java' != 'javascript', 'c' != 'c++')


def _alias_regex(alias: str) -> str:
    parts = [re.escape(p) for p in alias.split(" ")]
    return r"[\s\-]+".join(parts)     # 'machine learning' also matches 'machine-learning'


def _compile() -> dict[str, re.Pattern]:
    compiled = {}
    for cat, skills in SKILL_DB.items():
        for skill, aliases in skills.items():
            if skill in REGEX_OVERRIDES:
                body = REGEX_OVERRIDES[skill]
            else:
                body = "|".join(_alias_regex(a) for a in sorted(set(aliases), key=len, reverse=True))
            compiled[skill] = re.compile(_LEFT + "(?:" + body + ")" + _RIGHT, re.IGNORECASE)
    return compiled


_PATTERNS = _compile()

# ---------------------------------------------------------------- related (transferable) skills
RELATED_GROUPS = [
    ["React", "Angular", "Vue.js", "JavaScript", "TypeScript"],
    ["TensorFlow", "PyTorch", "Keras", "Deep Learning"],
    ["Django", "Flask", "FastAPI", "Python"],
    ["SQL", "MySQL", "PostgreSQL", "SQLite", "SQL Server", "Oracle DB", "DBMS"],
    ["MongoDB", "NoSQL", "Firebase", "Redis"],
    ["AWS", "Azure", "GCP"],
    ["Docker", "Kubernetes"],
    ["Java", "Spring Boot"],
    ["Machine Learning", "Scikit-learn", "Deep Learning", "NLP"],
    ["Matplotlib", "Seaborn", "Plotly", "Power BI", "Tableau", "Data Visualization"],
    ["Pandas", "NumPy", "Data Analysis", "Data Cleaning"],
    ["Git", "GitHub", "GitLab"],
    ["CI/CD", "Jenkins", "GitHub Actions"],
    ["Node.js", "Express.js", "REST APIs"],
    ["NLTK", "spaCy", "Hugging Face", "NLP"],
    ["Unit Testing", "Agile"],
    ["C", "C++"],
    ["Excel", "Power BI", "Tableau"],
]


def related_skills(skill: str) -> set[str]:
    out: set[str] = set()
    for group in RELATED_GROUPS:
        if skill in group:
            out.update(group)
    out.discard(skill)
    return out


# ---------------------------------------------------------------- extraction
def extract_skills(text: str) -> dict[str, dict]:
    """Return {skill: {'category': str, 'count': int}} for every dictionary skill found in text."""
    found = {}
    for skill, pattern in _PATTERNS.items():
        n = len(pattern.findall(text))
        if n:
            found[skill] = {"category": SKILL_CATEGORY[skill], "count": n}
    return found


_PREF_HEADING = re.compile(r"\b(preferred|nice[\s\-]to[\s\-]have|good[\s\-]to[\s\-]have|bonus|desirable|optional|plus points?)\b", re.I)
_REQ_HEADING = re.compile(r"\b(required|requirements?|must[\s\-]have|qualifications?|mandatory|essential|responsibilities|what you.ll (?:need|do)|skills)\b", re.I)
_PREF_INLINE = re.compile(r"(\ba plus\b|\bis a plus\b|\bnice to have\b|\bgood to have\b|\bis preferred\b|\bare preferred\b|\bpreferred\b|\bbonus\b|\badvantageous\b|\bdesirable\b)", re.I)


def extract_jd_skills(jd_text: str) -> dict[str, dict]:
    """
    Extract skills from a job description and label each one REQUIRED or PREFERRED.

    Rule: walk line by line, remembering the current section. Headings such as
    'Preferred qualifications' switch to PREFERRED mode; 'Requirements' switch to REQUIRED mode.
    A line containing 'a plus' / 'nice to have' is PREFERRED on its own. A skill is REQUIRED
    if it appears in at least one required line, otherwise PREFERRED.
    """
    mode = "required"
    req_counts: dict[str, int] = {}
    pref_counts: dict[str, int] = {}

    for raw in jd_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_heading = len(line) <= 60 and (line.endswith(":") or line.isupper() or not line.endswith("."))
        line_mode = mode
        if is_heading and _PREF_HEADING.search(line):
            mode = line_mode = "preferred"
        elif is_heading and _REQ_HEADING.search(line):
            mode = line_mode = "required"
        elif _PREF_INLINE.search(line):
            line_mode = "preferred"        # this line only

        for skill, pattern in _PATTERNS.items():
            n = len(pattern.findall(line))
            if n:
                bucket = pref_counts if line_mode == "preferred" else req_counts
                bucket[skill] = bucket.get(skill, 0) + n

    result = {}
    for skill in set(req_counts) | set(pref_counts):
        importance = "required" if skill in req_counts else "preferred"
        result[skill] = {
            "category": SKILL_CATEGORY[skill],
            "count": req_counts.get(skill, 0) + pref_counts.get(skill, 0),
            "importance": importance,
        }
    return result
