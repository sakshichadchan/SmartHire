"""
preprocessing.py
----------------
Step 2: NLP text pre-processing used before TF-IDF.

Pipeline:  raw text -> lowercase & clean -> tokenize -> stop-word removal -> stemming
           -> one space-separated string of normalised tokens.

NLTK is used when installed (Porter stemmer + stop-word list). If NLTK or its data
cannot be loaded (e.g. offline), built-in fallbacks keep the project running.
Technical tokens such as 'C++', 'C#', '.NET', 'Node.js' are protected so that cleaning
does not destroy them.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

# ----------------------------------------------------------------- stop words
_FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or other
our ours out over own same she should so some such than that the their theirs them then there these they this
those through to too under until up very was we were what when where which while who whom why will with would
you your yours yourself
""".split())

# Words that are common in resumes / job posts but carry no skill information.
DOMAIN_STOPWORDS = set("""
experience experienced skills skill ability able strong good excellent knowledge work working worked years year
required requirements requirement preferred responsibilities responsibility role looking candidate candidates
including include etc must understanding familiarity hands using use used like well also new based job position
company responsible proficient proficiency qualifications qualification plus nice opportunity join ideal
""".split())


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    words = set(_FALLBACK_STOPWORDS)
    try:
        from nltk.corpus import stopwords

        try:
            words |= set(stopwords.words("english"))
        except LookupError:                       # data not downloaded yet
            import nltk

            nltk.download("stopwords", quiet=True)
            words |= set(stopwords.words("english"))
    except Exception:
        pass                                      # keep built-in list
    return frozenset(words | DOMAIN_STOPWORDS)


# -------------------------------------------------------------------- stemmer
try:
    from nltk.stem import PorterStemmer          # needs no downloaded data

    _porter = PorterStemmer()
    STEMMER_NAME = "NLTK PorterStemmer"

    def _raw_stem(word: str) -> str:
        return _porter.stem(word)
except Exception:                                 # pragma: no cover - fallback
    STEMMER_NAME = "built-in suffix stripper"

    def _raw_stem(word: str) -> str:
        for suf in ("ingly", "edly", "ing", "ed", "ies", "es", "s", "ly"):
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                return word[: -len(suf)]
        return word


@lru_cache(maxsize=50000)
def stem(word: str) -> str:
    return _raw_stem(word)


# ---------------------------------------------------------------- cleaning
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")

# Protect technical tokens BEFORE punctuation is stripped.
TECH_NORMALIZATION = [
    (re.compile(r"(?<![a-z0-9])c\+\+"), " cpp "),
    (re.compile(r"(?<![a-z0-9])c#"), " csharp "),
    (re.compile(r"\.net\b"), " dotnet "),
    (re.compile(r"node\.js"), " nodejs "),
    (re.compile(r"react\.js"), " reactjs "),
    (re.compile(r"vue\.js"), " vuejs "),
    (re.compile(r"ci\s*/\s*cd"), " cicd "),
    (re.compile(r"scikit[\s\-]learn"), " scikitlearn "),
]


def clean_text(text: str) -> str:
    """Lowercase, drop URLs / e-mails / phone numbers, protect tech tokens, strip punctuation."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    for pattern, repl in TECH_NORMALIZATION:
        text = pattern.sub(repl, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(clean: str) -> list[str]:
    """Regex tokenizer (no NLTK 'punkt' download needed). Drops 1-char and pure-number tokens."""
    return [t for t in re.findall(r"[a-z0-9]+", clean) if len(t) > 1 and not t.isdigit()]


@dataclass
class PreprocessResult:
    cleaned: str = ""
    tokens: list[str] = field(default_factory=list)       # after tokenization
    filtered: list[str] = field(default_factory=list)     # after stop-word removal
    stemmed: list[str] = field(default_factory=list)      # after stemming
    processed: str = ""                                    # final string fed to TF-IDF


def preprocess(text: str) -> PreprocessResult:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    sw = get_stopwords()
    filtered = [t for t in tokens if t not in sw]
    stemmed = [stem(t) for t in filtered]
    return PreprocessResult(cleaned, tokens, filtered, stemmed, " ".join(stemmed))


def build_stem_lookup(text: str) -> dict[str, str]:
    """Map each stem back to its most common original word (so charts show 'develop', not 'develop' + 'ment')."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for tok in preprocess(text).filtered:
        counts[stem(tok)][tok] += 1
    return {s: c.most_common(1)[0][0] for s, c in counts.items()}
