"""
Resume Matching Engine - Redrob AI Campus Hackathon
====================================================

Reads a noisy resume skill dataset and three Korean-company job descriptions,
then ranks the Top 3 candidates per JD by cosine similarity of TF-IDF vectors.

Pipeline
--------
    1. Normalize skills      : split on ',', lowercase, alias-map, drop unknowns
    2. Deduplicate           : each canonical skill appears at most once per resume
    3. Build vocabulary      : sorted union of all canonical resume skills
    4. Resume TF-IDF         : TF = 1/N  (after dedup), IDF = ln(10/df), no smoothing
    5. JD binary vectors     : 1 if vocab term is in the JD's canonical skill set
    6. Cosine similarity     : Cosine(A,B) = (A . B) / (|A| * |B|)
    7. Ranking               : Top 3 by descending score; alphabetical tie-break

Constraints
-----------
    * Standard library only (math, sys).
    * SKILL_ALIASES is reproduced verbatim from the problem sheet and never mutated.

Run:
    python resume_matcher.py
"""

import math
import sys

# Ensure UTF-8 stdout so the em-dash (-) in the expected output format
# renders correctly on Windows consoles (default cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# =============================================================================
# Configuration constants
# =============================================================================
TOTAL_RESUMES = 10
TOP_K = 3


# =============================================================================
# SKILL_ALIASES  --  Use exactly as provided. DO NOT MODIFY.
# =============================================================================
SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    # Web -- Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    # Web -- Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android",
    "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}


# =============================================================================
# Datasets  --  10 candidate resumes and 3 job descriptions
# =============================================================================
RESUMES = {
    "Arjun Sharma":    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair":      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta":     "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel":     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh":    "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta":     "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao":     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar":    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer":      "python, R, statistics, ML, regression, clustering, Power-BI",
}

JOB_DESCRIPTIONS = {
    "JD-1": {
        "company": "Kakao",
        "role":    "ML Engineer",
        "skills":  "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, "
                   "SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics",
    },
    "JD-2": {
        "company": "Naver",
        "role":    "Backend Engineer",
        "skills":  "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, "
                   "Kubernetes, REST API, CI/CD, Redis",
    },
    "JD-3": {
        "company": "Line",
        "role":    "Frontend Engineer",
        "skills":  "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, "
                   "Node.js, GraphQL, Redux, Jest, AWS",
    },
}


# =============================================================================
# Stage 1 & 2  --  Skill normalization and deduplication
# =============================================================================
def normalize_skills(raw_skill_string: str) -> list:
    """Convert a noisy skill string into a deduplicated list of canonical skills.

    Steps:
        1. Split on commas and strip whitespace.
        2. Lowercase every token.
        3. Map through SKILL_ALIASES (multi-word phrases are already valid keys,
           so direct lookup correctly handles them).
        4. Discard tokens not present in the alias map.
        5. Deduplicate while preserving insertion order.
    """
    tokens = [token.strip().lower() for token in raw_skill_string.split(",")]
    canonical = [SKILL_ALIASES[t] for t in tokens if t in SKILL_ALIASES]

    seen = set()
    deduplicated = []
    for skill in canonical:
        if skill not in seen:
            seen.add(skill)
            deduplicated.append(skill)
    return deduplicated


# =============================================================================
# Stage 3  --  Vocabulary construction
# =============================================================================
def build_vocabulary(resume_skills_by_name: dict) -> list:
    """Return the sorted alphabetical union of all canonical resume skills."""
    vocabulary = set()
    for skills in resume_skills_by_name.values():
        vocabulary.update(skills)
    return sorted(vocabulary)


# =============================================================================
# Stage 4  --  TF-IDF for resumes
# =============================================================================
def compute_document_frequency(vocabulary: list, resume_skills_by_name: dict) -> dict:
    """For each vocabulary term, count how many resumes contain it."""
    return {
        term: sum(1 for skills in resume_skills_by_name.values() if term in skills)
        for term in vocabulary
    }


def compute_idf(vocabulary: list, document_frequency: dict, total_docs: int) -> dict:
    """IDF(term) = ln(total_docs / df(term))  --  natural log, no smoothing."""
    return {term: math.log(total_docs / document_frequency[term]) for term in vocabulary}


def compute_tfidf_vector(skills: list, vocabulary: list, idf: dict) -> list:
    """Build the TF-IDF vector for a single resume.

    Because every kept skill appears exactly once per resume after dedup,
    TF reduces to 1/N where N is the number of unique skills in the resume.
    """
    n_skills = len(skills)
    skill_set = set(skills)
    return [
        (1 / n_skills) * idf[term] if term in skill_set else 0.0
        for term in vocabulary
    ]


# =============================================================================
# Stage 5  --  JD binary vectors over the same vocabulary
# =============================================================================
def compute_jd_vector(jd_skills: list, vocabulary: list) -> list:
    """Binary vector: 1 if vocab term is in the JD's canonical skills, else 0.

    JD skills not present in the resume vocabulary are silently dropped.
    """
    skill_set = set(jd_skills)
    return [1.0 if term in skill_set else 0.0 for term in vocabulary]


# =============================================================================
# Stage 6  --  Cosine similarity
# =============================================================================
def cosine_similarity(vector_a: list, vector_b: list) -> float:
    """Cosine similarity = (A . B) / (|A| * |B|).  Returns 0.0 if either norm is 0."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# =============================================================================
# Stage 7  --  Ranking and output formatting
# =============================================================================
def rank_top_candidates(name_score_pairs: list, k: int = TOP_K) -> list:
    """Sort by descending score with alphabetical name as the tie-breaker; return top k."""
    return sorted(name_score_pairs, key=lambda pair: (-pair[1], pair[0]))[:k]


def format_candidates(top_candidates: list) -> str:
    """Format as 'Name(0.57), Name(0.53), Name(0.40)' with 2-decimal scores."""
    return ", ".join(
        f"{name}({round(score, 2):.2f})" for name, score in top_candidates
    )


# =============================================================================
# Main pipeline
# =============================================================================
def main() -> None:
    # Stage 1 & 2  --  Normalize + deduplicate every resume's skill string
    resume_skills = {
        name: normalize_skills(raw) for name, raw in RESUMES.items()
    }

    # Stage 3  --  Build the shared vocabulary from resumes only
    vocabulary = build_vocabulary(resume_skills)

    # Stage 4  --  Compute df, IDF, and the TF-IDF vector for every resume
    document_frequency = compute_document_frequency(vocabulary, resume_skills)
    idf = compute_idf(vocabulary, document_frequency, TOTAL_RESUMES)
    resume_vectors = {
        name: compute_tfidf_vector(skills, vocabulary, idf)
        for name, skills in resume_skills.items()
    }

    # Stage 5  --  Build a binary vector for each JD over the same vocabulary
    jd_vectors = {
        jd_id: compute_jd_vector(normalize_skills(jd["skills"]), vocabulary)
        for jd_id, jd in JOB_DESCRIPTIONS.items()
    }

    # Stage 6 & 7  --  Score every resume against every JD; print Top 3
    for jd_id, jd in JOB_DESCRIPTIONS.items():
        name_score_pairs = [
            (name, cosine_similarity(resume_vector, jd_vectors[jd_id]))
            for name, resume_vector in resume_vectors.items()
        ]
        top_candidates = rank_top_candidates(name_score_pairs)
        print(f"{jd_id} — {jd['company']} ({jd['role']})")
        print(format_candidates(top_candidates))
        print()


if __name__ == "__main__":
    main()
