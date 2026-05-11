# Resume Matching Engine

> Redrob AI Campus Hackathon — Individual Submission

A self-contained Python program that ranks the **Top 3 candidates per Job Description** using **TF-IDF + cosine similarity**, built using only the Python standard library.

---

## Overview

Given:
- **10 noisy resumes** from Indian university students (skills contain typos, mixed casing, inconsistent separators).
- **3 Job Descriptions** from Korean technology companies (Kakao, Naver, Line).

The program:
1. Normalizes every noisy skill token via a fixed `SKILL_ALIASES` map.
2. Deduplicates canonical skills per resume.
3. Builds a shared vocabulary from all normalized resume skills.
4. Computes **TF-IDF** vectors for resumes (TF = `1/N`, IDF = `ln(10/df)`).
5. Builds **binary** vectors for JDs over the same vocabulary.
6. Calculates **cosine similarity** between every (resume, JD) pair.
7. Prints the Top 3 candidates per JD with scores rounded to 2 decimal places.

---

## Algorithm

| Stage | Description |
|------:|-------------|
| 1 | **Skill Normalization** — split on commas, lowercase, look up in `SKILL_ALIASES`, drop unknown tokens. |
| 2 | **Deduplication** — each canonical skill appears at most once per resume (insertion order preserved). |
| 3 | **Vocabulary Construction** — alphabetically sorted union of every canonical skill across the 10 resumes. |
| 4 | **TF-IDF (resumes only)** — `TF = 1/N`, `IDF = ln(10/df)`. |
| 5 | **JD Binary Vectors** — `1` if vocab term is in the JD's canonical skills, else `0`. JD skills not in the resume vocabulary are silently dropped. |
| 6 | **Cosine Similarity** — `(A · B) / (|A| · |B|)`. |
| 7 | **Ranking** — sort by descending score with alphabetical tie-break on candidate name; keep top 3. |

---

## Reference Formulas

```
TF(skill, resume)   = count(skill in resume) / total_unique_skills(resume)
After deduplication : TF = 1 / N           where N = unique skills in the resume

IDF(skill)          = ln( 10 / df(skill) )   df = number of resumes containing skill
                                             Natural logarithm, no smoothing.

TF-IDF              = TF × IDF

Cosine(A, B)        = (A · B) / ( |A| × |B| )
                      A = resume TF-IDF vector
                      B = JD binary vector
                      |X| = Euclidean norm of X
```

---

## File Structure

```
RedRobe AI Sol/
├── resume_matcher.py    # Complete solution — run this
└── README.md            # This file
```

---

## How to Run

```bash
python resume_matcher.py
```

Requires only Python 3.7+ (no external packages).

---

## Final Output

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## Constraints (per problem sheet)

**Allowed**
- Standard library only — `math`, `sys`.
- Redrob AI as the coding assistant.

**Not Allowed**
- External libraries: `numpy`, `pandas`, `scikit-learn`, `nltk`, etc.
- Modifying `SKILL_ALIASES` (used verbatim from the problem sheet).
- Custom aliases beyond the provided map.

---

## Code Structure

`resume_matcher.py` is organised by pipeline stage:

| Function | Purpose |
|---|---|
| `normalize_skills(raw)` | Split, lowercase, alias-map, drop unknowns, dedup. |
| `build_vocabulary(skills_by_name)` | Alphabetically sorted union of all resume skills. |
| `compute_document_frequency(vocab, skills_by_name)` | Count resumes containing each vocab term. |
| `compute_idf(vocab, df, total_docs)` | `ln(total_docs / df)` per term. |
| `compute_tfidf_vector(skills, vocab, idf)` | TF-IDF vector for one resume. |
| `compute_jd_vector(jd_skills, vocab)` | Binary vector for one JD. |
| `cosine_similarity(a, b)` | Cosine between two vectors. |
| `rank_top_candidates(pairs, k)` | Sort by `(-score, name)`; take top *k*. |
| `format_candidates(top)` | Format as `Name(0.57), Name(0.53), ...`. |
| `main()` | Orchestrates the pipeline. |

---

## Verification Notes

| Check | Expected |
|---|---|
| Vocabulary size | 48 skills |
| Skills with df > 1 | `python=6, machine_learning=3, data_visualization=3, java=2, javascript=2, nodejs=2, react=2, rest_api=2, sql=2` |
| IDF spot values | df=1 → 2.302585, df=2 → 1.609438, df=3 → 1.203973, df=6 → 0.510826 |
| JD skills dropped (not in vocab) | `pytorch` (JD-1), `redis` (JD-2) |
| JD active positions | JD-1: 10, JD-2: 9, JD-3: 11 |

Spot-checking Rahul Gupta against JD-2 (all six of his canonical skills appear in JD-2):
```
|A| = sqrt(ln(5)² + 5·ln(10)²) / 6   ≈ 0.89907
A·B = ln(5)/6 + 5·ln(10)/6           ≈ 2.18706
|B| = sqrt(9) = 3
Cosine ≈ 2.18706 / (0.89907 × 3)     ≈ 0.8109 → rounds to 0.81 ✓
```
