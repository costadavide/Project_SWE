"""Build an Index from 3 hand-made profiles and print everything it stores.

Throwaway helper to see what the index looks like. No database needed.

Usage (from the project root):
    python -m scripts.try_index
"""

import json

from search_engine.index import Index

PROFILES = [
    {
        "profile_id": "toy_nurse",
        "job_family": "healthcare",
        "seniority": "senior",
        "city": "Toledo",
        "state": "OH",
        "experience_years": 9,
        "profile": {
            "user": {
                "bio": "Registered nurse with ICU experience.",
                "skills": ["Critical care", "IV therapy", "registered nurse"],
            },
            "experience": [
                {"title": "ICU Nurse", "start_date": "03/2021", "end_date": None,
                 "description": "Care for critically ill patients."},
                {"title": "Registered Nurse", "start_date": "06/2016", "end_date": "02/2021"},
            ],
            "education": [{"degree": "Bachelor of Science", "field_of_study": "Nursing"}],
            "certifications": [{"name": "BLS"}],
        },
    },
    {
        "profile_id": "toy_engineer",
        "job_family": "engineering",
        "seniority": "mid",
        "city": "Austin",
        "state": "TX",
        "experience_years": 4,
        "profile": {
            "user": {
                "bio": "Full-stack engineer who loves Python.",
                "skills": ["Python", "Node.js", "C++", "SQL"],
            },
            "experience": [
                {"title": "Software Engineer", "start_date": "01/2022", "end_date": None,
                 "description": "Build REST APIs in Python."},
            ],
            "education": [{"degree": "Bachelor of Science", "field_of_study": "Computer Science"}],
        },
    },
    {
        "profile_id": "toy_career_changer",
        "job_family": "engineering",
        "seniority": "entry",
        "city": "Toledo",
        "state": "OH",
        "experience_years": 0,
        "profile": {
            "user": {
                "bio": "Former nurse learning Python.",
                "skills": ["Python", "data analysis"],
            },
            "experience": [],
        },
    },
]


def plain(value):
    """Turn nested defaultdicts into plain dicts so they print cleanly."""
    if isinstance(value, dict):
        return {key: plain(inner) for key, inner in value.items()}
    return value


index = Index(PROFILES)

print("=== N (number of documents) ===")
print(len(index))

print("\n=== doc_store (internal doc_id -> stored info) ===")
for doc_id, info in enumerate(index.doc_store):
    print(doc_id, info)

print("\n=== postings (term -> doc_id -> field -> positions), sorted by term ===")
for term in sorted(index.postings):
    print(f"{term:15} df={index.doc_freq(term)}  {json.dumps(plain(index.read_term(term)))}")

print("\n=== field_lengths (doc_id -> field -> number of terms) ===")
for doc_id in range(len(index)):
    print(doc_id, plain(index.field_lengths.get(doc_id, {})))

print("\n=== per-field statistics ===")
fields = sorted(set(index.sum_length_field) | set(index.n_docs_per_field))
print(f"{'field':15} {'sum':>5} {'docs':>5} {'avg':>6}")
for field in fields:
    print(f"{field:15} {index.sum_length_field[field]:5} {index.n_docs_per_field[field]:5} "
          f"{index.avg_field_length(field):6.2f}")

print("\n=== example lookups ===")
for term in ["nurs", "python", "xyz"]:
    print(f"read_term({term!r}) -> {json.dumps(plain(index.read_term(term)))}")
print("read_field_length(2, 'current_title') ->", index.read_field_length(2, "current_title"))
print("read_doc(1) ->", index.read_doc(1))
