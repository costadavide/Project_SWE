"""Build an Index and print what it stores.

Default: build from the corpus snapshot (data/corpus/profiles.jsonl, made by
python -m search_engine.corpus), save it to data/index/lexical.pkl, and print
corpus-level statistics plus a few example lookups.
With --toy: build from 3 hand-made profiles and print everything (no files needed).

Usage (from the project root):
    python -m scripts.try_index                 # full corpus (10k), saves the index
    python -m scripts.try_index --limit 500     # first 500 corpus profiles, not saved
    python -m scripts.try_index --toy           # 3 hand-made profiles
    python -m scripts.try_index --terms nurs python engin   # choose the example terms
"""

import argparse
import json
import time
from pathlib import Path

from search_engine.corpus import load_corpus
from search_engine.index import Index

INDEX_PATH = Path("data") / "index" / "lexical.pkl"

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


def print_field_stats(index):
    fields = sorted(set(index.sum_length_field) | set(index.n_docs_per_field))
    print(f"{'field':15} {'sum':>9} {'docs':>6} {'avg':>7}")
    for field in fields:
        print(f"{field:15} {index.sum_length_field[field]:9} {index.n_docs_per_field[field]:6} "
              f"{index.avg_field_length(field):7.2f}")


def show_toy(index):
    print("=== doc_store (internal doc_id -> stored info) ===")
    for doc_id, info in enumerate(index.doc_store):
        print(doc_id, info)
    print("\n=== postings (term -> doc_id -> field -> positions), sorted by term ===")
    for term in sorted(index.postings):
        print(f"{term:15} df={index.doc_freq(term)}  {json.dumps(plain(index.read_term(term)))}")
    print("\n=== field_lengths (doc_id -> field -> number of terms) ===")
    for doc_id in range(len(index)):
        print(doc_id, plain(index.field_lengths.get(doc_id, {})))
    print("\n=== per-field statistics ===")
    print_field_stats(index)


def show_corpus(index, terms):
    print(f"documents: {len(index)}   distinct terms: {len(index.postings)}")

    print("\n=== per-field statistics ===")
    print_field_stats(index)

    print("\n=== 20 most common terms (document frequency) ===")
    common = sorted(index.postings, key=index.doc_freq, reverse=True)[:20]
    for term in common:
        print(f"{term:15} df={index.doc_freq(term):6}  ({index.doc_freq(term) / len(index):.0%} of docs)")

    print("\n=== example lookups (first 3 docs per term) ===")
    for term in terms:
        postings = index.read_term(term)
        print(f"\n{term!r}: df={len(postings)}")
        for doc_id in list(postings)[:3]:
            print(f"  doc {doc_id} {index.read_doc(doc_id)['profile_id']}: {json.dumps(plain(postings[doc_id]))}")

    print("\n=== doc 0 ===")
    print(index.read_doc(0))
    print({field: index.read_field_length(0, field) for field in index.field_lengths.get(0, {})})


def main():
    parser = argparse.ArgumentParser(description="Build an Index and print what it stores.")
    parser.add_argument("--toy", action="store_true", help="use 3 hand-made profiles instead of the corpus")
    parser.add_argument("--limit", type=int, help="only index the first N corpus profiles (not saved)")
    parser.add_argument("--terms", nargs="+", default=["nurs", "python", "engin", "icu", "c++"],
                        help="analyzed terms to look up in the example section")
    args = parser.parse_args()

    if args.toy:
        show_toy(Index(PROFILES))
        return

    rows = load_corpus()
    if args.limit:
        rows = rows[:args.limit]
    print(f"Building the index from {len(rows)} corpus profiles...")
    start = time.perf_counter()
    index = Index(rows)
    print(f"Built in {time.perf_counter() - start:.1f} s")

    if not args.limit:
        start = time.perf_counter()
        index.save(INDEX_PATH)
        size_mb = INDEX_PATH.stat().st_size / 1e6
        print(f"Saved to {INDEX_PATH} ({size_mb:.1f} MB) in {time.perf_counter() - start:.1f} s")

    print()
    show_corpus(index, args.terms)


if __name__ == "__main__":
    main()
