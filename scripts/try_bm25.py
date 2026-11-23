"""Run BM25 queries and print the ranked results with their score breakdown.

Default: load the saved index (data/index/lexical.pkl, made by
python -m scripts.try_index) and run a few example queries on the 10k corpus.
With --toy: build the 3-profile toy index and check one score against the
BM25 formula computed by hand from the raw postings.

Usage (from the project root):
    python -m scripts.try_bm25                                  # example queries
    python -m scripts.try_bm25 "icu nurse" "python developer"   # your own queries
    python -m scripts.try_bm25 "registered nurse" --filter state=OH seniority=senior
    python -m scripts.try_bm25 "data scientist" --judged        # only the 1,000 judged profiles
    python -m scripts.try_bm25 "nurse" --k1 2.0 --b 0.3 --k 5   # other parameters
    python -m scripts.try_bm25 "nurse" --titles                 # also print each profile's current title
    python -m scripts.try_bm25 --toy                            # hand check on the toy index
"""

import argparse
import math
import time

from scripts.try_index import INDEX_PATH, PROFILES
from search_engine.bm25 import BM25
from search_engine.index import Index

DEFAULT_QUERIES = ["icu nurse", "python backend developer", "c++ embedded", "registered nurse", "data scientist"]
N_JUDGED = 1000  # the corpus file puts the 1,000 judged profiles first


def allowed_docs(index, filters, judged):
    """Doc ids that pass every key=value filter (and, with judged, are judged). None = no restriction."""
    if not filters and not judged:
        return None
    allowed = set()
    for doc_id in range(len(index)):
        info = index.read_doc(doc_id)
        if judged and doc_id >= N_JUDGED:
            continue
        if all(str(info.get(key, "")).lower() == value.lower() for key, value in filters.items()):
            allowed.add(doc_id)
    return allowed


def load_titles(index):
    """profile_id -> current title, read from the corpus snapshot (only for printing)."""
    from search_engine.corpus import load_corpus
    from search_engine.fields import profile_to_dict

    titles = {}
    for row in load_corpus():
        title = profile_to_dict(row["profile"] or {}).get("current_title")
        titles[row["profile_id"]] = title[0] if title else ""
    return titles


def print_results(index, query, profiles, breakdowns, seconds, titles=None):
    doc_by_profile = {index.read_doc(d)["profile_id"]: index.read_doc(d) for d in range(len(index))} if titles is not None else None
    print(f"\n=== {query!r}: {len(profiles)} results in {seconds * 1000:.1f} ms ===")
    for rank, ((profile_id, score), (_, parts)) in enumerate(zip(profiles, breakdowns), start=1):
        parts_text = "  ".join(f"{term}={value:.3f}" for term, value in sorted(parts.items(), key=lambda p: -p[1]))
        print(f"{rank:2}. {score:7.3f}  {profile_id}  [{parts_text}]")
        if titles is not None:
            info = doc_by_profile[profile_id]
            print(f"      {titles.get(profile_id, '')!r}  {info.get('job_family')}/{info.get('seniority')}  "
                  f"{info.get('city')}, {info.get('state')}")


def hand_score(index, term, doc_id, k1, b):
    """BM25 for one term in one doc, straight from the raw index structures."""
    n = len(index)
    df = len(index.postings.get(term, {}))
    tf = sum(len(positions) for positions in index.postings[term][doc_id].values())
    dl = sum(index.field_lengths[doc_id].values())
    avgdl = sum(sum(lengths.values()) for lengths in index.field_lengths.values()) / n
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def run_toy(k1, b):
    index = Index(PROFILES)
    scorer = BM25(index, k1=k1, b=b)
    ok = True
    for query in ["nurse", "python", "icu nurse", "c++ python"]:
        start = time.perf_counter()
        profiles, breakdowns = scorer.search(query, k=10)
        print_results(index, query, profiles, breakdowns, time.perf_counter() - start)
        for profile_id, parts in breakdowns:
            doc_id = next(d for d in range(len(index)) if index.read_doc(d)["profile_id"] == profile_id)
            for term, value in parts.items():
                expected = hand_score(index, term, doc_id, k1, b)
                if abs(expected - value) > 1e-9:
                    ok = False
                    print(f"   MISMATCH {term} in {profile_id}: code {value} vs hand {expected}")
        for (profile_id, score), (_, parts) in zip(profiles, breakdowns):
            if abs(score - sum(parts.values())) > 1e-9:
                ok = False
                print(f"   MISMATCH total of {profile_id}: {score} vs sum of parts {sum(parts.values())}")

    # Edge cases
    empty, _ = scorer.search("nurse", k=10, allowed=set())
    unknown, _ = scorer.search("zzzunknownterm", k=10)
    only_ohio, _ = scorer.search("python", k=10, allowed=allowed_docs(index, {"state": "OH"}, False))
    print("\nempty allowed set ->", empty)
    print("unknown term       ->", unknown)
    print("python, state=OH   ->", only_ohio)
    ok = ok and empty == [] and unknown == [] and [p for p, _ in only_ohio] == ["toy_career_changer"]
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


def main():
    parser = argparse.ArgumentParser(description="Run BM25 queries on the saved index.")
    parser.add_argument("queries", nargs="*", help="queries to run (default: a few examples)")
    parser.add_argument("--k", type=int, default=10, help="results per query (default 10)")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only docs whose doc-store value matches, e.g. state=CA seniority=senior")
    parser.add_argument("--judged", action="store_true", help="only rank the 1,000 judged profiles")
    parser.add_argument("--titles", action="store_true", help="print current title and filters (reads the corpus)")
    parser.add_argument("--toy", action="store_true", help="hand check on the 3-profile toy index")
    args = parser.parse_args()

    if args.toy:
        run_toy(args.k1, args.b)
        return

    if not INDEX_PATH.exists():
        raise SystemExit(f"{INDEX_PATH} not found. Build it with: python -m scripts.try_index")
    start = time.perf_counter()
    index = Index.load(INDEX_PATH)
    print(f"Loaded {INDEX_PATH} ({len(index)} docs) in {time.perf_counter() - start:.1f} s")

    filters = dict(item.split("=", 1) for item in args.filter)
    allowed = allowed_docs(index, filters, args.judged)
    if allowed is not None:
        print(f"Filters {filters or ''}{' + judged only' if args.judged else ''}: {len(allowed)} docs allowed")

    titles = load_titles(index) if args.titles else None
    scorer = BM25(index, k1=args.k1, b=args.b)
    print(f"k1={scorer.k1}  b={scorer.b}  avgdl={scorer.avgdl:.2f}")

    for query in args.queries or DEFAULT_QUERIES:
        start = time.perf_counter()
        profiles, breakdowns = scorer.search(query, k=args.k, allowed=allowed)
        print_results(index, query, profiles, breakdowns, time.perf_counter() - start, titles)


if __name__ == "__main__":
    main()
