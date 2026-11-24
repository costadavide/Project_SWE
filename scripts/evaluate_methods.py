"""Evaluate search methods on the 16 graded queries over the 1,000 judged profiles.

Methods: bm25, bm25f, semantic (core embedding only) and semantic_weighted
(core 0.80 / projects 0.15 / interests 0.05), all run by default.
Every method is anything with search(query, k, allowed) returning
(profiles, breakdowns), where profiles = [(profile_id, score), ...] best first,
like BM25 and BM25F in search_engine/bm25.py. A semantic searcher or a fusion
method with the same search() can be added to build_methods() with one line.

Each method ranks only the 1,000 judged profiles (tests/retrieval_1000/
profile_ids.json): unjudged profiles would count as grade 0 and make every
method look worse at random. The queries are run as raw text, with no parsing:
the location query ("... in Arizona") needs the query parser's state filter,
and locations are not indexed, so expect it to be weak for now.

Output: a per-query NDCG@10 table, a summary per method (mean NDCG@10, MRR@10,
P@5, R@10, p50/p95 latency), means by query type, wins/losses against the
first method, and a JSON file with everything (default data/results/evaluate_methods.json).

Usage (from the project root, after python -m scripts.try_index):
    python -m scripts.evaluate_methods
    python -m scripts.evaluate_methods --methods bm25f
    python -m scripts.evaluate_methods --methods bm25f semantic   # semantic needs the embeddings
                                         # snapshot: python -m search_engine.corpus --embeddings-only
    python -m scripts.evaluate_methods --show-top 3      # also print the top 3 titles per query
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.try_index import INDEX_PATH
from search_engine.bm25 import BM25, BM25F
from search_engine.index import Index
from search_engine.metrics import mean_metrics, score_query

project_root = Path(__file__).resolve().parent.parent
JUDGMENTS_PATH = project_root / "tests" / "retrieval_1000" / "judgments.json"
JUDGED_IDS_PATH = project_root / "tests" / "retrieval_1000" / "profile_ids.json"
DEFAULT_OUTPUT = project_root / "data" / "results" / "evaluate_methods.json"
K = 10
MAIN_METRIC = f"NDCG@{K}"
AVAILABLE_METHODS = ["bm25", "bm25f", "semantic", "semantic_weighted"]
# test_embeddings.py's "weighted" mode
SEMANTIC_WEIGHTS = {"core": 0.80, "projects": 0.15, "interests": 0.05}


def load_judgments(path=JUDGMENTS_PATH):
    """The graded queries: a list of dicts with query_id, query, query_type and
    grades = {profile_id: grade} (grade 2 or 1; unlisted profiles are 0)."""
    with Path(path).open(encoding="utf-8") as file:
        queries = json.load(file)["queries"]
    for query in queries:
        grades = {}
        for grade, profile_ids in query["relevant"].items():
            for profile_id in profile_ids:
                if profile_id in grades:
                    raise ValueError(f"{profile_id} is graded twice in {query['query_id']}")
                grades[profile_id] = int(grade)
        query["grades"] = grades
    return queries


def judged_doc_ids(index, path=JUDGED_IDS_PATH):
    """Doc ids of the 1,000 judged profiles; fails if any of them is not in the index."""
    with Path(path).open(encoding="utf-8") as file:
        profile_ids = [row["profile_id"] for row in json.load(file)]
    missing = [pid for pid in profile_ids if index.read_doc_id(pid) is None]
    if missing:
        raise ValueError(f"{len(missing)} judged profiles are not in the index, e.g. {missing[:3]}. "
                         f"Rebuild it from the corpus: python -m scripts.try_index")
    allowed = {index.read_doc_id(pid) for pid in profile_ids}
    if len(allowed) != len(profile_ids):
        raise ValueError("Two judged profile ids map to the same doc id")
    return allowed


def build_methods(index, names=None):
    """name -> searcher, for the requested names (None = all). Add fusion here once it
    has the same search(). The semantic searchers share one model, loaded once."""
    names = names or AVAILABLE_METHODS
    methods = {}
    if "bm25" in names:
        methods["bm25"] = BM25(index)
    if "bm25f" in names:
        methods["bm25f"] = BM25F(index)
    if any(name.startswith("semantic") for name in names):
        from semantic_search.searcher import SemanticSearcher

        semantic = SemanticSearcher()
        semantic.check_alignment(index)
        if "semantic" in names:
            methods["semantic"] = semantic
        if "semantic_weighted" in names:
            methods["semantic_weighted"] = SemanticSearcher(weights=SEMANTIC_WEIGHTS, model=semantic.load_model())
    return methods


def percentile(values, share):
    """Nearest-rank percentile, e.g. share=0.95 for p95."""
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round(share * len(ordered)) - 1))
    return ordered[position]


def evaluate(name, searcher, queries, allowed, judged_profiles):
    """Run every query through one method; returns one result dict per query."""
    results = []
    for query in queries:
        start = time.perf_counter()
        profiles, _ = searcher.search(query["query"], K, allowed)
        latency_ms = (time.perf_counter() - start) * 1000
        ranked_ids = [profile_id for profile_id, _ in profiles]
        outside = [pid for pid in ranked_ids if pid not in judged_profiles]
        if outside:
            raise AssertionError(f"{name} returned unjudged profiles for {query['query_id']}: {outside[:3]}")
        results.append({
            "query_id": query["query_id"],
            "query_type": query["query_type"],
            "query": query["query"],
            "metrics": score_query(ranked_ids, query["grades"], k=K),
            "latency_ms": latency_ms,
            "top_10": [{"profile_id": pid, "score": round(score, 4), "grade": query["grades"].get(pid, 0)}
                       for pid, score in profiles],
        })
    return results


def summarize(results):
    latencies = [r["latency_ms"] for r in results]
    summary = mean_metrics([r["metrics"] for r in results])
    summary["p50_ms"] = percentile(latencies, 0.50)
    summary["p95_ms"] = percentile(latencies, 0.95)
    summary["queries"] = len(results)
    return summary


def print_tables(queries, all_results, summaries, by_type, show_top, titles):
    names = list(all_results)
    width = max(10, *(len(n) for n in names))

    print(f"\n=== {MAIN_METRIC} per query ===")
    print(f"{'query':20} {'type':11}" + "".join(f" {n:>{width}}" for n in names) + "   top-10 grades")
    for i, query in enumerate(queries):
        row = f"{query['query_id']:20} {query['query_type']:11}"
        grade_strings = []
        for name in names:
            result = all_results[name][i]
            row += f" {result['metrics'][MAIN_METRIC]:>{width}.3f}"
            grade_strings.append("".join(str(hit["grade"]) for hit in result["top_10"]).ljust(K, "-"))
        print(row + "   " + " | ".join(grade_strings))
        if show_top:
            for name in names:
                tops = [titles.get(hit["profile_id"], hit["profile_id"])[:30] for hit in all_results[name][i]["top_10"][:show_top]]
                print(f"{'':32}{name}: " + "; ".join(tops))

    metric_names = [m for m in summaries[names[0]] if m not in ("p50_ms", "p95_ms", "queries")]
    print("\n=== summary (mean over queries) ===")
    print(f"{'method':{width}}" + "".join(f" {m:>12}" for m in metric_names) + f" {'p50 ms':>8} {'p95 ms':>8}")
    for name in names:
        s = summaries[name]
        print(f"{name:{width}}" + "".join(f" {s[m]:12.3f}" for m in metric_names)
              + f" {s['p50_ms']:8.1f} {s['p95_ms']:8.1f}")

    print(f"\n=== {MAIN_METRIC} by query type ===")
    for query_type, per_method in by_type.items():
        count = per_method[names[0]]["queries"]
        print(f"{query_type:12} (n={count:2})" + "".join(f"  {n} {per_method[n][MAIN_METRIC]:.3f}" for n in names))

    if len(names) > 1:
        base = names[0]
        print(f"\n=== {MAIN_METRIC} wins / losses against {base} (difference > 0.001) ===")
        for name in names[1:]:
            diffs = [all_results[name][i]["metrics"][MAIN_METRIC] - all_results[base][i]["metrics"][MAIN_METRIC]
                     for i in range(len(queries))]
            better = sum(d > 0.001 for d in diffs)
            worse = sum(d < -0.001 for d in diffs)
            print(f"{name} vs {base}: {better} better, {worse} worse, {len(diffs) - better - worse} equal")


def load_titles():
    """profile_id -> current title from the corpus snapshot (only for --show-top)."""
    from search_engine.corpus import load_corpus
    from search_engine.fields import profile_to_dict

    titles = {}
    for row in load_corpus():
        title = profile_to_dict(row["profile"] or {}).get("current_title")
        titles[row["profile_id"]] = title[0] if title else ""
    return titles


def method_settings(searcher):
    """The parameters of a method, saved with the results so runs can be compared."""
    settings = {key: value for key, value in vars(searcher).items()
                if key in ("k1", "b", "w", "weights", "query_prompt", "model_name")
                and isinstance(value, (int, float, dict, str))}
    return settings


def main():
    parser = argparse.ArgumentParser(description="Evaluate search methods on the 1,000 judged profiles.")
    parser.add_argument("--methods", nargs="+", help="which methods to run (default: all in build_methods)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to save the JSON results")
    parser.add_argument("--show-top", type=int, default=0, metavar="N",
                        help="also print the current titles of the top N results per query")
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        raise SystemExit(f"{INDEX_PATH} not found. Build it with: python -m scripts.try_index")
    index = Index.load(INDEX_PATH)
    queries = load_judgments()
    allowed = judged_doc_ids(index)
    judged_profiles = {index.read_doc(doc_id)["profile_id"] for doc_id in allowed}
    print(f"Loaded {INDEX_PATH} ({len(index)} docs), {len(queries)} queries, "
          f"ranking only the {len(allowed)} judged profiles.")

    if args.methods:
        unknown = [name for name in args.methods if name not in AVAILABLE_METHODS]
        if unknown:
            parser.error(f"unknown methods {unknown}; available: {AVAILABLE_METHODS}")
    methods = build_methods(index, args.methods)

    all_results = {name: evaluate(name, searcher, queries, allowed, judged_profiles)
                   for name, searcher in methods.items()}
    summaries = {name: summarize(results) for name, results in all_results.items()}
    by_type = {}
    for query_type in sorted({q["query_type"] for q in queries}):
        by_type[query_type] = {name: summarize([r for r in results if r["query_type"] == query_type])
                               for name, results in all_results.items()}

    titles = load_titles() if args.show_top else {}
    print_tables(queries, all_results, summaries, by_type, args.show_top, titles)

    output = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": str(INDEX_PATH),
        "k": K,
        "judged_profiles": len(allowed),
        "settings": {name: method_settings(searcher) for name, searcher in methods.items()},
        "summary": summaries,
        "by_query_type": by_type,
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
