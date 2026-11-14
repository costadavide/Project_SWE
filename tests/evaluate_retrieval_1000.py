"""Graded retrieval test on a fixed set of 1000 profiles.

Run from the project root:
    python -m tests.evaluate_retrieval_1000
    python -m tests.evaluate_retrieval_1000 --data-dir data/sample_1000 --experience-weight 0.2

The index in --data-dir must contain exactly the profiles listed in
tests/retrieval_1000/profile_ids.json. Build it with:
    python -m scripts.build_index --ids tests/retrieval_1000/profile_ids.json --data-dir data/sample_1000

Unlike scripts/evaluate.py, which only checks where one intended profile ranks,
every profile was graded for every query (2 = relevant, 1 = partial, else 0),
so the whole top 10 is scored.
"""

import argparse
import json
import math
from pathlib import Path

from sentence_transformers import SentenceTransformer

from scripts.evaluate import display_path, load_index, load_json
from semantic_search.scoring import combined_scores

TOP_K = 10
TOP_P = 5

project_root = Path(__file__).resolve().parent.parent
test_dir = project_root / "tests" / "retrieval_1000"
default_data_dir = project_root / "data" / "sample_1000"
default_results_path = test_dir / "results" / "latest.json"


def load_judgments():
    queries = load_json(test_dir / "judgments.json")["queries"]
    for query in queries:
        grades = {}
        for grade, profile_ids in query["relevant"].items():
            for profile_id in profile_ids:
                if profile_id in grades:
                    raise ValueError(f"{profile_id} is graded twice in {query['query_id']}.")
                grades[profile_id] = int(grade)
        query["grades"] = grades
    return queries


def check_profile_set(profile_ids):
    """Grades assume exactly this profile set: an extra profile would silently count as 0."""
    expected = {row["profile_id"] for row in load_json(test_dir / "profile_ids.json")}
    actual = set(profile_ids)
    if actual != expected:
        raise ValueError(
            f"The index does not match tests/retrieval_1000/profile_ids.json "
            f"({len(expected - actual)} missing, {len(actual - expected)} extra). Rebuild it with:\n"
            f"  python -m scripts.build_index --ids tests/retrieval_1000/profile_ids.json "
            f"--data-dir <folder>"
        )


def dcg(gains):
    return sum((2 ** gain - 1) / math.log2(position + 2) for position, gain in enumerate(gains))


def score_query(ranked_ids, grades):
    gains = [grades.get(profile_id, 0) for profile_id in ranked_ids[:TOP_K]]
    ideal = sorted(grades.values(), reverse=True)[:TOP_K]
    relevant = [pid for pid, grade in grades.items() if grade == 2]
    first_relevant = next((rank for rank, gain in enumerate(gains, start=1) if gain == 2), None)
    return {
        "ndcg_at_10": dcg(gains) / dcg(ideal),
        "mrr_at_10": 1 / first_relevant if first_relevant else 0.0,
        "precision_at_5": sum(gain > 0 for gain in gains[:TOP_P]) / TOP_P,
        "recall_at_10": sum(gain == 2 for gain in gains) / len(relevant),
        "top_10_gains": gains,
    }


METRICS = ["ndcg_at_10", "mrr_at_10", "precision_at_5", "recall_at_10"]


def mean(results):
    return {metric: sum(r[metric] for r in results) / len(results) for metric in METRICS} | {
        "queries": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Graded retrieval test on 1000 profiles.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir,
                        help="index folder built from tests/retrieval_1000/profile_ids.json")
    parser.add_argument("--experience-weight", type=float,
                        help="override experience_weight from index_metadata.json (0 = full profile only)")
    parser.add_argument("--output", type=Path, default=default_results_path,
                        help="where to write the detailed results JSON")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    profile_ids, embeddings, experience_embeddings, metadata = load_index(data_dir)
    check_profile_set(profile_ids)
    queries = load_judgments()

    experience_weight = metadata.get("experience_weight", 0.0) if experience_embeddings is not None else 0.0
    if args.experience_weight is not None:
        if not 0 <= args.experience_weight <= 1:
            parser.error("--experience-weight must be between 0 and 1")
        if args.experience_weight and experience_embeddings is None:
            parser.error("this index has no experience_embeddings.npy; rebuild it with scripts.build_index")
        experience_weight = args.experience_weight

    print(f"Loaded {len(profile_ids)} profiles and {len(queries)} queries "
          f"(experience weight {experience_weight}).")

    model = SentenceTransformer(metadata["model_name"])
    query_embeddings = model.encode(
        [query["query"] for query in queries],
        prompt=metadata["query_prompt"],
        normalize_embeddings=metadata["normalize_embeddings"],
    )

    results = []
    for query, query_embedding in zip(queries, query_embeddings):
        scores = combined_scores(query_embedding, embeddings, experience_embeddings, experience_weight)
        ranking = scores.argsort()[::-1]
        ranked_ids = [profile_ids[index] for index in ranking]
        result = {"query_id": query["query_id"], "query_type": query["query_type"], "query": query["query"]}
        result |= score_query(ranked_ids, query["grades"])
        result["top_10"] = [
            {"profile_id": profile_ids[index], "score": round(float(scores[index]), 4),
             "grade": query["grades"].get(profile_ids[index], 0)}
            for index in ranking[:TOP_K]
        ]
        results.append(result)

    summary = {
        "overall": mean(results),
        "by_query_type": {
            query_type: mean([r for r in results if r["query_type"] == query_type])
            for query_type in sorted({r["query_type"] for r in results})
        },
    }

    print(f"\n{'query':20} {'nDCG@10':>8} {'MRR@10':>7} {'P@5':>5} {'R@10':>5}  top-10 grades")
    for r in results:
        print(f"{r['query_id']:20} {r['ndcg_at_10']:8.3f} {r['mrr_at_10']:7.3f} "
              f"{r['precision_at_5']:5.2f} {r['recall_at_10']:5.2f}  "
              f"{''.join(str(gain) for gain in r['top_10_gains'])}")
    print()
    for label, stats in [("overall", summary["overall"]), *summary["by_query_type"].items()]:
        print(f"{label:10} nDCG@10 = {stats['ndcg_at_10']:.3f}  MRR@10 = {stats['mrr_at_10']:.3f}  "
              f"P@5 = {stats['precision_at_5']:.2f}  R@10 = {stats['recall_at_10']:.2f}  "
              f"(n={stats['queries']})")

    output = {
        "model_name": metadata["model_name"],
        "data_dir": display_path(data_dir),
        "experience_weight": experience_weight,
        "summary": summary,
        "results": results,
    }
    results_path = args.output.resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {display_path(results_path)}")


if __name__ == "__main__":
    main()
