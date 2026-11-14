"""Intended-profile sanity test for the saved semantic-search index.

Run from the project root with: python -m scripts.evaluate
Evaluate another saved index:   python -m scripts.evaluate --data-dir <folder> --output <file.json>

Each case names one profile the query was written from. The script reports
where that profile ranks; it does not judge the other returned profiles,
which may also be valid matches.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_search.scoring import combined_scores, load_experience_embeddings

TOP_K = 5
QUERY_TYPES = {"direct", "paraphrase", "profile_details", "location"}

project_root = Path(__file__).resolve().parent.parent
default_data_dir = project_root / "data"
cases_path = project_root / "evaluation" / "cases.json"
default_results_path = project_root / "evaluation" / "results" / "intended_profile_results.json"


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def display_path(path):
    """Path relative to the project root when possible, for readable output."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_index(data_dir):
    profiles = load_json(data_dir / "profiles.json")
    embeddings = np.load(data_dir / "embeddings.npy", allow_pickle=False)
    metadata = load_json(data_dir / "index_metadata.json")

    if embeddings.ndim != 2 or len(profiles) != embeddings.shape[0]:
        raise ValueError("Profiles and embeddings do not match.")
    if embeddings.shape[1] != metadata["embedding_dimension"]:
        raise ValueError("Embedding dimensions do not match the metadata.")

    profile_ids = [profile["profile_id"] for profile in profiles]
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("Duplicate profile IDs in profiles.json.")

    if metadata["normalize_embeddings"]:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-2):
            raise ValueError("Saved embeddings are not normalized.")

    experience_embeddings = load_experience_embeddings(data_dir, metadata, embeddings.shape)
    return profile_ids, embeddings, experience_embeddings, metadata


def load_cases(profile_ids):
    cases = load_json(cases_path)["cases"]
    known_ids = set(profile_ids)
    seen_case_ids = set()
    required = {"case_id", "query", "intended_profile_id", "query_type", "support"}

    for case in cases:
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {case.get('case_id')} is missing {sorted(missing)}.")
        if case["case_id"] in seen_case_ids:
            raise ValueError(f"Duplicate case ID: {case['case_id']}")
        seen_case_ids.add(case["case_id"])
        if case["query_type"] not in QUERY_TYPES:
            raise ValueError(f"Unknown query type in {case['case_id']}: {case['query_type']}")
        if not case["query"].strip():
            raise ValueError(f"Empty query in {case['case_id']}.")
        if case["intended_profile_id"] not in known_ids:
            raise ValueError(
                f"Intended profile {case['intended_profile_id']} "
                f"({case['case_id']}) is not in the saved index."
            )

    return cases


def summarize(results):
    count = len(results)
    hits = sum(result["in_top_5"] for result in results)
    reciprocal_ranks = [
        1 / result["intended_rank"] if result["in_top_5"] else 0.0
        for result in results
    ]
    return {
        "cases": count,
        "hit_rate_at_5": hits / count,
        "mrr_at_5": sum(reciprocal_ranks) / count,
    }


def main():
    parser = argparse.ArgumentParser(description="Intended-profile sanity test.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir,
                        help="folder with profiles.json, embeddings.npy and index_metadata.json")
    parser.add_argument("--output", type=Path, default=default_results_path,
                        help="where to write the detailed results JSON")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    results_path = args.output.resolve()

    profile_ids, embeddings, experience_embeddings, metadata = load_index(data_dir)
    experience_weight = metadata.get("experience_weight", 0.0) if experience_embeddings is not None else 0.0
    cases = load_cases(profile_ids)
    print(f"Loaded {len(profile_ids)} profiles and {len(cases)} cases.")

    model = SentenceTransformer(metadata["model_name"])
    query_embeddings = model.encode(
        [case["query"] for case in cases],
        prompt=metadata["query_prompt"],
        normalize_embeddings=metadata["normalize_embeddings"],
    )

    results = []
    for case, query_embedding in zip(cases, query_embeddings):
        # Same ranking as scripts/search.py (full-profile + experience similarity).
        scores = combined_scores(query_embedding, embeddings, experience_embeddings, experience_weight)
        ranking = scores.argsort()[::-1]
        ranked_ids = [profile_ids[index] for index in ranking]

        intended_index = profile_ids.index(case["intended_profile_id"])
        intended_rank = ranked_ids.index(case["intended_profile_id"]) + 1

        results.append({
            "case_id": case["case_id"],
            "query_type": case["query_type"],
            "query": case["query"],
            "intended_profile_id": case["intended_profile_id"],
            "intended_rank": intended_rank,
            "intended_score": round(float(scores[intended_index]), 4),
            "in_top_5": intended_rank <= TOP_K,
            "top_5": [
                {"profile_id": profile_ids[index], "score": round(float(scores[index]), 4)}
                for index in ranking[:TOP_K]
            ],
        })

    summary = {
        "overall": summarize(results),
        "by_query_type": {
            query_type: summarize([r for r in results if r["query_type"] == query_type])
            for query_type in sorted({r["query_type"] for r in results})
        },
    }

    output = {
        "model_name": metadata["model_name"],
        "data_dir": display_path(data_dir),
        "indexed_profiles": len(profile_ids),
        "experience_weight": experience_weight,
        "top_k": TOP_K,
        "summary": summary,
        "results": results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"\n{'case':34} {'rank':>5} {'score':>7}  top-1 profile")
    for result in results:
        print(
            f"{result['case_id']:34} {result['intended_rank']:>5} "
            f"{result['intended_score']:>7.4f}  {result['top_5'][0]['profile_id']}"
        )

    print()
    for label, stats in [("overall", summary["overall"]), *summary["by_query_type"].items()]:
        print(
            f"{label:10} hit rate@5 = {stats['hit_rate_at_5']:.2f}  "
            f"MRR@5 = {stats['mrr_at_5']:.3f}  (n={stats['cases']})"
        )
    print(f"\nDetailed results saved to {display_path(results_path)}")


if __name__ == "__main__":
    main()
