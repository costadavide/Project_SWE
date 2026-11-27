import argparse
import json
import time
from scripts.config import DIR, K, POOL, TEXT_OPTIONS, WEIGHTS
from semantic_search.profile_text import text_generation
from semantic_search.searcher import SemanticSearcher
from semantic_search.reranker import Reranker


def load_lines():
    """Return {profile_id: line of the file} for every profile."""
    lines = {}
    with (DIR / "profiles.jsonl").open("rb") as file:
        for line in file:
            lines[json.loads(line)["profile_id"]] = line
    return lines


def print_result(rank, score, parts, weights, row, reranked=None):
    print(f"\n{rank}.")
    print(f"   Profile ID: {row['profile_id']}")
    if reranked is not None:
        print(f"   Rank before reranking: {reranked['retrieval_rank']}")
        print(f"   Reranker score: {reranked['reranker_score']:.4f}")
    print(f"   Final score: {score:.4f}")
    for kind, part in parts.items():
        # parts are weight * similarity: divide to get the similarity back
        print(f"   {kind.capitalize()}: {part / weights[kind]:.4f}")

    print("\n   Profile:")
    print(text_generation(row, **TEXT_OPTIONS)["core_text"])
    print("-" * 60)


def get_text(line):
    """The text the reranker reads: core, projects and interests."""
    sections = text_generation(json.loads(line), **TEXT_OPTIONS).values()
    return "\n\n".join(section for section in sections if section)


def run_query(searcher, lines, query, k, reranker=None):
    if reranker is not None:
        profiles, breakdowns = searcher.search(query, POOL)
        texts = {profile_id: get_text(lines[profile_id]) for profile_id, score in profiles}
        reranked, details = reranker.rerank(query, profiles, texts, k)
        details = dict(details)
    else:
        profiles, breakdowns = searcher.search(query, k)
        reranked, details = profiles, {}

    scores = dict(profiles)
    parts = dict(breakdowns)
    for rank, (profile_id, _) in enumerate(reranked, start=1):
        row = json.loads(lines[profile_id])
        print_result(rank, scores[profile_id], parts[profile_id], searcher.weights, row, details.get(profile_id))


def main():
    print("Loading...")
    searcher = SemanticSearcher(DIR, weights=WEIGHTS)
    reranker = Reranker()
    reranker.load_model()
    searcher.load_model()
    lines = load_lines()

    while True:
        query = input("\nQuery (or 'exit'): ").strip()
        if query == "exit":
            break
        if not query:
            continue
        run_query(searcher, lines, query, k=K, reranker=reranker)



if __name__ == "__main__":
    main()
