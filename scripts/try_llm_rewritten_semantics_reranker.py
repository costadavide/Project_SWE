"""Same as try_llm_semantics_reranker.py, but the query that is embedded is the one
rewritten by the LLM. The reranker still reads the original query.

Run from the project root, with LM Studio on:

    python -m scripts.try_llm_rewritten_semantics_reranker
"""

import json

from scripts.ask_llm import warm_up
from scripts.try_llm_semantics_reranker import K, POOL, understand_query
from scripts.try_semantics import DIR, get_text, load_lines, print_result
from semantic_search.filters import load_metadata, load_skills_index
from semantic_search.reranker import Reranker
from semantic_search.searcher import SemanticSearcher


def run_query(searcher, reranker, lines, metadata, query, k, skills_index=None):
    # the rewritten query (or the original, if the LLM gives no usable answer)
    rewritten, allowed = understand_query(query, metadata, rewrite=True, skills_index=skills_index)
    if allowed is not None and not allowed:
        print("No profile passes these filters.")
        return

    profiles, breakdowns = searcher.search(rewritten, POOL, allowed)
    texts = {profile_id: get_text(lines[profile_id]) for profile_id, score in profiles}
    reranked, details = reranker.rerank(query, profiles, texts, k)

    scores = dict(profiles)
    parts = dict(breakdowns)
    details = dict(details)
    for rank, (profile_id, _) in enumerate(reranked, start=1):
        row = json.loads(lines[profile_id])
        print_result(rank, scores[profile_id], parts[profile_id], searcher.weights, row, details[profile_id])


def main():
    print("Loading...")
    weights = {"core": 0.80,
        "projects": 0.15,
        "interests": 0.05,}
    searcher = SemanticSearcher(DIR, weights=weights)
    reranker = Reranker()
    reranker.load_model()
    searcher.load_model()
    warm_up()  # the LLM
    lines = load_lines()
    metadata = load_metadata(DIR / "profiles.jsonl")
    skills_index = load_skills_index(DIR / "profiles.jsonl")

    while True:
        query = input("\nQuery (or 'exit'): ").strip()
        if query == "exit":
            break
        if not query:
            continue
        run_query(searcher, reranker, lines, metadata, query, k=K, skills_index=skills_index)


if __name__ == "__main__":
    main()
