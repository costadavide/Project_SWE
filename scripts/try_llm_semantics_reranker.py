import argparse
import json

from scripts.ask_llm import ask, warm_up
from scripts.config import DIR, FILTERS, K, POOL, WEIGHTS
from scripts.try_semantics import get_text, load_lines, print_result
from semantic_search.filters import build_allowed, load_metadata, load_skills_index
from semantic_search.reranker import Reranker
from semantic_search.searcher import SemanticSearcher


def understand_query(query, metadata, rewrite, skills_index=None):
    """Return (the text to search with, the allowed rows or None for no filter).

    With rewrite, the text is the query rewritten by the LLM. The original query
    is kept if the LLM cannot be reached or gives no usable answer."""
    try:
        answer = ask(query)
    except (OSError, ValueError) as error:
        print(f"Warning: no LLM, searching the original query without filters ({error})")
        return query, None

    text = query
    if rewrite:
        text = (answer["semantic_query"] or "").strip() or query
        print(f"Searching for: {text}")

    constraints = answer["hard_constraints"]
    try:
        allowed = build_allowed(metadata, constraints, skills_index)
    except (ValueError, TypeError, AttributeError) as error:
        print(f"Warning: searching without filters ({error})")
        return text, None

    filters = {name: constraints[name] for name in FILTERS if constraints[name] not in (None, [])}
    left = len(metadata) if allowed is None else len(allowed)
    print(f"Filters: {filters or 'none'} -> {left:,} of {len(metadata):,} profiles")
    return text, allowed


def run_query(searcher, reranker, lines, metadata, query, k, rewrite=False, skills_index=None):
    text, allowed = understand_query(query, metadata, rewrite, skills_index)
    if allowed is not None and not allowed:
        print("No profile passes these filters.")
        return

    profiles, breakdowns = searcher.search(text, POOL, allowed)
    texts = {profile_id: get_text(lines[profile_id]) for profile_id, score in profiles}
    reranked, details = reranker.rerank(text, profiles, texts, k)

    scores = dict(profiles)
    parts = dict(breakdowns)
    details = dict(details)
    for rank, (profile_id, _) in enumerate(reranked, start=1):
        row = json.loads(lines[profile_id])
        print_result(rank, scores[profile_id], parts[profile_id], searcher.weights, row, details[profile_id])


def main():
    parser = argparse.ArgumentParser(description="Search with LLM filters, embeddings and a reranker.")
    parser.add_argument("--rewrite", action="store_true",
                        help="search with the query rewritten by the LLM instead of the original")
    args = parser.parse_args()

    print("Loading...")
    searcher = SemanticSearcher(DIR, weights=WEIGHTS)
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
        run_query(searcher, reranker, lines, metadata, query, k=K, rewrite=args.rewrite, skills_index=skills_index)


if __name__ == "__main__":
    main()
