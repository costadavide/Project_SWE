"""Search the full corpus with LLM filters, fusion (BM25 + embeddings) and a reranker.

Same interface as try_llm_semantics_reranker.py, but the retrieval step is
search_engine/fusion.py: BM25 on the full lexical index (data/full/lexical.pkl)
and the semantic searcher on the full embeddings (data/full), merged with
reciprocal rank fusion. Then, like the semantic script:

    1. the LLM reads the query and gives hard constraints (location, years of
       experience, excluded skills), turned into the allowed rows by build_allowed;
    2. fusion retrieves POOL profiles among the allowed ones;
    3. the cross-encoder reranks them and the top K are printed, with each
       profile's rank in BM25 and in the semantic search.

Row i of data/full/profiles.jsonl must be doc_id i in lexical.pkl and row i of
the embeddings; this is checked once at start-up.

Run from the project root, with LM Studio on (without it the query is searched
as it is, with no filters):

    python -m scripts.try_bm25
    python -m scripts.try_bm25 --rewrite        # search with the query rewritten by the LLM
    python -m scripts.try_bm25 --no-rerank      # show the fusion order, without the reranker
    python -m scripts.try_bm25 --lexical bm25f  # BM25F instead of BM25 in the fusion

The old version of this script (toy checks, phrase filters, BM25 vs BM25F on
the 10k corpus) is in the git history.
"""

import argparse
import json

from scripts.ask_llm import warm_up
from scripts.config import DEPTH, DIR, INDEX_PATH, K, POOL, TEXT_OPTIONS, WEIGHTS
from scripts.try_llm_semantics_reranker import understand_query
from scripts.try_semantics import get_text, load_lines
from search_engine.bm25 import BM25, BM25F
from search_engine.fusion import Fusion
from search_engine.index import Index
from semantic_search.filters import load_metadata, load_skills_index
from semantic_search.profile_text import text_generation
from semantic_search.reranker import Reranker
from semantic_search.searcher import SemanticSearcher


def print_result(rank, score, ranks, methods, row, reranked=None):
    print(f"\n{rank}.")
    print(f"   Profile ID: {row['profile_id']}")
    if reranked is not None:
        print(f"   Rank before reranking: {reranked['retrieval_rank']}")
        print(f"   Reranker score: {reranked['reranker_score']:.4f}")
    print(f"   Fusion score: {score:.4f}")
    # ranks = {method: (rank, contribution)}; a missing method did not return the profile
    for method in methods:
        if method in ranks:
            print(f"   {method} rank: {ranks[method][0]}")
        else:
            print(f"   {method} rank: not in its top {DEPTH}")

    print("\n   Profile:")
    print(text_generation(row, **TEXT_OPTIONS)["core_text"])
    print("-" * 60)


def run_query(fusion, reranker, lines, metadata, query, k, rewrite=False, skills_index=None):
    text, allowed = understand_query(query, metadata, rewrite, skills_index)
    if allowed is not None and not allowed:
        print("No profile passes these filters.")
        return

    profiles, breakdowns = fusion.search(text, POOL, allowed)
    if reranker is not None:
        texts = {profile_id: get_text(lines[profile_id]) for profile_id, score in profiles}
        shown, details = reranker.rerank(text, profiles, texts, k)
        details = dict(details)
    else:
        shown, details = profiles[:k], {}

    scores = dict(profiles)
    ranks = dict(breakdowns)
    for rank, (profile_id, _) in enumerate(shown, start=1):
        row = json.loads(lines[profile_id])
        print_result(rank, scores[profile_id], ranks[profile_id], fusion.searchers, row, details.get(profile_id))


def main():
    parser = argparse.ArgumentParser(description="Search with LLM filters, fusion (BM25 + embeddings) and a reranker.")
    parser.add_argument("--rewrite", action="store_true",
                        help="search with the query rewritten by the LLM instead of the original")
    parser.add_argument("--no-rerank", action="store_true", help="show the fusion order, without the reranker")
    parser.add_argument("--lexical", choices=["bm25", "bm25f"], default="bm25",
                        help="lexical method in the fusion (default bm25, the better one on the judged set)")
    args = parser.parse_args()

    print("Loading...")
    index = Index.load(INDEX_PATH)
    semantic = SemanticSearcher(DIR, weights=WEIGHTS)
    semantic.check_alignment(index)  # doc_id i in the index = row i of the embeddings
    semantic.load_model()
    if args.lexical == "bm25":
        lexical = BM25(index)
    else:
        lexical = BM25F(index)
    # the key "semantic" matters: Fusion breaks ties on the semantic rank
    fusion = Fusion({args.lexical: lexical, "semantic": semantic}, method="rrf", depth=DEPTH)

    reranker = None
    if not args.no_rerank:
        reranker = Reranker()
        reranker.load_model()
    warm_up()  # the LLM
    lines = load_lines()
    metadata = load_metadata(DIR / "profiles.jsonl")
    skills_index = load_skills_index(DIR / "profiles.jsonl")
    print(f"Ready: {len(index):,} profiles.")

    while True:
        query = input("\nQuery (or 'exit'): ").strip()
        if query == "exit":
            break
        if not query:
            continue
        run_query(fusion, reranker, lines, metadata, query, k=K, rewrite=args.rewrite, skills_index=skills_index)


if __name__ == "__main__":
    main()
