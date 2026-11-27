"""The final search engine, with switches for the other pipelines.

By default:

    1. the LLM reads the query: it rewrites it and gives hard constraints
       (location, years of experience, excluded skills), turned into the
       allowed rows by build_allowed;
    2. the semantic searcher (weights in scripts/config.py) retrieves POOL
       profiles among the allowed ones, with the rewritten query;
    3. the cross-encoder reranks them with the original query and the top K
       are printed.

The shared settings (K, POOL, DEPTH, WEIGHTS, paths) are in scripts/config.py.

Run from the project root, with LM Studio on (without it the query is searched
as it is, with no filters):

    python -m scripts.search_cli                          # the default above
    python -m scripts.search_cli --retrieval fusion       # BM25 + embeddings (RRF) instead of embeddings only
    python -m scripts.search_cli --retrieval fusion --lexical bm25f
    python -m scripts.search_cli --query original         # retrieve with the original query
    python -m scripts.search_cli --rerank-query rewritten # the reranker reads the rewritten query
    python -m scripts.search_cli --no-rerank              # show the retrieval order
    python -m scripts.search_cli --no-llm                 # no LLM: original query, no filters
    python -m scripts.search_cli --k 10 --pool 200        # profiles shown / given to the reranker

To reproduce the older scripts:

    try_llm_rewritten_semantics_reranker.py  (no flags)
    try_llm_semantics_reranker.py            --query original
    try_llm_semantics_reranker.py --rewrite  --rerank-query rewritten
    try_bm25.py                              --retrieval fusion --query original
    try_bm25.py --rewrite                    --retrieval fusion --rerank-query rewritten
    try_semantics.py                         --no-llm
"""

import argparse
import json

from scripts.ask_llm import ask, warm_up
from scripts.config import DEPTH, DIR, FILTERS, INDEX_PATH, K, POOL, TEXT_OPTIONS, WEIGHTS
from search_engine.bm25 import BM25, BM25F
from search_engine.fusion import Fusion
from search_engine.index import Index
from semantic_search.filters import build_allowed, load_metadata, load_skills_index
from semantic_search.profile_text import text_generation
from semantic_search.reranker import Reranker
from semantic_search.searcher import SemanticSearcher


# ==========================================
# ARGUMENTS
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Search people: LLM filters and rewrite, retrieval, reranker.")
    parser.add_argument("--retrieval", choices=["semantic", "fusion"], default="semantic",
                        help="semantic: embeddings only (default); fusion: BM25 + embeddings with RRF")
    parser.add_argument("--lexical", choices=["bm25", "bm25f"],
                        help="lexical method in the fusion (default bm25); only with --retrieval fusion")
    parser.add_argument("--query", choices=["rewritten", "original"], default="rewritten",
                        help="the query given to the retrieval (default rewritten)")
    parser.add_argument("--rerank-query", choices=["original", "rewritten"], default="original",
                        help="the query the reranker reads (default original)")
    parser.add_argument("--no-rerank", action="store_true", help="show the retrieval order, without the reranker")
    parser.add_argument("--no-llm", action="store_true",
                        help="do not call the LLM: search the original query with no filters "
                             "(--query and --rerank-query are then ignored)")
    parser.add_argument("--k", type=int, default=K, help=f"profiles shown (default {K})")
    parser.add_argument("--pool", type=int, default=POOL,
                        help=f"profiles retrieved and given to the reranker (default {POOL})")
    args = parser.parse_args()

    if args.k < 1 or args.pool < 1:
        parser.error("--k and --pool must be at least 1")
    if args.lexical is not None and args.retrieval != "fusion":
        parser.error("--lexical only works with --retrieval fusion")
    if args.lexical is None:
        args.lexical = "bm25"
    return args


# ==========================================
# PROFILES
# ==========================================

def load_lines():
    """Return {profile_id: line of the file} for every profile."""
    lines = {}
    with (DIR / "profiles.jsonl").open("rb") as file:
        for line in file:
            lines[json.loads(line)["profile_id"]] = line
    return lines


def get_text(line):
    """The text the reranker reads: core, projects and interests."""
    sections = text_generation(json.loads(line), **TEXT_OPTIONS).values()
    return "\n\n".join(section for section in sections if section)


# ==========================================
# QUERY UNDERSTANDING (LLM)
# ==========================================

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
        print(f"Rewritten query: {text}")

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


def queries_for(query, args, metadata, skills_index):
    """Return (retrieval query, reranker query, allowed rows or None for no filter)."""
    if args.no_llm:
        return query, query, None

    needs_rewrite = args.query == "rewritten" or args.rerank_query == "rewritten"
    rewritten, allowed = understand_query(query, metadata, rewrite=needs_rewrite, skills_index=skills_index)
    retrieval_query = rewritten if args.query == "rewritten" else query
    rerank_query = rewritten if args.rerank_query == "rewritten" else query
    return retrieval_query, rerank_query, allowed


# ==========================================
# RETRIEVAL
# ==========================================

def build_searcher(args):
    """The retrieval step: an object with search(query, k, allowed) -> (profiles, breakdowns)."""
    semantic = SemanticSearcher(DIR, weights=WEIGHTS)
    if args.retrieval == "semantic":
        semantic.load_model()
        return semantic

    index = Index.load(INDEX_PATH)
    semantic.check_alignment(index)  # doc_id i in the index = row i of the embeddings
    semantic.load_model()
    lexical = BM25(index) if args.lexical == "bm25" else BM25F(index)
    # the key "semantic" matters: Fusion breaks ties on the semantic rank
    return Fusion({args.lexical: lexical, "semantic": semantic}, method="rrf", depth=DEPTH)


# ==========================================
# OUTPUT
# ==========================================

def print_result(rank, score, breakdown, searcher, row, reranked=None):
    """breakdown is {part: weight * similarity} for the semantic searcher and
    {method: (rank, contribution)} for the fusion."""
    print(f"\n{rank}.")
    print(f"   Profile ID: {row['profile_id']}")
    if reranked is not None:
        print(f"   Rank before reranking: {reranked['retrieval_rank']}")
        print(f"   Reranker score: {reranked['reranker_score']:.4f}")

    if isinstance(searcher, Fusion):
        print(f"   Fusion score: {score:.4f}")
        for method in searcher.searchers:
            if method in breakdown:
                print(f"   {method} rank: {breakdown[method][0]}")
            else:  # the method did not return the profile
                print(f"   {method} rank: not in its top {DEPTH}")
    else:
        print(f"   Final score: {score:.4f}")
        for kind, part in breakdown.items():
            # parts are weight * similarity: divide to get the similarity back
            print(f"   {kind.capitalize()}: {part / searcher.weights[kind]:.4f}")

    print("\n   Profile:")
    print(text_generation(row, **TEXT_OPTIONS)["core_text"])
    print("-" * 60)


def describe(args):
    if args.retrieval == "fusion":
        retrieval = f"fusion ({args.lexical} + semantic)"
    else:
        retrieval = "semantic"
    if args.no_llm:
        llm = "no LLM (original query, no filters)"
    else:
        llm = f"LLM filters, retrieval reads the {args.query} query"
        if not args.no_rerank:
            llm += f", reranker reads the {args.rerank_query} query"
    rerank = "no reranker" if args.no_rerank else f"reranker on the top {args.pool}"
    return f"{retrieval}; {llm}; {rerank}; showing {args.k}"


# ==========================================
# SEARCH
# ==========================================

def run_query(searcher, reranker, lines, metadata, query, args, skills_index=None):
    retrieval_query, rerank_query, allowed = queries_for(query, args, metadata, skills_index)
    if allowed is not None and not allowed:
        print("No profile passes these filters.")
        return

    if reranker is not None:
        profiles, breakdowns = searcher.search(retrieval_query, args.pool, allowed)
        texts = {profile_id: get_text(lines[profile_id]) for profile_id, score in profiles}
        shown, details = reranker.rerank(rerank_query, profiles, texts, args.k)
        details = dict(details)
    else:
        profiles, breakdowns = searcher.search(retrieval_query, args.k, allowed)
        shown, details = profiles, {}

    scores = dict(profiles)
    parts = dict(breakdowns)
    for rank, (profile_id, _) in enumerate(shown, start=1):
        row = json.loads(lines[profile_id])
        print_result(rank, scores[profile_id], parts[profile_id], searcher, row, details.get(profile_id))


def main():
    args = parse_args()

    print("Loading...")
    searcher = build_searcher(args)
    reranker = None
    if not args.no_rerank:
        reranker = Reranker()
        reranker.load_model()
    if not args.no_llm:
        warm_up()  # the LLM
    lines = load_lines()
    metadata = load_metadata(DIR / "profiles.jsonl")
    skills_index = load_skills_index(DIR / "profiles.jsonl")
    print(f"Ready: {describe(args)}.")

    while True:
        query = input("\nQuery (or 'exit'): ").strip()
        if query == "exit":
            break
        if not query:
            continue
        run_query(searcher, reranker, lines, metadata, query, args, skills_index=skills_index)


if __name__ == "__main__":
    main()
