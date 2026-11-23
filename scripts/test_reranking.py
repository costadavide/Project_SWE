"""
Interactive check of cross-encoder reranking on top of embedding retrieval.

For each query it retrieves a candidate pool with the bi-encoder, reranks the
whole pool once, and shows three top-k lists:

    1. Retrieval  - bi-encoder order
    2. Reranker   - cross-encoder score only        (fusion="none")
    3. RRF        - retrieval rank + reranker rank  (fusion="rrf")

Run from the project root:

    python -m scripts.test_reranking
"""

from sentence_transformers import SentenceTransformer

from scripts.test_embeddings import MODELS, VARIANTS, device, search
from semantic_search.reranker import DEFAULT_RERANKER, load_reranker, rerank_profiles


# ==========================================
# CONFIGURATION
# ==========================================

DEFAULT_MODEL = "qwen3"
DEFAULT_VARIANT = "no_seed_role"   # same as scripts/eval_embeddings.py
DEFAULT_MODE = "core"
DEFAULT_POOL = 50                  # candidates retrieved and reranked
DEFAULT_TOP_K = 10                 # rows shown per list


# ==========================================
# INPUT HELPERS
# ==========================================

def ask_choice(prompt, choices, default):
    while True:
        value = input(f"{prompt} {list(choices)} [{default}]: ").strip() or default
        if value in choices:
            return value
        print(f"  Unknown value {value!r}, pick one of {list(choices)}")


def ask_int(prompt, default, minimum=1):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if value < minimum:
            print(f"  Must be at least {minimum}.")
            continue
        return value


# ==========================================
# RANKING
# ==========================================

def rank_all(query, candidates, reranker):
    """
    Score the whole pool once and return it with a rank column per method.
    """
    # top_k=len(...) keeps every candidate, with retrieval_rank,
    # reranker_rank and rrf_score filled in for all of them.
    ranked = rerank_profiles(
        query,
        candidates,
        reranker,
        top_k=len(candidates),
        fusion="none",
    )

    # Same ordering rerank_profiles uses for fusion="rrf":
    # RRF score, ties broken by retrieval order.
    rrf_order = ranked.sort_values(
        ["rrf_score", "retrieval_rank"],
        ascending=[False, True],
    ).index
    ranked.loc[rrf_order, "rrf_rank"] = range(1, len(ranked) + 1)
    ranked["rrf_rank"] = ranked["rrf_rank"].astype(int)

    return ranked


def movement(new_rank, retrieval_rank, top_k):
    if retrieval_rank > top_k:
        return f"NEW (was #{retrieval_rank})"
    delta = retrieval_rank - new_rank
    if delta > 0:
        return f"↑{delta}"
    if delta < 0:
        return f"↓{-delta}"
    return "="


# ==========================================
# DISPLAY
# ==========================================

def print_list(title, ranked, rank_column, top_k, show_movement):
    print(f"\n--- {title} ---")
    rows = ranked.sort_values(rank_column).head(top_k)

    for _, row in rows.iterrows():
        rank = row[rank_column]
        line = (
            f"{rank:>3}. {row['profile_id']:<34} "
            f"{str(row['seed_role'])[:32]:<32} "
            f"ret={row['final_score']:.3f}  rr={row['reranker_score']:+.3f}"
        )
        if show_movement:
            line += f"  {movement(rank, row['retrieval_rank'], top_k)}"
        print(line)


def print_summary(ranked, top_k):
    retrieval_top = set(ranked.loc[ranked["retrieval_rank"] <= top_k, "profile_id"])

    print("\n--- Summary ---")
    for name, column in [("Reranker", "reranker_rank"), ("RRF", "rrf_rank")]:
        method_top = set(ranked.loc[ranked[column] <= top_k, "profile_id"])
        new = len(method_top - retrieval_top)
        print(
            f"{name:<9} top {top_k}: {top_k - new} kept from retrieval, "
            f"{new} brought in from ranks {top_k + 1}-{len(ranked)}"
        )


def print_details(ranked, key):
    """key is a profile ID or a retrieval rank number."""
    if key.isdigit():
        match = ranked[ranked["retrieval_rank"] == int(key)]
    else:
        match = ranked[ranked["profile_id"] == key]

    if match.empty:
        print(f"  No candidate {key!r} in this pool.")
        return

    row = match.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"{row['profile_id']}  ({row['seed_role']})")
    print(
        f"Ranks  retrieval #{row['retrieval_rank']}  "
        f"reranker #{row['reranker_rank']}  RRF #{row['rrf_rank']}"
    )
    print(
        f"Scores retrieval {row['final_score']:.4f}  "
        f"reranker {row['reranker_score']:+.4f}  RRF {row['rrf_score']:.5f}"
    )
    for column in ("core_text", "projects_text", "interests_text"):
        text = row[column]
        if isinstance(text, str) and text.strip():
            print(f"\n[{column}]\n{text}")
    print("=" * 60)


# ==========================================
# MAIN
# ==========================================

def main():
    model_name = ask_choice("Embedding model", MODELS, DEFAULT_MODEL)
    variant = ask_choice("Variant", VARIANTS, DEFAULT_VARIANT)
    mode = ask_choice("Mode", ["core", "weighted"], DEFAULT_MODE)
    pool = ask_int("Candidate pool size", DEFAULT_POOL)
    top_k = ask_int("Results to show", min(DEFAULT_TOP_K, pool))
    top_k = min(top_k, pool)

    print(f"\nLoading embedding model: {MODELS[model_name]}")
    model = SentenceTransformer(MODELS[model_name], device=device)

    print(f"Loading reranker: {DEFAULT_RERANKER}")
    reranker = load_reranker()

    while True:
        query = input("\nEnter a search query (empty to quit): ").strip()
        if not query:
            break

        candidates = search(
            query,
            model_name,
            mode=mode,
            variant=variant,
            model=model,
            top_k=pool,
        )
        ranked = rank_all(query, candidates, reranker)

        print_list("1. Retrieval (bi-encoder)", ranked, "retrieval_rank", top_k, False)
        print_list("2. Reranker only", ranked, "reranker_rank", top_k, True)
        print_list("3. RRF (retrieval + reranker)", ranked, "rrf_rank", top_k, True)
        print_summary(ranked, top_k)

        while True:
            key = input(
                "\nProfile ID or retrieval # for full text (Enter for next query): "
            ).strip()
            if not key:
                break
            print_details(ranked, key)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
