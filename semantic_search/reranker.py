"""
Cross-encoder reranking of retrieved profiles.

Typical use, on top of scripts/test_embeddings.search():

    reranker = load_reranker()
    candidates = search(query, "qwen3", ...)          # retrieve a wide pool
    top = rerank_profiles(query, candidates, reranker, top_k=10)

Rerank a pool larger than the final top_k (e.g. 50 -> 10). Reranking only
the top 10 just shuffles them and can never surface anything new.
"""

from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder


# Multilingual, long-context reranker from the same family as bge-m3.
# Profile texts here are ~1.6k chars on average and up to ~4k chars
# (~1k tokens), which a 512-token model like ms-marco-MiniLM would truncate.
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LENGTH = 1024

# Same fields the bi-encoder retrieval scores, so the reranker judges
# what retrieval actually matched on.
DEFAULT_TEXT_COLUMNS = ("core_text", "projects_text", "interests_text")

FUSION_METHODS = ("rrf", "none")


def load_reranker(model_name=DEFAULT_RERANKER, device=None, max_length=DEFAULT_MAX_LENGTH):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    return CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
    )


def _join_texts(row, text_columns):
    parts = []
    for column in text_columns:
        value = row[column]
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def rerank_profiles(
    query,
    candidates,
    reranker,
    top_k=10,
    text_columns=DEFAULT_TEXT_COLUMNS,
    score_column="final_score",
    fusion="rrf",
    rrf_k=60,
    batch_size=32,
):
    """
    Rerank retrieval candidates with a cross-encoder.

    Args:
        query: the search query.
        candidates: DataFrame of retrieved profiles.
        reranker: a loaded CrossEncoder (see load_reranker).
        top_k: number of rows to return.
        text_columns: column name or list of column names joined to form the
            text the reranker reads.
        score_column: retrieval score used to set `retrieval_rank`
            (higher = better). If it is None or not in `candidates`, the
            input row order is taken as the retrieval order.
        fusion: "rrf" sorts by reciprocal rank fusion of the retrieval rank
            and the reranker rank; "none" sorts by the reranker score alone.
        rrf_k: RRF smoothing constant (60 is the usual default).
        batch_size: cross-encoder batch size.

    Returns:
        The top_k rows with extra columns `retrieval_rank`, `reranker_score`,
        `reranker_rank`, `rrf_score` and `final_rank`.
    """
    if isinstance(text_columns, str):
        text_columns = (text_columns,)
    elif not isinstance(text_columns, Sequence) or not text_columns:
        raise ValueError("text_columns must be a column name or a non-empty list")

    missing = [c for c in text_columns if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")

    if fusion not in FUSION_METHODS:
        raise ValueError(f"fusion must be one of {FUSION_METHODS}, got {fusion!r}")

    if candidates.empty:
        return candidates.copy()

    results = candidates.copy()

    # Retrieval order: from the score if we have one, else the input order.
    if score_column is not None and score_column in results.columns:
        results = results.sort_values(score_column, ascending=False, kind="stable")
    results = results.reset_index(drop=True)
    results["retrieval_rank"] = np.arange(1, len(results) + 1)

    # Cross-encoder relevance scores: one (query, profile) pair per candidate
    pairs = [
        (query, _join_texts(row, text_columns))
        for _, row in results.iterrows()
    ]
    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    results["reranker_score"] = np.asarray(scores, dtype=np.float32).reshape(-1)

    # Rank 1 = highest reranker score; ties keep retrieval order.
    reranker_order = np.argsort(-results["reranker_score"].to_numpy(), kind="stable")
    reranker_rank = np.empty(len(results), dtype=int)
    reranker_rank[reranker_order] = np.arange(1, len(results) + 1)
    results["reranker_rank"] = reranker_rank

    results["rrf_score"] = (
        1.0 / (rrf_k + results["retrieval_rank"])
        + 1.0 / (rrf_k + results["reranker_rank"])
    )

    sort_key = "rrf_score" if fusion == "rrf" else "reranker_score"
    results = results.sort_values(sort_key, ascending=False, kind="stable").reset_index(drop=True)
    results["final_rank"] = np.arange(1, len(results) + 1)

    return results.head(top_k).copy()
