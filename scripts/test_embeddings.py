
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MODELS = {
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "bge_m3": "BAAI/bge-m3",
    "gte_modernbert": "Alibaba-NLP/gte-modernbert-base",
}

VARIANTS = [
    "original",
    "no_seed_role",
    "no_seed_role_no_bio"
]

WEIGHTS = {
    "core": 0.80,
    "projects": 0.15,
    "interests": 0.05,
}

TOP_K = 10

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ==========================================
# LOAD DATA
# ==========================================

# File names must match semantic_search/generate_embeddings.py, which writes
# one file per variant (including "original").

def load_embeddings(model_name, variant):
    path = DATA_DIR / f"{model_name}_embeddings_{variant}.parquet"
    return pd.read_parquet(path)

def load_texts(variant):
    path = DATA_DIR / f"sample_texts_{variant}.parquet"
    return pd.read_parquet(path)

df_profiles = pd.read_parquet(DATA_DIR / "sample_profiles.parquet")

# ==========================================
# PREPARE EMBEDDING MATRICES
# ==========================================

def prepare_matrix(series, dimensions):
    """
    Convert a column of embeddings to a NumPy matrix.
    Missing embeddings become zero vectors.
    """

    matrix = np.zeros(
        (len(series), dimensions),
        dtype=np.float32
    )

    for i, embedding in enumerate(series):
        if embedding is not None:
            matrix[i] = np.asarray(
                embedding, dtype=np.float32
            )

    return matrix


# ==========================================
# SIMILARITY SEARCH
# ==========================================

def search(query, model_name, mode="weighted", variant="original", model=None, top_k=TOP_K):

    if model is None:
        model = SentenceTransformer(
            MODELS[model_name],
            device=device
        )
    # Qwen3 recommends a query instruction
    if model_name == "qwen3":
        query_embedding = model.encode(
            query,
            prompt_name="query",
            normalize_embeddings=True
        )
    else:
        query_embedding = model.encode(
            query,
            normalize_embeddings=True
        )

    dimensions = len(query_embedding)

    df = load_embeddings(model_name, variant)
    df_texts = load_texts(variant)

    # Ensure embeddings and profile texts use the same IDs
    df = df.merge(
        df_texts,
        on="profile_id",
        validate="one_to_one"
    )

    df = df.merge(
        df_profiles[["profile_id", "seed_role"]],
        on="profile_id",
        validate="one_to_one"
    )

    # Build embedding matrices
    core = prepare_matrix(
        df["core_embedding"], dimensions
    )

    projects = prepare_matrix(
        df["projects_embedding"], dimensions
    )

    interests = prepare_matrix(
        df["interests_embedding"], dimensions
    )

    # Cosine similarity (vectors are normalized)
    core_scores = core @ query_embedding
    project_scores = projects @ query_embedding
    interest_scores = interests @ query_embedding

    # Missing embeddings contribute zero
    if mode == "core":
        final_scores = core_scores
    elif mode == "weighted":
        final_scores = (
            WEIGHTS["core"] * core_scores
            + WEIGHTS["projects"] * project_scores
            + WEIGHTS["interests"] * interest_scores
        )
    else:
        raise ValueError("Mode must be 'core' or 'weighted'")

    # Get top candidates
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = df.iloc[top_indices][[
        "profile_id",
        "seed_role",
        "core_text",
        "projects_text",
        "interests_text"
    ]].copy()

    results["core_score"] = core_scores[top_indices]
    results["project_score"] = project_scores[top_indices]
    results["interest_score"] = interest_scores[top_indices]
    results["final_score"] = final_scores[top_indices]

    return results


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(results, model_name):

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")

    for rank, (_, row) in enumerate(results.iterrows(), 1):

        print(f"\n{rank}. {row['seed_role']}")
        print(f"   Profile ID: {row['profile_id']}")
        print(f"   Final score: {row['final_score']:.4f}")
        print(f"   Core: {row['core_score']:.4f}")
        print(f"   Projects: {row['project_score']:.4f}")
        print(f"   Interests: {row['interest_score']:.4f}")

        print("\n   Profile:")
        print(row["core_text"])
        print("-" * 60)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    query = input("Enter your search query: ")

    mode = (
        input("Mode (core/weighted) [core]: ").strip()
        or "core"
    )

    print("\nAvailable variants:")
    for variant in VARIANTS:
        print(f"- {variant}")

    variant = (
        input("\nChoose variant [original]: ").strip()
        or "original"
    )

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    for model_name in MODELS:

        results = search(
            query,
            model_name,
            variant=variant,
            mode=mode
        )

        display_results(results, model_name)
