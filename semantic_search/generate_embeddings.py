"""
Generate profile texts and embeddings for the embedding-model experiments.

Run from the project root:

    python -m semantic_search.generate_embeddings          # rebuild only what is stale
    python -m semantic_search.generate_embeddings --force  # rebuild everything

Every output parquet gets a sidecar `<name>.meta.json` with a fingerprint of
its inputs. An output is reused only when that fingerprint still matches:

- text files depend on the sample, the source of profile_text.py and the
  variant options;
- embedding files depend on their text file's fingerprint and the model id.

So editing profile_text.py (or replacing the sample) automatically
invalidates the texts and every embedding built from them.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from semantic_search import profile_text
from semantic_search.profile_text import text_generation


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_PATH = DATA_DIR / "sample_profiles.parquet"
SAMPLE_SIZE = 1000
SAMPLE_SEED = 42

MODELS = {
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "bge_m3": "BAAI/bge-m3",
    "gte_modernbert": "Alibaba-NLP/gte-modernbert-base",
}

VARIANTS = {
    "original": {
        "include_seed_role": True,
        "include_bio": True,
    },
    "no_seed_role": {
        "include_seed_role": False,
        "include_bio": True,
    },
    "no_seed_role_no_bio": {
        "include_seed_role": False,
        "include_bio": False,
    },
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16


def text_path(variant_name):
    return DATA_DIR / f"sample_texts_{variant_name}.parquet"


def embeddings_path(model_name, variant_name):
    return DATA_DIR / f"{model_name}_embeddings_{variant_name}.parquet"


# ==========================================
# FINGERPRINTS (stale-output detection)
# ==========================================

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _meta_path(output_path: Path) -> Path:
    return output_path.with_suffix(".meta.json")


def read_fingerprint(output_path: Path):
    meta = _meta_path(output_path)
    if not output_path.exists() or not meta.exists():
        return None
    try:
        return json.loads(meta.read_text())["fingerprint"]
    except (json.JSONDecodeError, KeyError):
        return None


def write_fingerprint(output_path: Path, fingerprint: str, details: dict):
    _meta_path(output_path).write_text(
        json.dumps({"fingerprint": fingerprint, **details}, indent=2)
    )


def is_fresh(output_path: Path, fingerprint: str, force: bool) -> bool:
    return not force and read_fingerprint(output_path) == fingerprint


def text_fingerprint(sample_hash, profile_text_hash, options):
    payload = json.dumps(
        {
            "sample": sample_hash,
            "profile_text": profile_text_hash,
            "options": options,
        },
        sort_keys=True,
    )
    return _sha256(payload.encode())


def embedding_fingerprint(text_fp, model_id):
    payload = json.dumps({"texts": text_fp, "model": model_id}, sort_keys=True)
    return _sha256(payload.encode())


# ==========================================
# LOAD THE SAME 1000 PROFILES
# ==========================================

def load_sample():
    if SAMPLE_PATH.exists():
        df = pd.read_parquet(SAMPLE_PATH)
        print("Loaded existing sample.")
    else:
        load_dotenv(PROJECT_ROOT / ".env.local")

        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            with conn.cursor() as cursor:
                # Fetch in a stable order, then sample in pandas with a
                # fixed seed, so the sample can be recreated if the parquet
                # is ever lost (same table + same seed = same rows).
                cursor.execute(
                    """
                    SELECT profile_id, seed_role, job_family,
                           skills, profile
                    FROM people
                    ORDER BY profile_id
                    """
                )
                columns = [desc.name for desc in cursor.description]
                df_all = pd.DataFrame(cursor.fetchall(), columns=columns)

        df = df_all.sample(
            n=min(SAMPLE_SIZE, len(df_all)), random_state=SAMPLE_SEED
        ).reset_index(drop=True)

        df.to_parquet(SAMPLE_PATH, index=False)
        print("Created and saved new random sample.")

    print("Profiles loaded:", len(df))
    return df


# ==========================================
# GENERATE TEXT VARIANTS
# ==========================================

def build_texts(df, force):
    """Write one text parquet per variant. Returns {variant: fingerprint}."""

    sample_hash = _sha256(SAMPLE_PATH.read_bytes())
    profile_text_hash = _sha256(inspect.getsource(profile_text).encode())

    rows = df.to_dict("records")
    fingerprints = {}

    for variant_name, options in VARIANTS.items():
        path = text_path(variant_name)
        fp = text_fingerprint(sample_hash, profile_text_hash, options)
        fingerprints[variant_name] = fp

        if is_fresh(path, fp, force):
            print(f"Up to date, skipping: {path.name}")
            continue

        generated = [text_generation(row, **options) for row in rows]

        df_texts = pd.DataFrame(generated)
        df_texts.insert(0, "profile_id", df["profile_id"].tolist())
        df_texts.to_parquet(path, index=False)

        write_fingerprint(path, fp, {"variant": variant_name, "options": options})
        print(f"Saved text variant: {variant_name}")

    return fingerprints


# ==========================================
# EMBEDDING FUNCTION
# ==========================================

def generate_embeddings(model, texts):
    """
    Generate embeddings for non-empty texts.
    Missing texts produce None.
    """

    valid_indices = [
        i for i, text in enumerate(texts)
        if isinstance(text, str) and text.strip()
    ]

    results = [None] * len(texts)

    if not valid_indices:
        return results

    valid_texts = [texts[i] for i in valid_indices]

    # Documents are encoded without a prompt; only queries get the
    # "query" prompt at search time (asymmetric setup, as Qwen3 recommends).
    embeddings = model.encode(
        valid_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    for i, embedding in zip(valid_indices, embeddings):
        results[i] = embedding.astype(np.float32).tolist()

    return results


# ==========================================
# GENERATE MODEL EMBEDDINGS
# ==========================================

def build_embeddings(text_fingerprints, force):
    for model_name, model_id in MODELS.items():

        # Work out what is stale *before* loading the model.
        todo = []
        for variant_name in VARIANTS:
            path = embeddings_path(model_name, variant_name)
            fp = embedding_fingerprint(text_fingerprints[variant_name], model_id)

            if is_fresh(path, fp, force):
                print(f"Up to date, skipping: {path.name}")
            else:
                todo.append((variant_name, path, fp))

        if not todo:
            continue

        print(f"\nLoading model: {model_name}")
        model = SentenceTransformer(model_id, device=DEVICE)

        for variant_name, path, fp in todo:
            print(f"\nGenerating {model_name} embeddings for variant: {variant_name}")

            df_texts = pd.read_parquet(text_path(variant_name))

            results = pd.DataFrame({
                "profile_id": df_texts["profile_id"],
                "core_embedding": generate_embeddings(
                    model, df_texts["core_text"].tolist()
                ),
                "projects_embedding": generate_embeddings(
                    model, df_texts["projects_text"].tolist()
                ),
                "interests_embedding": generate_embeddings(
                    model, df_texts["interests_text"].tolist()
                ),
            })

            results.to_parquet(path, index=False)
            write_fingerprint(
                path, fp, {"model": model_id, "variant": variant_name}
            )
            print(f"Saved: {path.name}")

        del model
        gc.collect()

        if DEVICE == "mps":
            torch.mps.empty_cache()


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate profile texts and embeddings for the experiments."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all texts and embeddings, even if they look up to date.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    df = load_sample()
    text_fps = build_texts(df, force=args.force)
    build_embeddings(text_fps, force=args.force)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
