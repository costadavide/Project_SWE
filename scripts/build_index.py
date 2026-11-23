"""Build the saved search index from the profiles in the PostgreSQL database (AWS RDS).

Usage:
    python -m scripts.build_index                              # photographer + 99 random profiles
    python -m scripts.build_index --ids data/profiles.json     # same profiles as the saved index
    python -m scripts.build_index --sample-size 1000 --data-dir data/sample_1000
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from semantic_search.profile_text1 import experience_to_text, profile_to_text
from semantic_search.scoring import DEFAULT_EXPERIENCE_WEIGHT
from sentence_transformers import SentenceTransformer
import json
import numpy as np


project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env.local"
load_dotenv(env_path)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--ids",
    type=Path,
    help="profiles.json from an earlier index; rebuild with exactly those profile IDs, in that order",
)
parser.add_argument(
    "--sample-size",
    type=int,
    default=100,
    help="total profiles in a random index, including the photographer (default 100; ignored with --ids)",
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data"),
    help="folder to write the index to, relative to the project root (default data)",
)
args = parser.parse_args()
if args.sample_size < 1:
    parser.error("--sample-size must be at least 1")

requested_ids = None
if args.ids:
    with args.ids.open(encoding="utf-8") as file:
        requested_ids = [profile["profile_id"] for profile in json.load(file)]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError(f"Duplicate profile IDs in {args.ids}.")

database_url = (
    os.getenv("DATABASE_URL_UNPOOLED")
    or os.getenv("DATABASE_URL")
)

if not database_url:
    raise RuntimeError("No database URL found in .env.local")



with psycopg.connect(
    database_url,
    row_factory=dict_row,
    connect_timeout=10,
) as connection:
    connection.execute("SET TRANSACTION READ ONLY")

    if requested_ids is not None:
        cursor = connection.execute(
            "SELECT profile_id, profile FROM people WHERE profile_id = ANY(%s)",
            (requested_ids,),
        )
        rows_by_id = {row["profile_id"]: row for row in cursor.fetchall()}
        missing = [pid for pid in requested_ids if pid not in rows_by_id]
        if missing:
            raise ValueError(f"{len(missing)} requested profiles not found, e.g. {missing[:3]}")
        rows = [rows_by_id[pid] for pid in requested_ids]
    else:
        photographer_id = "syn2_bootcamp_self_taught_0000775"
        cursor = connection.execute(
            """
            SELECT profile_id, profile
            FROM people
            WHERE profile_id = %s

            UNION ALL

            SELECT *
            FROM (
                SELECT profile_id, profile
                FROM people
                WHERE profile_id <> %s
                ORDER BY random()
                LIMIT %s
            ) AS sample;
            """,
            (photographer_id, photographer_id, args.sample_size - 1),
        )
        rows = cursor.fetchall()

if not rows:
    print("The people table is empty.")
else:
    profile_ids = []
    texts = []
    experience_texts = []

    for row in rows:
        text = profile_to_text(row["profile"])

        if text.strip():
            profile_ids.append(row["profile_id"])
            texts.append(text)
            experience_texts.append(experience_to_text(row["profile"]))

    print(f"Prepared {len(texts)} profiles.")
    if len(texts) < len(rows):
        print(f"Warning: skipped {len(rows) - len(texts)} profiles with empty text.")

    if texts:
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        embeddings = model.encode(
            texts,
            batch_size=4,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print("Embeddings shape:", embeddings.shape)

        # Profiles without any job title get a zero vector: experience similarity 0.
        experience_embeddings = np.zeros_like(embeddings)
        with_jobs = [i for i, text in enumerate(experience_texts) if text.strip()]
        if with_jobs:
            experience_embeddings[with_jobs] = model.encode(
                [experience_texts[i] for i in with_jobs],
                batch_size=4,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

        data_dir = project_root / args.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        profiles = [
            {"profile_id": profile_id, "text": text, "experience_text": experience_text}
            for profile_id, text, experience_text in zip(profile_ids, texts, experience_texts)
        ]

        with (data_dir / "profiles.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(profiles, file, ensure_ascii=False, indent=2)

        np.save(data_dir / "embeddings.npy", embeddings)
        np.save(data_dir / "experience_embeddings.npy", experience_embeddings)
        metadata = {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dimension": int(embeddings.shape[1]),
            "normalize_embeddings": True,
            "experience_weight": DEFAULT_EXPERIENCE_WEIGHT,
            "query_prompt": (
                "Instruct: Given a search query, retrieve professional "
                "profiles that match the requested qualifications.\nQuery: "
            ),
        }

        with (data_dir / "index_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2)

        print(f"Saved {len(profiles)} profiles and their embeddings.")
