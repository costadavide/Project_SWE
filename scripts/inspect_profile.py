import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from semantic_search.profile_text import profile_to_text
from sentence_transformers import SentenceTransformer
import json
import numpy as np


project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env.local"
load_dotenv(env_path)



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
            LIMIT 99
        ) AS sample;
        """,
        (photographer_id, photographer_id),
    )

    rows = cursor.fetchall()

if not rows:
    print("The people table is empty.")
else:
    profile_ids = []
    texts = []

    for row in rows:
        text = profile_to_text(row["profile"])

        if text.strip():
            profile_ids.append(row["profile_id"])
            texts.append(text)

    print(f"Prepared {len(texts)} profiles.")

    if texts:
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        embeddings = model.encode(
            texts,
            batch_size=4,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print("Embeddings shape:", embeddings.shape)

        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)

        profiles = [
            {"profile_id": profile_id, "text": text}
            for profile_id, text in zip(profile_ids, texts)
        ]

        with (data_dir / "profiles.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(profiles, file, ensure_ascii=False, indent=2)

        np.save(data_dir / "embeddings.npy", embeddings)
        metadata = {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dimension": int(embeddings.shape[1]),
            "normalize_embeddings": True,
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

        while True:
            query = input(
                "\nEnter a search query (or 'exit' to stop): "
            ).strip()

            if query.lower() == "exit":
                break

            if not query:
                print("Please enter a non-empty query.")
                continue

            query_embedding = model.encode(
                query,
                prompt=(
                    "Instruct: Given a search query, retrieve professional "
                    "profiles that match the requested qualifications.\nQuery: "
                ),
                normalize_embeddings=True,
            )

            scores = embeddings @ query_embedding
            top_indices = scores.argsort()[::-1][:5]

            for rank, index in enumerate(top_indices, start=1):
                print(
                    f"\n{rank}. Profile: {profile_ids[index]} "
                    f"| Similarity: {scores[index]:.4f}"
                )
                print(texts[index])
