import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_search.scoring import combined_scores, load_experience_embeddings

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

with (data_dir / "profiles.json").open(
    encoding="utf-8"
) as file:
    profiles = json.load(file)

embeddings = np.load(
    data_dir / "embeddings.npy",
    allow_pickle=False,
)

if embeddings.ndim != 2 or len(profiles) != embeddings.shape[0]:
    raise ValueError("Profiles and embeddings do not match.")

with (data_dir / "index_metadata.json").open(
    encoding="utf-8"
) as file:
    metadata = json.load(file)

if embeddings.shape[1] != metadata["embedding_dimension"]:
    raise ValueError("Embedding dimensions do not match the metadata.")

experience_embeddings = load_experience_embeddings(data_dir, metadata, embeddings.shape)
experience_weight = metadata.get("experience_weight", 0.0) if experience_embeddings is not None else 0.0

print(f"Loaded {len(profiles)} profiles (experience weight {experience_weight}).")

model = SentenceTransformer(metadata["model_name"])


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
        prompt=metadata["query_prompt"],
        normalize_embeddings=metadata["normalize_embeddings"],
    )

    scores = combined_scores(query_embedding, embeddings, experience_embeddings, experience_weight)
    top_indices = scores.argsort()[::-1][:5]

    for rank, index in enumerate(top_indices, start=1):
        profile = profiles[index]

        print(
            f"\n{rank}. Profile: {profile['profile_id']} "
            f"| Similarity: {scores[index]:.4f}"
        )
        print(profile["text"])