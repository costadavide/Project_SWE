from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from test_embeddings import MODELS, search, device


queries = [
    "Machine Learning Engineer",
    "Financial analyst",
    "Someone experienced with Python, Docker and Tensorflow",
    "Professional skilled in SQL, Tableau, and data visualization",
    "Developer with experience using React, TypeScript, and REST APIs",
    "Find someone with experience implementing HR information systems, managing employee relations, and handling labor law compliance.",
    "Looking for a backend developer experienced in Python, PostgreSQL, and building scalable APIs.",
    "Looking for a project manager experienced in Agile methodologies, stakeholder management, and coordinating cross-functional teams.",
    "Senior software engineer with at least 5 years of experience developing production applications.",
    "Entry-level data analyst with SQL and Excel skills.",
    "Experienced HR manager who has led teams and managed recruitment processes.",
    "Someone who transitioned from software development into machine learning engineering.",
    "Professional experienced in identifying fraudulent transactions using predictive analytics.",
]


# ==========================================
# CONFIGURATION
# ==========================================

VARIANT = "no_seed_role"
MODE = "core"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# RUN EVALUATION
# ==========================================

def generate_evaluation_data():

    rankings = []

    for model_name, model_path in MODELS.items():

        print(f"\nLoading model: {model_name}")

        # Load each model only once
        model = SentenceTransformer(
            model_path,
            device=device
        )

        for query_id, query in enumerate(queries, start=1):

            print(f"Query {query_id}/{len(queries)}")

            results = search(
                query=query,
                model_name=model_name,
                mode=MODE,
                variant=VARIANT,
                model=model
            )

            for rank, (_, row) in enumerate(results.iterrows(), start=1):

                rankings.append({
                    "query_id": query_id,
                    "query": query,
                    "model": model_name,
                    "rank": rank,
                    "profile_id": row["profile_id"],
                    "seed_role": row["seed_role"],
                    "core_text": row["core_text"],
                    "final_score": row["final_score"],
                })

        del model

        if device == "mps":
            torch.mps.empty_cache()

    # ==========================================
    # SAVE MODEL RANKINGS
    # ==========================================

    df_rankings = pd.DataFrame(rankings)

    rankings_path = OUTPUT_DIR / "rankings.csv"
    df_rankings.to_csv(rankings_path, index=False)

    # ==========================================
    # CREATE MANUAL LABELING DATASET
    # ==========================================

    df_labels = (
        df_rankings[
            ["query_id", "query", "profile_id", "seed_role", "core_text"]
        ]
        .drop_duplicates(subset=["query_id", "profile_id"])
        .copy()
    )

    df_labels["relevance"] = ""

    labels_path = OUTPUT_DIR / "labels.csv"
    df_labels.to_csv(labels_path, index=False)

    print("\nEvaluation dataset created!")
    print(f"Total rankings: {len(df_rankings)}")
    print(f"Unique query-profile pairs: {len(df_labels)}")
    print(f"\nRankings: {rankings_path}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    generate_evaluation_data()
