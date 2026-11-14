"""Combine full-profile and experience-only similarity into one ranking score.

score = (1 - w) * cosine(query, full profile) + w * cosine(query, job history)

The full profile carries skills, certifications and education; the job history
carries only what the person has actually worked as. A career changer whose
skills match the query but whose jobs do not gets a lower experience score and
ranks below people already doing the job, without disappearing from the results.
"""

import numpy as np

DEFAULT_EXPERIENCE_WEIGHT = 0.3


def combined_scores(query_embedding, embeddings, experience_embeddings=None, experience_weight=0.0):
    scores = embeddings @ query_embedding
    if experience_embeddings is None or not experience_weight:
        return scores
    experience_scores = experience_embeddings @ query_embedding
    return (1 - experience_weight) * scores + experience_weight * experience_scores


def load_experience_embeddings(data_dir, metadata, expected_shape):
    """Experience vectors if this index has them, else None (older indexes)."""
    path = data_dir / "experience_embeddings.npy"
    if not metadata.get("experience_weight") or not path.exists():
        return None
    experience_embeddings = np.load(path, allow_pickle=False)
    if experience_embeddings.shape != expected_shape:
        raise ValueError("Experience embeddings do not match the profile embeddings.")
    return experience_embeddings
