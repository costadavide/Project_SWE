"""Semantic search over the corpus snapshot, with the same interface as BM25 / BM25F.

The profile vectors come from data/corpus/embeddings_{core,projects,interests}.npy
(written by python -m search_engine.corpus --embeddings-only). Row i of every
matrix is line i of data/corpus/profile_ids.json, which is doc_id i in the
lexical index, so a set of doc_ids (`allowed`) means the same profiles here and
there. check_alignment(index) verifies that once.

Scoring is the one in scripts/test_embeddings.py: all vectors are normalized,
so cosine similarity is a dot product, and a profile's score is

    score = sum over kinds of  weight[kind] * cosine(query, profile[kind])

with weights {"core": 1.0} by default, or e.g. {"core": 0.8, "projects": 0.15,
"interests": 0.05} (test_embeddings' "weighted" mode). A missing embedding is a
row of zeros and contributes 0.

Only the query is embedded at search time, with the same model as the profiles
and a task instruction (QUERY_PROMPT). Profiles were embedded without one.

Usage:
    from semantic_search.searcher import SemanticSearcher
    searcher = SemanticSearcher()                       # core only
    profiles, breakdowns = searcher.search("icu nurse", k=10, allowed=None)

The model (about 1.2 GB) is downloaded by sentence-transformers on first use
and cached; later runs load it from the cache.
"""

import json
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = project_root / "data" / "corpus"

# Same instruction as data/index_metadata.json and tests/evaluate_retrieval_1000.py.
QUERY_PROMPT = ("Instruct: Given a search query, retrieve professional profiles "
                "that match the requested qualifications.\nQuery: ")
DEFAULT_WEIGHTS = {"core": 1.0}


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SemanticSearcher:
    def __init__(self, corpus_dir=DEFAULT_CORPUS_DIR, weights=None, model=None,
                 query_prompt=QUERY_PROMPT, device=None):
        self.corpus_dir = Path(corpus_dir)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.query_prompt = query_prompt
        self.device = device
        self.model = model

        info_path = self.corpus_dir / "embeddings.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found. Build it with: "
                                    f"python -m search_engine.corpus --embeddings-only")
        with info_path.open(encoding="utf-8") as file:
            self.info = json.load(file)
        self.model_name = self.info["model"]

        with (self.corpus_dir / "profile_ids.json").open(encoding="utf-8") as file:
            self.profile_ids = [row["profile_id"] for row in json.load(file)]

        self.matrices = {}
        for kind in self.weights:
            if kind not in self.info["files"]:
                raise ValueError(f"Unknown embedding kind {kind!r}; available: {list(self.info['files'])}")
            matrix = np.load(self.corpus_dir / self.info["files"][kind], allow_pickle=False)
            if matrix.shape != (len(self.profile_ids), self.info["dimension"]):
                raise ValueError(f"{kind} embeddings have shape {matrix.shape}, expected "
                                 f"({len(self.profile_ids)}, {self.info['dimension']})")
            self.matrices[kind] = matrix

    def __len__(self):
        return len(self.profile_ids)

    def load_model(self):
        """Load the embedding model once (first search), unless one was passed in."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device or pick_device())
        return self.model

    def encode(self, query):
        """The query as a normalized vector, embedded with the profiles' model and the query prompt."""
        vector = self.load_model().encode(query, prompt=self.query_prompt, normalize_embeddings=True)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.info["dimension"],):
            raise ValueError(f"Query vector has shape {vector.shape}, expected ({self.info['dimension']},)")
        return vector

    def check_alignment(self, index):
        """Fail unless row i here is the same profile as doc_id i in the lexical index."""
        if len(index) != len(self.profile_ids):
            raise ValueError(f"Index has {len(index)} docs but there are {len(self.profile_ids)} embedding rows")
        for doc_id, profile_id in enumerate(self.profile_ids):
            if index.read_doc(doc_id)["profile_id"] != profile_id:
                raise ValueError(f"doc_id {doc_id} is {index.read_doc(doc_id)['profile_id']} in the index "
                                 f"but {profile_id} in the embeddings")

    def search(self, query, k, allowed=None):
        """Top k profiles by weighted cosine similarity, among the allowed doc_ids (None = all).
        Returns (profiles, breakdowns) like BM25F: [(profile_id, score)] best first, and
        [(profile_id, {kind: weight * cosine})] whose parts add up to the score."""
        if allowed is not None:
            rows = np.fromiter(sorted(allowed), dtype=np.int64, count=len(allowed))
            if len(rows) == 0:
                return [], []
        else:
            rows = np.arange(len(self.profile_ids))

        query_vector = self.encode(query)
        parts = {kind: self.weights[kind] * (self.matrices[kind][rows] @ query_vector) for kind in self.weights}
        scores = sum(parts.values())

        # best k by score, ties broken by the lower doc_id (rows are sorted, so a stable sort keeps that order)
        order = np.argsort(-scores, kind="stable")[:k]

        profiles = []
        breakdowns = []
        for position in order:
            doc_id = int(rows[position])
            profile_id = self.profile_ids[doc_id]
            profiles.append((profile_id, float(scores[position])))
            breakdowns.append((profile_id, {kind: float(parts[kind][position]) for kind in parts}))
        return profiles, breakdowns
