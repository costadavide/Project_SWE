"""Snapshot of the working corpus, written once so later steps never query the database.

Profiles: the 1,000 judged profiles (tests/retrieval_1000) first, then a seeded
random sample of the others (default 10,000 in total, seed 42). Protected keys
in the profile JSON are removed. Embeddings (core, projects, interests) come
from public.profile_embeddings, normalized, with NULL stored as a zero row.
Line i of every output is the same profile, which is also doc_id i in the
lexical index.

Output (data/corpus/): profiles.jsonl, profile_ids.json, metadata.json,
embeddings_{core,projects,interests}.npy, embeddings.json.

Usage (from the project root, DATABASE_URL in .env.local, read-only):
    python -m search_engine.corpus [--size N] [--seed S] [--out-dir DIR]
    python -m search_engine.corpus --embeddings-only   # or --no-embeddings
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

DEFAULT_SIZE = 10_000
DEFAULT_SEED = 42
DEFAULT_DIR = project_root / "data" / "corpus"
JUDGED_IDS_PATH = project_root / "tests" / "retrieval_1000" / "profile_ids.json"
COLUMNS = [
    "profile_id",
    "job_family",
    "seniority",
    "city",
    "state",
    "experience_years",
    "seed_role",
    "cohort",
    "profile",
]
FETCH_BATCH = 1000

# Embedding columns in public.profile_embeddings: kind -> column name.
EMBEDDING_KINDS = {
    "core": "core_embedding",
    "projects": "projects_embedding",
    "interests": "interests_embedding",
}

# Protected attributes inside the profile JSON (profile.user). Leaving out the
# age / desired_salary columns is not enough: the JSON holds its own copies.
# They are removed before writing, so the snapshot never contains them.
SENSITIVE_USER_KEYS = ["age", "desired_salary", "ethnicity", "legal_status", "sponsorship_needed"]


def strip_sensitive(profile):
    """The profile JSON without the protected keys in profile['user']."""
    user = profile.get("user")
    if isinstance(user, dict):
        profile = dict(profile)
        profile["user"] = {key: value for key, value in user.items() if key not in SENSITIVE_USER_KEYS}
    return profile


def read_judged_ids(path=JUDGED_IDS_PATH):
    """profile_ids of the judged profiles; fails on duplicates."""
    with Path(path).open(encoding="utf-8") as file:
        ids = [row["profile_id"] for row in json.load(file)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate profile IDs in {path}")
    return ids


def sample_ids(all_ids, judged_ids, size, seed):
    """Judged ids first, then size - len(judged) others sampled with a fixed seed."""
    judged = set(judged_ids)
    others = sorted(set(all_ids) - judged)
    n_random = size - len(judged_ids)
    if n_random < 0:
        raise ValueError(f"--size {size} is smaller than the {len(judged_ids)} judged profiles")
    if n_random > len(others):
        raise ValueError(f"Only {len(others)} other profiles exist, cannot sample {n_random}")
    sampled = random.Random(seed).sample(others, n_random)
    return list(judged_ids) + sorted(sampled)


def connect():
    """Open a read connection to the database URL in .env.local."""
    import psycopg
    from dotenv import load_dotenv
    from psycopg.rows import dict_row

    load_dotenv(project_root / ".env.local")
    database_url = os.getenv("DATABASE_URL_UNPOOLED") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("No database URL found in .env.local")
    return psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10)


def fetch_rows(connection, ids):
    """Fetch the given profiles (protected keys removed), in the order of ids."""
    rows_by_id = {}
    query = f"SELECT {', '.join(COLUMNS)} FROM people WHERE profile_id = ANY(%s)"
    for start in range(0, len(ids), FETCH_BATCH):
        batch = ids[start:start + FETCH_BATCH]
        for row in connection.execute(query, (batch,)).fetchall():
            row["profile"] = strip_sensitive(row["profile"] or {})
            rows_by_id[row["profile_id"]] = row
    missing = [pid for pid in ids if pid not in rows_by_id]
    if missing:
        raise ValueError(f"{len(missing)} profiles not found in the database, e.g. {missing[:3]}")
    return [rows_by_id[pid] for pid in ids]


def write_corpus(rows, out_dir, metadata):
    """Write profiles.jsonl, profile_ids.json and metadata.json (temp file, then rename)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, text):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(out_dir / name)

    write("profiles.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write("profile_ids.json", json.dumps([{"profile_id": row["profile_id"]} for row in rows], indent=1))
    write("metadata.json", json.dumps(metadata, indent=2))


def normalize_rows(matrix):
    """Scale every row to length 1; all-zero rows stay zero."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def to_array(value):
    """A pgvector value as a float32 numpy array (None stays None)."""
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def fetch_embeddings(connection, ids):
    """The three embeddings of ids as normalized float32 matrices (row i = ids[i]),
    plus an info dict. Fails on missing rows, duplicates or mixed models.
    """
    import numpy as np
    from pgvector.psycopg import register_vector

    register_vector(connection)
    columns = ", ".join(EMBEDDING_KINDS.values())
    query = f"SELECT profile_id, model, {columns} FROM public.profile_embeddings WHERE profile_id = ANY(%s)"
    rows_by_id = {}
    for start in range(0, len(ids), FETCH_BATCH):
        batch = ids[start:start + FETCH_BATCH]
        for row in connection.execute(query, (batch,), binary=True).fetchall():
            if row["profile_id"] in rows_by_id:
                raise ValueError(f"Two embedding rows for {row['profile_id']}")
            for column in EMBEDDING_KINDS.values():
                row[column] = to_array(row[column])
            rows_by_id[row["profile_id"]] = row

    missing = [pid for pid in ids if pid not in rows_by_id]
    if missing:
        raise ValueError(f"{len(missing)} profiles have no row in profile_embeddings, e.g. {missing[:3]}")
    models = {row["model"] for row in rows_by_id.values()}
    if len(models) != 1:
        raise ValueError(f"Embeddings come from more than one model: {sorted(models)}")

    dimension = None
    for row in rows_by_id.values():
        for column in EMBEDDING_KINDS.values():
            if row[column] is not None:
                dimension = len(row[column])
                break
        if dimension:
            break
    if dimension is None:
        raise ValueError("Every embedding is NULL")

    matrices = {}
    nulls = {}
    for kind, column in EMBEDDING_KINDS.items():
        matrix = np.zeros((len(ids), dimension), dtype=np.float32)
        nulls[kind] = 0
        for i, pid in enumerate(ids):
            vector = rows_by_id[pid][column]
            if vector is None:
                nulls[kind] += 1
                continue
            if len(vector) != dimension:
                raise ValueError(f"{pid} {column} has {len(vector)} dimensions, expected {dimension}")
            matrix[i] = vector
        matrices[kind] = normalize_rows(matrix)

    info = {
        "model": models.pop(),
        "dimension": dimension,
        "rows": len(ids),
        "normalized": True,
        "null_rows": nulls,
        "files": {kind: f"embeddings_{kind}.npy" for kind in EMBEDDING_KINDS},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return matrices, info


def write_embeddings(matrices, info, out_dir):
    """Write one .npy per kind plus embeddings.json (temp file, then rename)."""
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, matrix in matrices.items():
        target = out_dir / info["files"][kind]
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("wb") as file:
            np.save(file, matrix)
        tmp.replace(target)
    tmp = out_dir / "embeddings.json.tmp"
    tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "embeddings.json")


def load_embeddings(kind="core", corpus_dir=DEFAULT_DIR):
    """Read one embedding matrix back (row i = doc_id i)."""
    import numpy as np

    path = Path(corpus_dir) / f"embeddings_{kind}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build it with: python -m search_engine.corpus --embeddings-only")
    return np.load(path)


def read_corpus_ids(corpus_dir=DEFAULT_DIR):
    """profile_ids of the existing corpus, in file order."""
    path = Path(corpus_dir) / "profile_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the corpus first: python -m search_engine.corpus")
    with path.open(encoding="utf-8") as file:
        return [row["profile_id"] for row in json.load(file)]


def load_corpus(path=DEFAULT_DIR / "profiles.jsonl"):
    """Read profiles.jsonl back as a list of row dicts, in file order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build it with: python -m search_engine.corpus")
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    """Command line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Build the fixed corpus snapshot from the database.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"total profiles (default {DEFAULT_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR, help="output folder (default data/corpus)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--embeddings-only", action="store_true",
                       help="only snapshot the embeddings of the existing corpus (profiles untouched)")
    group.add_argument("--no-embeddings", action="store_true", help="only snapshot the profiles")
    args = parser.parse_args()

    if args.embeddings_only:
        ids = read_corpus_ids(args.out_dir)
        with connect() as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            matrices, info = fetch_embeddings(connection, ids)
        write_embeddings(matrices, info, args.out_dir)
        print(f"Wrote {info['rows']} x {info['dimension']} embeddings ({info['model']}) to {args.out_dir}; "
              f"NULL rows: {info['null_rows']}")
        return

    judged_ids = read_judged_ids()
    with connect() as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        all_ids = [row["profile_id"] for row in connection.execute("SELECT profile_id FROM people").fetchall()]
        print(f"{len(all_ids)} profiles in the database, {len(judged_ids)} judged.")
        ids = sample_ids(all_ids, judged_ids, args.size, args.seed)
        rows = fetch_rows(connection, ids)
        if not args.no_embeddings:
            matrices, info = fetch_embeddings(connection, ids)

    metadata = {
        "size": len(rows),
        "judged": len(judged_ids),
        "sampled": len(rows) - len(judged_ids),
        "seed": args.seed,
        "database_profiles": len(all_ids),
        "columns": COLUMNS,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_corpus(rows, args.out_dir, metadata)
    print(f"Wrote {len(rows)} profiles to {args.out_dir / 'profiles.jsonl'}")
    if not args.no_embeddings:
        write_embeddings(matrices, info, args.out_dir)
        print(f"Wrote {info['rows']} x {info['dimension']} embeddings ({info['model']}); NULL rows: {info['null_rows']}")


if __name__ == "__main__":
    main()
