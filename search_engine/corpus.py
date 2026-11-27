"""Build and read the fixed working corpus: a snapshot of 10,000 profiles.

The corpus is the one set of profiles that every method uses: the lexical
index, the embeddings, learning-to-rank and the evaluation. It is written to
disk once, so later steps never touch the database again and always see
exactly the same profiles in exactly the same order.

Which profiles
--------------
- all 1,000 profiles in tests/retrieval_1000/profile_ids.json, so the graded
  judgments stay valid;
- plus 9,000 other profiles chosen at random with a fixed seed.

Reproducible sampling: all other profile IDs are fetched, sorted, and sampled
with random.Random(seed). Sorting first means the result does not depend on
the order the database returns rows in, so the same seed gives the same
profiles on any machine and any PostgreSQL server holding the same table.
(SQL ORDER BY random() cannot be seeded reliably, so it is not used.)

Order in the file: the 1,000 judged profiles first (in profile_ids.json
order), then the 9,000 sampled ones sorted by profile_id.

Which columns
-------------
profile_id, profile (the JSON), the filter columns job_family, seniority,
city, state, experience_years, and seed_role and cohort. seed_role is needed
to build weak training labels for learning-to-rank and cohort is handy for
analysis; neither is ever indexed. Protected or useless columns (age,
desired_salary, the duplicate skills column, created_at) are not read at all,
and the protected keys inside the profile JSON (user.age, user.desired_salary,
user.ethnicity, user.legal_status, user.sponsorship_needed) are removed
before writing. user.skills stays: fields.py reads the skills from there.

Embeddings
----------
The teammate's table public.profile_embeddings holds, for every profile, three
pgvector columns (core_embedding, projects_embedding, interests_embedding) made
with one model (Qwen/Qwen3-Embedding-0.6B). They are snapshotted too, so the
semantic searcher never touches the database either:
- row i of each matrix is the profile on line i of profiles.jsonl, which is
  doc_id i in the lexical index: the two methods are aligned by construction;
- every row is normalized to length 1 (normalizing an already normalized
  vector changes nothing), so cosine similarity is a dot product;
- a NULL embedding (e.g. a profile with no projects) is stored as a row of
  zeros: it scores 0 against every query. The counts are in embeddings.json.
The pgvector adapter (pip install pgvector) makes psycopg return vector
columns as numpy arrays instead of '[0.05, -0.06, ...]' strings.

Output (data/ is not in Git)
----------------------------
data/corpus/profiles.jsonl              one JSON object per line, one line per profile
data/corpus/profile_ids.json            [{"profile_id": ...}, ...] in the same order
data/corpus/metadata.json               seed, sizes, source and build time
data/corpus/embeddings_core.npy         float32 (size x 1024), same row order
data/corpus/embeddings_projects.npy     idem
data/corpus/embeddings_interests.npy    idem
data/corpus/embeddings.json             model, dimension, NULL counts, build time

Usage (from the project root; reads DATABASE_URL from .env.local, read-only):
    python -m search_engine.corpus                    # profiles + embeddings
    python -m search_engine.corpus --size 10000 --seed 42
    python -m search_engine.corpus --embeddings-only  # embeddings for the existing
                                                      # corpus, profiles untouched
    python -m search_engine.corpus --no-embeddings    # profiles only
    python -m search_engine.corpus --all              # every profile in the database,
                                                      # written to data/full/
    python -m search_engine.corpus --all --embeddings-only

With --all the same files are written, to data/full/ instead of data/corpus/,
so the 10,000-profile corpus is left alone. The order is the same rule: the
1,000 judged profiles first, then all the others sorted by profile_id (the
seed has no effect). Expect about 3 GB of embeddings (3 x 1 GB).

Reading it back:
    from search_engine.corpus import load_corpus, load_embeddings, FULL_DIR
    rows = load_corpus()                  # list of dicts, in file order
    core = load_embeddings("core")        # numpy array, row i = rows[i]
    rows = load_corpus(FULL_DIR / "profiles.jsonl")     # the full corpus
    core = load_embeddings("core", FULL_DIR)
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
DEFAULT_DIR = project_root / "data" / "corpus"   # the 10,000-profile corpus
FULL_DIR = project_root / "data" / "full"        # every profile (--all)
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
    """Return the profile JSON without the protected keys in profile['user']."""
    user = profile.get("user")
    if isinstance(user, dict):
        profile = dict(profile)
        profile["user"] = {key: value for key, value in user.items() if key not in SENSITIVE_USER_KEYS}
    return profile


def read_judged_ids(path=JUDGED_IDS_PATH):
    with Path(path).open(encoding="utf-8") as file:
        ids = [row["profile_id"] for row in json.load(file)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate profile IDs in {path}")
    return ids


def sample_ids(all_ids, judged_ids, size, seed):
    """Judged IDs first, then (size - len(judged)) others sampled with a fixed seed."""
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
    import psycopg
    from dotenv import load_dotenv
    from psycopg.rows import dict_row

    load_dotenv(project_root / ".env.local")
    database_url = os.getenv("DATABASE_URL_UNPOOLED") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("No database URL found in .env.local")
    return psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10)


def fetch_rows(connection, ids):
    """Fetch the given profiles, returned in the same order as ids."""
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
    """Write the three output files. Each is written to a temporary file first and
    then renamed, so a crash never leaves a half-written corpus behind."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, lines):
        tmp = out_dir / (name + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            file.writelines(lines)
        tmp.replace(out_dir / name)

    # profiles.jsonl is written line by line: with --all, one big string would
    # cost as much memory again as the rows themselves.
    write("profiles.jsonl", (json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write("profile_ids.json", [json.dumps([{"profile_id": row["profile_id"]} for row in rows], indent=1)])
    write("metadata.json", [json.dumps(metadata, indent=2)])


def normalize_rows(matrix):
    """Scale every row to length 1, in place (no second copy of a 1 GB matrix).
    All-zero rows (missing embeddings) stay zero."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def to_array(value):
    """A pgvector value as a float32 numpy array (None stays None). Depending on the
    pgvector version the adapter returns a numpy array or a pgvector Vector."""
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def fetch_embeddings(connection, ids):
    """Fetch the three embeddings of the given profiles, as normalized float32
    matrices whose row i belongs to ids[i]. Returns (matrices, info)."""
    import numpy as np
    from pgvector.psycopg import register_vector

    register_vector(connection)
    columns = ", ".join(EMBEDDING_KINDS.values())
    query = f"SELECT profile_id, model, {columns} FROM public.profile_embeddings WHERE profile_id = ANY(%s)"

    # Each vector goes straight into its row of the matrix as it arrives, so the
    # vectors are never held twice (matters with --all: about 1 GB per kind).
    # The matrices are made when the first vector shows the dimension.
    position = {pid: i for i, pid in enumerate(ids)}
    seen = set()
    models = set()
    matrices = None
    dimension = None
    nulls = {kind: 0 for kind in EMBEDDING_KINDS}
    for start in range(0, len(ids), FETCH_BATCH):
        batch = ids[start:start + FETCH_BATCH]
        for row in connection.execute(query, (batch,), binary=True).fetchall():
            pid = row["profile_id"]
            if pid in seen:
                raise ValueError(f"Two embedding rows for {pid}")
            seen.add(pid)
            models.add(row["model"])
            for kind, column in EMBEDDING_KINDS.items():
                vector = to_array(row[column])
                if vector is None:
                    nulls[kind] += 1
                    continue
                if matrices is None:
                    dimension = len(vector)
                    matrices = {k: np.zeros((len(ids), dimension), dtype=np.float32) for k in EMBEDDING_KINDS}
                if len(vector) != dimension:
                    raise ValueError(f"{pid} {column} has {len(vector)} dimensions, expected {dimension}")
                matrices[kind][position[pid]] = vector

    missing = [pid for pid in ids if pid not in seen]
    if missing:
        raise ValueError(f"{len(missing)} profiles have no row in profile_embeddings, e.g. {missing[:3]}")
    if len(models) != 1:
        raise ValueError(f"Embeddings come from more than one model: {sorted(models)}")
    if matrices is None:
        raise ValueError("Every embedding is NULL")
    for kind in matrices:
        normalize_rows(matrices[kind])

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
    """Write one .npy per kind plus embeddings.json (temporary file, then rename)."""
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
    """Read one embedding matrix back (row i = line i of profiles.jsonl = doc_id i)."""
    import numpy as np

    path = Path(corpus_dir) / f"embeddings_{kind}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build it with: python -m search_engine.corpus --embeddings-only")
    return np.load(path)


def read_corpus_ids(corpus_dir=DEFAULT_DIR):
    """The profile ids of the existing corpus, in file order."""
    path = Path(corpus_dir) / "profile_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the corpus first: python -m search_engine.corpus")
    with path.open(encoding="utf-8") as file:
        return [row["profile_id"] for row in json.load(file)]


def load_corpus(path=DEFAULT_DIR / "profiles.jsonl"):
    """Read the corpus back: a list of row dicts, in file order."""
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
    parser = argparse.ArgumentParser(description="Build the fixed corpus snapshot from the database.")
    how_many = parser.add_mutually_exclusive_group()
    how_many.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"total profiles (default {DEFAULT_SIZE})")
    how_many.add_argument("--all", action="store_true",
                          help="take every profile in the database (default output folder data/full)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder (default data/corpus, or data/full with --all)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--embeddings-only", action="store_true",
                       help="only snapshot the embeddings of the existing corpus (profiles untouched)")
    group.add_argument("--no-embeddings", action="store_true", help="only snapshot the profiles")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = FULL_DIR if args.all else DEFAULT_DIR

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
        # --all: the size is whatever the database holds now, so sample_ids takes
        # every other profile (judged first, the rest sorted; the seed does nothing).
        size = len(set(all_ids)) if args.all else args.size
        ids = sample_ids(all_ids, judged_ids, size, args.seed)
        rows = fetch_rows(connection, ids)
        if not args.no_embeddings:
            matrices, info = fetch_embeddings(connection, ids)

    metadata = {
        "all": args.all,
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
