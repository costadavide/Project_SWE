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

Output (data/ is not in Git)
----------------------------
data/corpus/profiles.jsonl     one JSON object per line, one line per profile
data/corpus/profile_ids.json   [{"profile_id": ...}, ...] in the same order,
                               the format scripts.build_index --ids expects
data/corpus/metadata.json      seed, sizes, source and build time

Usage (from the project root; reads DATABASE_URL from .env.local, read-only):
    python -m search_engine.corpus
    python -m search_engine.corpus --size 10000 --seed 42

Reading it back:
    from search_engine.corpus import load_corpus
    rows = load_corpus()          # list of dicts, in file order
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

    def write(name, text):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(out_dir / name)

    write("profiles.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write("profile_ids.json", json.dumps([{"profile_id": row["profile_id"]} for row in rows], indent=1))
    write("metadata.json", json.dumps(metadata, indent=2))


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
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"total profiles (default {DEFAULT_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR, help="output folder (default data/corpus)")
    args = parser.parse_args()

    judged_ids = read_judged_ids()
    with connect() as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        all_ids = [row["profile_id"] for row in connection.execute("SELECT profile_id FROM people").fetchall()]
        print(f"{len(all_ids)} profiles in the database, {len(judged_ids)} judged.")
        ids = sample_ids(all_ids, judged_ids, args.size, args.seed)
        rows = fetch_rows(connection, ids)

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


if __name__ == "__main__":
    main()
