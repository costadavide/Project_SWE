"""
Embed every profile in the `people` table and save the vectors in PostgreSQL.

Run from the project root:

    python -m scripts.embed_people              # embed what is still missing
    python -m scripts.embed_people --limit 500  # at most 500 profiles, then stop
    python -m scripts.embed_people --status     # show progress, change nothing
    python -m scripts.embed_people --refresh    # also redo profiles whose text changed

Press Ctrl+C to stop: the current batch is finished and saved first (press it
again to quit at once and lose that batch). Run the same command again to
continue: a profile is done when it has a row in `profile_embeddings`, so there
is no progress file to keep.

Each profile gets one row with up to three vectors (core, projects, interests).
A vector is NULL when the profile has no such text.
"""

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from semantic_search.profile_text import text_generation


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024
TEXT_OPTIONS = {"include_seed_role": False, "include_bio": True}  # "no_seed_role"
PARTS = ("core", "projects", "interests")

CHUNK_SIZE = 256        # profiles saved per transaction
ENCODE_BATCH_SIZE = 16  # texts per model batch; 32 and 64 were slower on an M5 Pro
CONNECT_ATTEMPTS = 6    # when the connection drops during the run

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS profile_embeddings (
  profile_id           TEXT PRIMARY KEY REFERENCES people (profile_id) ON DELETE CASCADE,
  model                TEXT NOT NULL,
  text_hash            TEXT NOT NULL,
  core_embedding       vector({DIMENSIONS}),
  projects_embedding   vector({DIMENSIONS}),
  interests_embedding  vector({DIMENSIONS}),
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

UPSERT = """
INSERT INTO profile_embeddings (profile_id, model, text_hash, core_embedding,
                                projects_embedding, interests_embedding)
VALUES (%s, %s, %s, %b, %b, %b)
ON CONFLICT (profile_id) DO UPDATE SET
  model = EXCLUDED.model,
  text_hash = EXCLUDED.text_hash,
  core_embedding = EXCLUDED.core_embedding,
  projects_embedding = EXCLUDED.projects_embedding,
  interests_embedding = EXCLUDED.interests_embedding,
  updated_at = now()
"""


# ==========================================
# DATABASE
# ==========================================

def is_connection_loss(error):
    """True for a dropped connection; False for e.g. a full disk, where retrying is useless."""
    code = error.sqlstate
    return code is None or code.startswith("08") or code.startswith("57P")


class Database:
    """A connection that is opened again if it drops during a long run."""

    def __init__(self):
        load_dotenv(PROJECT_ROOT / ".env.local")
        self.url = os.getenv("DATABASE_URL_UNPOOLED") or os.getenv("DATABASE_URL")
        if not self.url:
            raise RuntimeError("No database URL found in .env.local")
        self.connection = None
        self.connect()

    def connect(self):
        self.connection = psycopg.connect(
            self.url,
            row_factory=dict_row,
            connect_timeout=15,
            # Notice a dead connection within about a minute instead of hanging.
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=3,
        )

    def run(self, operation):
        """Run operation(connection) in one transaction, reconnecting if needed."""
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                if self.connection.closed:
                    self.connect()
                with self.connection.transaction():
                    return operation(self.connection)
            except psycopg.OperationalError as error:
                if not is_connection_loss(error) or attempt == CONNECT_ATTEMPTS:
                    raise
                self.connection.close()
                wait = 2 ** attempt
                print(f"Connection lost ({str(error).strip()}). Trying again in {wait}s...")
                time.sleep(wait)

    def close(self):
        self.connection.close()


def create_table(connection):
    connection.execute(SCHEMA)


def load_todo(connection, refresh):
    """IDs to look at: profiles with no embedding for MODEL, or all of them to refresh."""
    if refresh:
        cursor = connection.execute("SELECT profile_id FROM people ORDER BY profile_id")
    else:
        cursor = connection.execute(
            """
            SELECT p.profile_id
            FROM people p
            LEFT JOIN profile_embeddings e
              ON e.profile_id = p.profile_id AND e.model = %s
            WHERE e.profile_id IS NULL
            ORDER BY p.profile_id
            """,
            (MODEL,),
        )
    return [row["profile_id"] for row in cursor.fetchall()]


def load_profiles(connection, profile_ids):
    """Profiles to embed, with the hash of the text embedded last time (if any)."""
    return connection.execute(
        """
        SELECT p.profile_id, p.seed_role, p.job_family, p.skills, p.profile,
               e.text_hash
        FROM people p
        LEFT JOIN profile_embeddings e
          ON e.profile_id = p.profile_id AND e.model = %s
        WHERE p.profile_id = ANY(%s)
        ORDER BY p.profile_id
        """,
        (MODEL, profile_ids),
    ).fetchall()


def save(connection, rows):
    register_vector(connection)
    with connection.cursor() as cursor:
        cursor.executemany(UPSERT, rows)


def print_status(connection):
    total = connection.execute("SELECT count(*) AS n FROM people").fetchone()["n"]
    exists = connection.execute(
        "SELECT to_regclass('profile_embeddings') IS NOT NULL AS ok"
    ).fetchone()["ok"]

    if not exists:
        print(f"No embeddings yet. Profiles to embed: {total:,}")
        return

    row = connection.execute(
        """
        SELECT count(*) AS done,
               count(core_embedding) AS core,
               count(projects_embedding) AS projects,
               count(interests_embedding) AS interests,
               pg_size_pretty(pg_total_relation_size('profile_embeddings')) AS size
        FROM profile_embeddings
        WHERE model = %s
        """,
        (MODEL,),
    ).fetchone()

    print(f"Model:     {MODEL}")
    print(f"Profiles:  {row['done']:,} of {total:,} embedded ({row['done'] / max(total, 1):.1%})")
    print(f"Vectors:   core {row['core']:,}, projects {row['projects']:,}, "
          f"interests {row['interests']:,}")
    print(f"Table:     {row['size']}")


# ==========================================
# EMBEDDINGS
# ==========================================

def load_model():
    # Imported here so that --status does not have to load torch.
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    print(f"Loading {MODEL} on {device}...")
    return SentenceTransformer(MODEL, device=device)


def hash_texts(texts):
    payload = json.dumps([texts[f"{part}_text"] for part in PARTS], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def embed(model, texts):
    """Unit-length float32 vectors, in the same order; None for empty texts."""
    results = [None] * len(texts)
    filled = [i for i, text in enumerate(texts) if text.strip()]
    if not filled:
        return results

    # Documents get no prompt; only search queries use the "query" prompt.
    vectors = model.encode(
        [texts[i] for i in filled],
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Normalized here and not by the model, which runs in bfloat16 and would
    # give vectors that are not exactly unit length.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.shape[1] != DIMENSIONS or not np.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError("The model returned invalid vectors; this batch was not saved.")

    for i, vector in zip(filled, vectors / norms):
        results[i] = vector
    return results


def embed_profiles(model, rows, texts):
    """Return one row of UPSERT values per profile."""
    # All three kinds in one call, so short texts are batched together.
    all_texts = [text[f"{part}_text"] for text in texts for part in PARTS]
    vectors = embed(model, all_texts)

    step = len(PARTS)
    return [
        (row["profile_id"], MODEL, hash_texts(text), *vectors[i * step:(i + 1) * step])
        for i, (row, text) in enumerate(zip(rows, texts))
    ]


# ==========================================
# STOPPING
# ==========================================

class StopRequest:
    """The first Ctrl+C asks to stop after the current batch, the second quits at once."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self.handle)
        signal.signal(signal.SIGTERM, self.handle)

    def handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nStopping after this batch is saved (Ctrl+C again to quit at once)...", flush=True)


# ==========================================
# MAIN
# ==========================================

def format_time(seconds):
    minutes = round(seconds / 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def run(database, stop, limit, refresh):
    database.run(create_table)
    todo = database.run(lambda connection: load_todo(connection, refresh))

    if not todo:
        print("All profiles already have embeddings.")
        return

    if limit is not None and not refresh:
        todo = todo[:limit]

    print(f"Profiles to {'check' if refresh else 'embed'}: {len(todo):,}. Press Ctrl+C to stop.")

    model = None
    embedded = 0
    started = time.monotonic()

    for start in range(0, len(todo), CHUNK_SIZE):
        if stop.requested or (limit is not None and embedded >= limit):
            break

        ids = todo[start:start + CHUNK_SIZE]
        rows = database.run(lambda connection: load_profiles(connection, ids))
        texts = [text_generation(row, **TEXT_OPTIONS) for row in rows]

        # Skips unchanged profiles with --refresh; otherwise nothing has a hash yet.
        changed = [
            (row, text) for row, text in zip(rows, texts)
            if row["text_hash"] != hash_texts(text)
        ]
        if limit is not None:
            changed = changed[:limit - embedded]

        if changed:
            if model is None:
                model = load_model()
                started = time.monotonic()  # leave loading out of the time estimate

            values = embed_profiles(model, *zip(*changed))
            database.run(lambda connection: save(connection, values))
            embedded += len(values)

        checked = start + len(ids)
        elapsed = time.monotonic() - started
        remaining = elapsed / checked * (len(todo) - checked)
        print(f"{checked:,}/{len(todo):,} ({checked / len(todo):.1%})  "
              f"embedded {embedded:,}  about {format_time(remaining)} left", flush=True)

    if stop.requested:
        print("Stopped. Run the same command again to continue.")
    print(f"Saved embeddings for {embedded:,} profiles.")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description="Embed all profiles and save them in PostgreSQL.")
    parser.add_argument("--limit", type=positive_int, help="embed at most this many profiles")
    parser.add_argument("--refresh", action="store_true",
                        help="check all profiles and redo those whose text changed")
    parser.add_argument("--status", action="store_true",
                        help="show how many profiles are embedded, then exit")
    args = parser.parse_args()

    database = Database()
    try:
        if args.status:
            database.run(print_status)
        else:
            run(database, StopRequest(), args.limit, args.refresh)
    except KeyboardInterrupt:
        print("\nStopped at once. Run the same command again to continue.")
    finally:
        database.close()


if __name__ == "__main__":
    main()
