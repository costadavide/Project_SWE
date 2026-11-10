"""Seed the `people` table with a random sample from the akzaidan/People dataset (MIT).

Usage: python scripts/seed_people.py [count]   (default 20000)
"""
import json
import os
import sys

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv(".env.local")

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
URL = "https://huggingface.co/datasets/akzaidan/People/resolve/main/profiles.parquet"

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
  profile_id        TEXT PRIMARY KEY,
  cohort            TEXT NOT NULL,
  job_family        TEXT NOT NULL,
  seniority         TEXT NOT NULL,
  seed_role         TEXT NOT NULL,
  age               INT,
  city              TEXT,
  state             TEXT,
  desired_salary    INT,
  experience_years  INT,
  skills            TEXT[] NOT NULL DEFAULT '{}',
  profile           JSONB NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS people_job_family_idx ON people (job_family);
CREATE INDEX IF NOT EXISTS people_seniority_idx  ON people (seniority);
CREATE INDEX IF NOT EXISTS people_state_idx      ON people (state);
CREATE INDEX IF NOT EXISTS people_skills_idx     ON people USING GIN (skills);
"""


def to_int(v):
    try:
        return round(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None


def to_row(r):
    p = json.loads(r.profile_json)
    u = p.get("user") or {}
    return (
        r.profile_id, r.cohort, r.job_family, r.seniority, r.seed_role,
        to_int(u.get("age")), u.get("city"), u.get("state"),
        to_int(u.get("desired_salary")), to_int(u.get("experience_years")),
        [str(s) for s in (u.get("skills") or [])], json.dumps(p),
    )


def main():
    print("Downloading dataset…")
    df = pd.read_parquet(URL)
    df = df.sample(n=COUNT, random_state=42)  # random but repeatable sample
    rows = [to_row(r) for r in df.itertuples(index=False)]
    print(f"Sampled {len(rows)} of the dataset's rows.")

    with psycopg.connect(os.environ.get("DATABASE_URL_UNPOOLED") or os.environ["DATABASE_URL"]) as conn:
        conn.execute(SCHEMA)
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO people (profile_id, cohort, job_family, seniority, seed_role, age,
                                       city, state, desired_salary, experience_years, skills, profile)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::text[], %s::jsonb)
                   ON CONFLICT (profile_id) DO NOTHING""",
                rows,
            )
        count = conn.execute("SELECT count(*) FROM people").fetchone()[0]
    print(f"Done. people table now has {count} rows.")


if __name__ == "__main__":
    main()
