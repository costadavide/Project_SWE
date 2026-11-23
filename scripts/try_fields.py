"""Print what search_engine.fields.profile_to_dict returns for profiles from the database.

Throwaway helper for checking fields.py by eye. Read-only.

Usage (from the project root):
    python -m scripts.try_fields                                  # the toxicologist
    python -m scripts.try_fields syn2_visa_constrained_0142771 <other_id> ...
    python -m scripts.try_fields --random 3                       # 3 random profiles
    python -m scripts.try_fields --raw                            # also print the source JSON
"""

import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from search_engine.fields import profile_to_dict

DEFAULT_ID = "syn2_visa_constrained_0142771"

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env.local")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("profile_ids", nargs="*", help=f"profile IDs to show (default {DEFAULT_ID})")
parser.add_argument("--random", type=int, metavar="N", help="show N random profiles instead")
parser.add_argument("--raw", action="store_true", help="also print the parts of the source JSON that fields.py reads")
args = parser.parse_args()

database_url = os.getenv("DATABASE_URL_UNPOOLED") or os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("No database URL found in .env.local")

with psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10) as connection:
    connection.execute("SET TRANSACTION READ ONLY")
    if args.random:
        rows = connection.execute(
            "SELECT profile_id, profile FROM people ORDER BY random() LIMIT %s", (args.random,)
        ).fetchall()
    else:
        ids = args.profile_ids or [DEFAULT_ID]
        found = {
            row["profile_id"]: row
            for row in connection.execute(
                "SELECT profile_id, profile FROM people WHERE profile_id = ANY(%s)", (ids,)
            ).fetchall()
        }
        for missing in [pid for pid in ids if pid not in found]:
            print(f"Not found: {missing}")
        rows = [found[pid] for pid in ids if pid in found]

for row in rows:
    profile = row["profile"]
    print("=" * 80)
    print(row["profile_id"])
    if args.raw:
        source = {
            "user.bio": (profile.get("user") or {}).get("bio"),
            "user.skills": (profile.get("user") or {}).get("skills"),
            "experience": [
                {key: job.get(key) for key in ("title", "start_date", "end_date", "is_internship", "description")}
                for job in profile.get("experience") or []
            ],
            "education": [
                {key: entry.get(key) for key in ("degree", "field_of_study")}
                for entry in profile.get("education") or []
            ],
            "certifications": [cert.get("name") for cert in profile.get("certifications") or []],
            "projects": [project.get("name") for project in profile.get("projects") or []],
        }
        print("\n--- source ---")
        print(json.dumps(source, indent=2, ensure_ascii=False))
        print("\n--- fields ---")
    fields = profile_to_dict(profile)
    print(json.dumps(fields, indent=2, ensure_ascii=False))
    for name, value in fields.items():
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            print(f"  note: '{name}' is not a list of non-empty strings: {value!r}")
