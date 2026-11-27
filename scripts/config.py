"""Settings shared by the search scripts (search_cli.py and the try_* scripts).

Change a value here and every one of those scripts uses it.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# the full corpus: profiles.jsonl, the embeddings and the lexical index
DIR = PROJECT_ROOT / "data" / "full"
INDEX_PATH = DIR / "lexical.pkl"  # BM25 / BM25F index of the full corpus

K = 5        # profiles shown
POOL = 100   # profiles retrieved and given to the reranker
DEPTH = 200  # profiles each method gives to the fusion

# semantic score = weighted sum of the similarities to each part of the profile
WEIGHTS = {"core": 0.80, "projects": 0.15, "interests": 0.05}

# how profile text is built; must match TEXT_OPTIONS in scripts/embed_people.py,
# the options the embeddings were made with
TEXT_OPTIONS = {"include_seed_role": False, "include_bio": True}  # "no_seed_role"

# the hard constraints build_allowed uses
FILTERS = ("location", "min_years_experience", "max_years_experience", "excluded_skills")
