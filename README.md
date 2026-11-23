# Software Engineering Project

Semantic search over professional profiles using Qwen embeddings and cosine similarity.

## Folder structure

- `semantic_search/`: reusable functions, including profile-to-text conversion.
- `scripts/`: database import and interactive search scripts.
- `data/`: local profile samples and embeddings for future use (ignored by Git).
- `.env.local`: local database connection settings (ignored by Git).

## Run the search prototype

From the project root, activate the environment and run:

```bash
source .venv/bin/activate
python -m scripts.inspect_profile
```

The script reads up to 100 profiles (one known photographer plus 99 random
profiles), generates their embeddings, and accepts repeated queries. Enter
`exit` to stop. Each new run selects a new random sample and regenerates embeddings.

`profile_to_text()` sorts employment history by date (ongoing jobs first, then
newest start date), includes the user's city and state, and leaves out
protected and filter-style fields (age, ethnicity, legal status, sponsorship,
salary), company names, job locations, work location preferences and stated
career interests (what someone wants to do is not a qualification). Rebuild
the index after changing it.

`experience_to_text()` builds a second text from the job history only (titles
and descriptions). `build_index` embeds both, and search ranks profiles by
`0.7 × full-profile similarity + 0.3 × experience similarity` (the weight is
`experience_weight` in `index_metadata.json`; see `semantic_search/scoring.py`).
This puts people already doing a job above career changers whose skills match
but whose jobs do not. The experience score knows nothing about skills,
certifications or location, so a higher weight weakens those signals. Indexes
built before this change have no `experience_embeddings.npy` and are ranked by
full-profile similarity only.

Build a larger random index in its own folder with
`python -m scripts.build_index --sample-size 1000 --data-dir data/sample_1000`.

## Evaluate the saved index

With `data/profiles.json`, `data/embeddings.npy` and `data/index_metadata.json`
in place (created by `python -m scripts.build_index`), run from the project root:

```bash
source .venv/bin/activate
python -m scripts.evaluate
```

The script does not rebuild the index or contact the database. It checks that
profiles and vectors line up and that every case in `evaluation/cases.json`
has a unique ID and an intended profile present in the index. It then embeds
each query with the same instruction and normalization as `scripts/search.py`.

Each case is one query written from one saved profile, tagged with a query
type: `direct` (role and key skills), `paraphrase` (the same need in plain
words), `profile_details` (needs information beyond the role and skills, such
as certifications, bio, earlier roles, or the most recent role by date) or
`location` (a role that appears more than once, told apart by city and state). The
script prints, per case, the rank and cosine similarity of the intended
profile, then two summary numbers overall and per query type:

- **Hit rate@5**: fraction of queries whose intended profile is in the top five.
- **MRR@5**: mean of 1/rank for the intended profile, counting 0 when it is
  outside the top five (1.0 means it always ranked first).

Full results, including each query's top five profile IDs and scores, are
written to `evaluation/results/intended_profile_results.json` (change with
`--output`). Use `--data-dir` to evaluate another saved index.

This is a sanity test, not a relevance benchmark. Other profiles in the top
five may also be good matches (for example, a nurse manager for a nursing
query), so they are not counted as errors and no Precision@5 is reported.
The cases only apply to an index containing their intended profiles: a plain
`build_index` run samples new random profiles and the script will stop with
an error. To rebuild with the same profiles (for example after changing
`profile_to_text()`), run:

```bash
python -m scripts.build_index --ids data/profiles.json
```

## Graded test on 1000 profiles

`tests/retrieval_1000/` holds a stricter test on a fixed set of 1000 profiles
(`profile_ids.json`). For each of its 16 queries (13 `direct`, 2 `paraphrase`,
1 `location`), every one of the 1000 profiles was reviewed and graded in
`judgments.json`: 2 = clearly relevant (already does this kind of work),
1 = partially relevant (adjacent role, or a career changer trained for the
role who has not held it), unlisted = 0. Each query has a note explaining the
harder grading decisions.

Build the index for this profile set once (and again after changing
`profile_to_text()` or `experience_to_text()`), then run the test:

```bash
python -m scripts.build_index --ids tests/retrieval_1000/profile_ids.json --data-dir data/sample_1000
python -m tests.evaluate_retrieval_1000
```

Use `--data-dir` for another index built from the same IDs and
`--experience-weight 0.2` (for example) to try another weight without
rebuilding. The script stops with an error if the index holds a different set
of profiles, because unlisted profiles count as not relevant. It prints, per
query and overall:

- **nDCG@10**: quality of the whole top 10, using the grades (1.0 = the best
  possible order of the graded profiles).
- **MRR@10**: 1/rank of the first grade-2 profile.
- **P@5**: fraction of the top five with grade 1 or 2.
- **R@10**: fraction of the grade-2 profiles that appear in the top 10.

Full results are written to `tests/retrieval_1000/results/latest.json`. The
grades were made by one reviewer and only apply to this profile set; new
queries must be graded against all 1000 profiles by hand.

## Import profiles

From the project root:

```bash
python -m scripts.seed_people 20000
```

This downloads the source dataset and inserts profiles into the PostgreSQL database (AWS RDS). Configure
`DATABASE_URL` in `.env.local` before running either script (`DATABASE_URL_UNPOOLED`, if set, takes precedence).
