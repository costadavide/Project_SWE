# People Search Engine

A search engine over about 248,000 professional profiles. You type a request in
plain language, such as _"ICU nurse in Ohio with at least 5 years of experience"_,
and it returns the best-matching profiles, with the scores that put them there.

It combines three parts:

- a **local LLM** that reads the request, rewrites it as a clean search query and
  pulls out hard constraints (location, years of experience, skills to exclude);
- **retrieval**, by embeddings (semantic search), optionally fused with a
  classic BM25 lexical index;
- a **cross-encoder reranker** that reorders the best candidates.

## How a search works

```
query ──► LLM (LM Studio) ──► rewritten query + filters
                                   │
                                   ▼
             filters ──► allowed profiles (location, years, excluded skills)
                                   │
                                   ▼
         retrieval: semantic search (default) or BM25 + semantic fused with RRF
                                   │  top POOL profiles (default 100)
                                   ▼
             reranker (BAAI/bge-reranker-v2-m3) reads the original query
                                   │
                                   ▼
                          top K profiles shown (default 5)
```

1. **Query understanding** (`scripts/ask_llm.py`). The query goes to a small
   model (`qwen/qwen3-1.7b`) served by LM Studio. It returns JSON with a rewritten
   `semantic_query` and `hard_constraints`. If LM Studio is not running, the
   query is searched as typed, with no filters.
2. **Filters** (`semantic_search/filters.py`). `build_allowed` turns the
   constraints into the set of profiles that may be returned: city and/or state,
   minimum and maximum years of experience, and excluded skills. An excluded
   skill is looked up as a phrase in the lexical index, so it removes every
   profile that mentions it in any indexed field (titles, skills, summary,
   education, certifications), not only in the skills list. "Python" also
   excludes "Python 3", but "C" does not exclude "C++".
3. **Retrieval.**
   - _Semantic_ (`semantic_search/searcher.py`): each profile has three
     embeddings made with `Qwen/Qwen3-Embedding-0.6B`, one for the core profile,
     one for projects and one for career interests. A profile's score is
     `0.80 × core + 0.15 × projects + 0.05 × interests` cosine similarity.
   - _Lexical_ (`search_engine/`): a positional inverted index over six fields
     (current title, past titles, skills, summary, education, certifications),
     scored with BM25 or BM25F.
   - _Fusion_ (`search_engine/fusion.py`): Reciprocal Rank Fusion of the
     lexical and semantic top 200.
4. **Reranking** (`semantic_search/reranker.py`). The cross-encoder reads the
   query together with each candidate's text and gives the final order.

Protected and filter-style attributes (age, ethnicity, legal status,
sponsorship, salary) are never embedded or indexed, and are removed from the
local snapshot.

## Project layout

| Folder                  | Contents                                                                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/`              | Command-line entry points: data import, embedding, the search CLI, evaluation and experiments (`try_*.py`). Shared search settings are in `scripts/config.py`. |
| `semantic_search/`      | Profile-to-text conversion, the semantic searcher, filters and the reranker.                                                                                   |
| `search_engine/`        | The lexical engine: corpus snapshot, analyzer, fields, inverted index, BM25/BM25F, fusion and metrics.                                                         |
| `tests/`, `evaluation/` | Graded test queries and their results.                                                                                                                         |
| `data/`                 | Local snapshot, embeddings and indexes. Not in Git.                                                                                                            |
| `.env.local`            | Database connection string. Not in Git.                                                                                                                        |

## 1. Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env.local` with the connection string of the PostgreSQL database
(AWS RDS).

## 2. Get the data

The search engine never queries the database while searching. It reads a local
snapshot, `data/full/` by default, which you build once with the steps below. Every
command runs from the project root with the virtual environment active.

### 2.1 Profiles and embeddings in the database (usually already done)

The shared database already holds 248,522 profiles in `people` and their
vectors in `profile_embeddings`. Skip to 2.2 unless you are setting up a new
database.

```bash
python -m scripts.seed_people 20000   # download the akzaidan/People dataset
                                      # (Hugging Face) and insert a random sample
python -m scripts.embed_people        # embed every profile that has no vectors yet
```

`embed_people` takes several hours for the whole table. Useful options:

```bash
python -m scripts.embed_people --status     # show progress, change nothing
python -m scripts.embed_people --limit 500  # at most 500 profiles, then stop
python -m scripts.embed_people --refresh    # also redo profiles whose text changed
```

Press Ctrl+C to stop after the current batch (256 profiles) is saved, and run
the same command again to continue. On a Mac,
`caffeinate -i python -m scripts.embed_people` keeps the computer awake.

### 2.2 Download the snapshot

```bash
python -m search_engine.corpus --all
```

This reads every profile and its three embeddings (read-only) and writes:

```
data/full/profiles.jsonl            one profile per line (~800 MB)
data/full/profile_ids.json          the profile IDs in the same order
data/full/embeddings_core.npy       float32, one row per profile (~1 GB each)
data/full/embeddings_projects.npy
data/full/embeddings_interests.npy
data/full/embeddings.json           model, dimension, missing-vector counts
data/full/metadata.json             sizes and build time
```

Plan for about 4 GB of disk. Row _i_ of every file is the same profile, which
is how the lexical and semantic parts stay aligned. The 1,000 judged profiles
of the tests come first.

`--no-embeddings` and `--embeddings-only` fetch only one of the two parts.

#### Only 10,000 profiles instead

If you do not need the whole database (less disk, faster start), fetch the
smaller corpus instead:

```bash
python -m search_engine.corpus        # 10,000 profiles into data/corpus/
```

It holds the 1,000 judged profiles plus 9,000 sampled with seed 42, in the same
file format as `data/full/`. The search engine reads `data/full/` by default,
so point it at the new folder by changing `DIR` in `scripts/config.py`:

```python
DIR = PROJECT_ROOT / "data" / "corpus"
```

`INDEX_PATH` follows `DIR`, so the lexical index then lives in
`data/corpus/lexical.pkl`. Change `DIR` back to `data/full` to search the whole
database again.

### 2.3 The lexical index

`lexical.pkl`, in the snapshot folder (`DIR`), is one inverted index over the
full profiles. It serves both the excluded-skills filter and the BM25 side of
fusion mode.

You do not have to build it by hand: the first run of `scripts.search_cli`
builds it from `profiles.jsonl` and saves it for later runs. It is built again
automatically when `profiles.jsonl` is newer than the saved index, so a new
snapshot never meets an old index. On the full corpus this first build takes a
while and needs several GB of RAM.

### 2.4 Start the LLM

1. Install [LM Studio](https://lmstudio.ai) and download the model
   `qwen/qwen3-1.7b`.
2. Load it and start the local server on the default port (`http://localhost:1234`).

You can check it with `python -m scripts.ask_llm "nurse in Texas, not Python"`,
which prints the JSON the search engine receives. The model name and URL are
at the top of `scripts/ask_llm.py`.

The embedding model (about 1.2 GB) and the reranker (about 2.3 GB) are
downloaded from Hugging Face on the first search and cached afterwards.

## 3. Run the search engine

```bash
python -m scripts.search_cli
```

After loading, it asks for queries until you type `exit`. It first shows what
the LLM understood, for example (the exact wording depends on the model):

```
Query (or 'exit'): registered nurse in Ohio with at least 5 years of experience, no pediatrics
Rewritten query: Registered nurse with at least 5 years of experience
Filters: {'location': 'Ohio', 'min_years_experience': 5, 'excluded_skills': ['pediatrics']} -> <n> of 248,522 profiles
```

For each result it prints the profile ID, its rank before reranking, the
reranker score, the retrieval scores and the profile text. With the
10,000-profile corpus the filter line counts out of 10,000 instead.

### Options

| Flag                       | Effect                                                         |
| -------------------------- | -------------------------------------------------------------- |
| `--retrieval fusion`       | BM25 + embeddings fused with RRF, instead of embeddings only   |
| `--lexical bm25f`          | use BM25F in the fusion (only with `--retrieval fusion`)       |
| `--query original`         | retrieve with the original query instead of the rewritten one  |
| `--rerank-query rewritten` | the reranker reads the rewritten query instead of the original |
| `--no-rerank`              | show the retrieval order, without the reranker                 |
| `--no-llm`                 | do not call the LLM: original query, no filters                |
| `--k 10`                   | number of profiles shown (default 5)                           |
| `--pool 200`               | profiles retrieved and given to the reranker (default 100)     |

Defaults for these, the semantic weights and the snapshot folder (`DIR`, see
section 2.2) are in `scripts/config.py`. `TEXT_OPTIONS` there must match the one in
`scripts/embed_people.py`, since the reranker reads the same text the
embeddings were made from.

## 4. Evaluation

Two sets of graded queries measure the ranking.

**Method comparison** (`scripts/evaluate_methods.py`). Runs BM25, BM25F,
semantic, weighted semantic and fusion on the 16 graded queries in
`tests/retrieval_1000/judgments.json`, ranking only the 1,000 judged profiles
(any unjudged profile would count as not relevant). It needs the 10,000-profile
corpus and its index:

```bash
python -m search_engine.corpus        # data/corpus/
python -m scripts.try_index           # data/index/lexical.pkl
python -m scripts.evaluate_methods
python -m scripts.evaluate_methods --methods bm25f semantic --show-top 3
```

It prints NDCG@10, MRR@10, P@5, R@10 and p50/p95 latency per method and per
query type, and saves the details to `data/results/evaluate_methods.json`.

- **NDCG@10**: quality of the whole top 10 using the grades (1.0 = best possible order).
- **MRR@10**: 1 / rank of the first clearly relevant profile.
- **P@5**: share of the top five that is relevant or partially relevant.
- **R@10**: share of the clearly relevant profiles found in the top 10.

In `judgments.json`, 2 = clearly relevant (already does this kind of work),
1 = partially relevant (adjacent role, or trained for it without having held
it), unlisted = 0. The grades were made by one reviewer, and new queries must
be graded against all 1,000 profiles by hand.

**Earlier prototype.** `scripts/build_index.py`, `scripts/inspect_profile.py`,
`scripts/evaluate.py` and `tests/evaluate_retrieval_1000.py` belong to the
first version, which embedded a small sample into `data/` and scored
`0.7 × full profile + 0.3 × experience`. They still run:

```bash
python -m scripts.build_index --ids tests/retrieval_1000/profile_ids.json --data-dir data/sample_1000
python -m tests.evaluate_retrieval_1000
python -m scripts.evaluate        # rank of one intended profile per case in evaluation/cases.json
```

## Experiments

The `scripts/try_*.py` files are the steps that led to `search_cli.py`, and
each one matches a set of `search_cli` flags (listed at the top of
`scripts/search_cli.py`). `try_index.py --toy` prints a full index for three
hand-made profiles, which is a quick way to see what the lexical engine stores.
