# Search engine roadmap

Roadmap for the classic search engine in `search_engine/` (BM25F, query language,
filters, fusion with embeddings, LambdaMART). Written 2026-11-20. Deadline 2026-11-27.

Each step lists what it is, the theory to learn, what it depends on, and a done-check.

## Before you start

### Evaluate on the 1,000 judged profiles

`tests/retrieval_1000/judgments.json` grades only the 1,000 profiles in
`profile_ids.json`; any profile not listed counts as grade 0. Ranking the full 10k
would count relevant but ungraded profiles as wrong and lower every method's NDCG.
`check_profile_set` in `tests/evaluate_retrieval_1000.py` already refuses other sets.

- Main comparison table (BM25 / semantic / fusion / LTR): the 1,000-profile set.
- Demo and latency: the 10k corpus and the full 248k.

### What the Neon to AWS move affects

The database is now AWS RDS PostgreSQL 18 (checked 2026-11-20: `people` has
248,522 rows, same columns, all 1,000 judged IDs present).

- SQL and psycopg stay the same; only the connection string changes.
- `.env.local` now sets `DATABASE_URL` to AWS; the old Neon URLs are kept as
  `NEON_DATABASE_URL` and `NEON_DATABASE_URL_UNPOOLED` (unused). Teammates must
  make the same change in their own `.env.local`, which is not in Git.
- The RDS security group may allow only your IP; teammates may need theirs added.
- **Design rule:** only `corpus.py` talks to the database. It writes a snapshot
  once (profile JSON plus structured columns), and everything downstream reads
  that file. Do the fixed-seed sampling in Python over a sorted ID list, not with
  SQL `random()`, so it is reproducible on any database.

## Build order

### 1. Corpus snapshot — `search_engine/corpus.py`
- **What:** 1k judged profiles + 9k seeded random ones, cached to `data/corpus/profiles.jsonl`.
- **Theory:** reproducible sampling; external document IDs vs internal integer IDs; read-only access.
- **Depends on:** database access.
- **Done when:** two runs produce identical files; all 1,000 judged IDs present; no other module imports psycopg.

### 2. Analyzer — `search_engine/analyzer.py`
- **What:** text to terms.
- **Theory:** tokenization, normalization, stopwords, stemming vs lemmatization (Porter/Snowball), over- and under-stemming, special tokens (`c++`, `c#`, `node.js`), same analyzer for queries and documents.
- **Depends on:** nothing.
- **Done when:** tests show `Engineering` and `engineers` share a stem, `c++` survives, stopwords are removed.

### 3. Fields — `search_engine/fields.py`
- **What:** profile to `title`, `skills`, `summary`, `education`, `certifications`.
- **Theory:** fielded documents; why a title match is stronger evidence than a bio match.
- **Depends on:** 1. Reuse helpers from `semantic_search/profile_text1.py`.
- **Done when:** a sample profile prints sensible fields; protected attributes are excluded.

### 4. Inverted index — `search_engine/index.py`
- **What:** term to postings.
- **Theory:** postings lists, term frequency, document frequency, positional postings, per-field lengths, doc store, serialization tradeoffs.
- **Depends on:** 2, 3.
- **Done when:** every posting of a 3-document toy corpus matches your hand prediction; save then load is identical.

### 5. BM25 — `search_engine/bm25.py`
- **Theory:** TF-IDF and its weaknesses, probabilistic relevance model, IDF, TF saturation (k1), length normalization (b), term-at-a-time vs document-at-a-time.
- **Depends on:** 4.
- **Done when:** a hand-computed score matches the code to 1e-6.

### 6. BM25F and phrases
- **Theory:** why summing per-field BM25 breaks saturation; BM25F weights TF per field and saturates once; phrase matching by position intersection.
- **Depends on:** 5.
- **Done when:** a title match outranks a bio match; `"software engineer"` does not match "engineer ... software".

### 7. Metrics and evaluation — `search_engine/metrics.py`, `scripts/evaluate_methods.py`
- **Theory:** NDCG@10 (graded gain, log discount, ideal DCG), MRR, P@5, p50/p95 latency. Move existing code from `tests/evaluate_retrieval_1000.py`.
- **Depends on:** 5.
- **Done when:** first BM25 vs semantic table on the 1,000 set. Build early so every later step is measured.

### 8. Query parser — `search_engine/query.py`
- **What:** `python backend -"software engineer" seniority:senior state:CA` to `StructuredQuery`.
- **Theory:** query languages, tokenizing a small grammar, lexical terms vs field filters. Agree the contract with your colleague.
- **Depends on:** 2.
- **Done when:** tests cover phrases, `-term`, `-"phrase"`, `field:value`, malformed input.

### 9. Boolean evaluation and filters — `search_engine/query.py`, `search_engine/filters.py`
- **Theory:** boolean retrieval, postings intersection and difference, bitmap/set pre-filters, filter-then-rank vs rank-then-filter.
- **Depends on:** 4, 8.
- **Done when:** no result contains a `must_not` term; filters match a direct check on the snapshot.

### 10. Corpus embeddings — `scripts/build_corpus_embeddings.py`
- **Theory:** little new; keep IDs aligned with the index.
- **Depends on:** 1.
- **Done when:** vectors align with doc IDs. Start in the background right after step 1 (30–60 min on CPU).

### 11. Fusion — `search_engine/fusion.py`
- **Theory:** score incomparability, min-max normalization, weighted linear fusion, Reciprocal Rank Fusion and its k, candidate generation (top-200 union).
- **Depends on:** 6, 7, 10.
- **Done when:** fusion row in the table, and you can explain why it beats or loses to each method.

### 12. LTR features — `search_engine/ltr/features.py`
- **Theory:** ranking feature engineering, per-query normalization, leakage.
- **Depends on:** 6, 10, 11.
- **Done when:** you can explain every feature for one (query, profile) pair.

### 13. Weak labels
- **Theory:** weak/distant supervision, label noise; never train on the 16 judged queries.
- **Depends on:** 1, 12.
- **Done when:** ~2,000 training queries with graded candidates, no overlap with test queries.

### 14. LambdaMART — `search_engine/ltr/train.py`
- **Theory:** pointwise vs pairwise vs listwise, RankNet, LambdaRank's NDCG-weighted gradients, gradient-boosted trees, query groups, grouped train/validation split.
- **Depends on:** 12, 13.
- **Done when:** LTR row in the table; feature ablation done.

### 15. API and UI — `api/app.py`, `web/index.html`
- **Theory:** REST basics with FastAPI; explainable ranking (breakdowns sum to the total).
- **Depends on:** 9, 11 (14 if ready).
- **Done when:** the plan's demo queries work in the browser.

### 16. Scale and latency
- **Theory:** index size, query cost, why postings length drives latency.
- **Depends on:** 6.
- **Done when:** p50/p95 for 10k vs 248k.

### 17. Integration and negation experiment
- **What:** plug in the colleague's LLM rewrite; compare against embeddings alone.
- **Depends on:** 8, 9, colleague's layer.

### 18. Report and README

## Priorities

Must-have: steps 1–9, 11, 15 (matches the plan's fallback). Stretch: 12–14, then 16–17.
