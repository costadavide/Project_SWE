"""Run BM25 / BM25F queries, with phrase filters, and print the ranked results.

Default: load the saved index (data/index/lexical.pkl, made by
python -m scripts.try_index) and run a small set of example queries scored
with BM25F. Phrases in the query are filters, using Index.phrase_docs:
    "software engineer"   only profiles containing that exact phrase
    -"travel nurse"       drop profiles containing that phrase
    -senior               drop profiles containing that word
The words of a "phrase" are still scored; excluded words and phrases are not.
This little query syntax is only for testing until query.py (roadmap step 8).

With --toy: build small toy indexes and check BM25 / BM25F scores against the
formulas computed by hand, and phrase_docs against the expected matches.

Usage (from the project root):
    python -m scripts.try_bm25                                   # example queries, BM25F
    python -m scripts.try_bm25 "icu nurse" "python developer"    # your own queries
    python -m scripts.try_bm25 "python" --phrase "software engineer" --not-phrase "data scientist"
    python -m scripts.try_bm25 --method both                     # BM25 vs BM25F side by side
    python -m scripts.try_bm25 --method bm25 --k1 2.0 --b 0.3
    python -m scripts.try_bm25 "registered nurse" --filter state=OH seniority=senior
    python -m scripts.try_bm25 "data scientist" --judged         # only the 1,000 judged profiles
    python -m scripts.try_bm25 --toy                             # hand checks

Quotes on Windows: PowerShell drops double quotes inside an argument, so
    python -m scripts.try_bm25 '"software engineer" python'
may arrive without the inner quotes. Use --phrase / --not-phrase instead
(they apply to every query), or run from cmd.exe with \"...\".

--method both always prints titles (reads the corpus snapshot, a few seconds),
because comparing two rankings of bare profile IDs tells you nothing.
--k1 and --b only change the plain BM25 scorer; BM25F uses its own per-field
defaults from search_engine/bm25.py.
"""

import argparse
import math
import re
import time

from scripts.try_index import INDEX_PATH, PROFILES
from search_engine.bm25 import BM25, BM25F
from search_engine.index import Index

DEFAULT_QUERIES = [
    "icu nurse",
    '"registered nurse" -"travel nurse"',
    '"software engineer" python backend',
    "python backend developer",
    '"data scientist" -senior',
    '"machine learning" engineer',
    '"head of engineering"',
    "c++ embedded",
]
N_JUDGED = 1000  # the corpus file puts the 1,000 judged profiles first
FIELD_SHORT = {
    "current_title": "title",
    "past_titles": "past",
    "skills": "skills",
    "summary": "summ",
    "education": "edu",
    "certifications": "cert",
}


def allowed_docs(index, filters, judged):
    """Doc ids that pass every key=value filter (and, with judged, are judged). None = no restriction."""
    if not filters and not judged:
        return None
    allowed = set()
    for doc_id in range(len(index)):
        info = index.read_doc(doc_id)
        if judged and doc_id >= N_JUDGED:
            continue
        if all(str(info.get(key, "")).lower() == value.lower() for key, value in filters.items()):
            allowed.add(doc_id)
    return allowed


def parse_query(text):
    """Split a test query into (words to score, must phrases, must_not phrases).

    "a b"   -> must phrase, its words are also scored
    -"a b"  -> must_not phrase
    -word   -> must_not (a one-word phrase)
    Temporary: the real parser is search_engine/query.py (roadmap step 8).
    """
    must_not = re.findall(r'-"([^"]*)"', text)
    text = re.sub(r'-"[^"]*"', " ", text)
    must = re.findall(r'"([^"]*)"', text)
    text = text.replace('"', " ")
    words = []
    for word in text.split():
        if word.startswith("-") and len(word) > 1:
            must_not.append(word[1:])
        else:
            words.append(word)
    return " ".join(words), [p for p in must if p.strip()], [p for p in must_not if p.strip()]


def apply_phrases(index, allowed, must, must_not):
    """Narrow the allowed set with phrase_docs. Returns (allowed, report lines)."""
    report = []
    for phrase in must:
        docs = index.phrase_docs(phrase)
        if docs is None:
            report.append(f'"{phrase}": only stopwords, ignored')
            continue
        allowed = set(docs) if allowed is None else allowed & docs
        report.append(f'"{phrase}": {len(docs)} docs')
    for phrase in must_not:
        docs = index.phrase_docs(phrase)
        if docs is None:
            report.append(f'-"{phrase}": only stopwords, ignored')
            continue
        if allowed is None:
            allowed = set(range(len(index)))
        allowed = allowed - docs
        report.append(f'-"{phrase}": {len(docs)} docs removed')
    return allowed, report


def load_titles():
    """profile_id -> current title, read from the corpus snapshot (only for printing)."""
    from search_engine.corpus import load_corpus
    from search_engine.fields import profile_to_dict

    titles = {}
    for row in load_corpus():
        title = profile_to_dict(row["profile"] or {}).get("current_title")
        titles[row["profile_id"]] = title[0] if title else ""
    return titles


def term_total(value):
    """A breakdown value is a number (BM25) or {field: part} (BM25F)."""
    return sum(value.values()) if isinstance(value, dict) else value


def format_parts(parts):
    """'icu=10.16  nurs=6.92' for BM25, 'icu=10.16(title 8.1, summ 2.0)  ...' for BM25F."""
    pieces = []
    for term, value in sorted(parts.items(), key=lambda item: -term_total(item[1])):
        text = f"{term}={term_total(value):.2f}"
        if isinstance(value, dict):
            fields = sorted(value.items(), key=lambda item: -item[1])
            text += "(" + ", ".join(f"{FIELD_SHORT.get(f, f)} {v:.2f}" for f, v in fields) + ")"
        pieces.append(text)
    return "  ".join(pieces)


def run_query(scorer, query, k, allowed):
    start = time.perf_counter()
    profiles, breakdowns = scorer.search(query, k=k, allowed=allowed)
    return profiles, breakdowns, time.perf_counter() - start


def print_results(name, query, profiles, breakdowns, seconds, titles=None):
    print(f"\n=== {name} {query!r}: {len(profiles)} results in {seconds * 1000:.1f} ms ===")
    for rank, ((profile_id, score), (_, parts)) in enumerate(zip(profiles, breakdowns), start=1):
        title = f"  {titles.get(profile_id, '')!r}" if titles is not None else ""
        print(f"{rank:2}. {score:7.3f}  {profile_id}{title}")
        print(f"      {format_parts(parts)}")


def print_comparison(query, k, bm25_result, bm25f_result, titles):
    """Side by side: rank, BM25 hit, BM25F hit, and where each BM25F hit was in the BM25 list."""
    p1, _, t1 = bm25_result
    p2, b2, t2 = bm25f_result
    rank_in_bm25 = {pid: rank for rank, (pid, _) in enumerate(p1, start=1)}
    shared = len(set(rank_in_bm25) & {pid for pid, _ in p2})
    print(f"\n=== {query!r}   top-{k} overlap {shared}/{min(len(p1), len(p2))}   "
          f"BM25 {t1 * 1000:.1f} ms, BM25F {t2 * 1000:.1f} ms ===")
    print(f"{'':3} {'BM25':44} | {'BM25F':44} {'was':>4}")
    width = 34
    for rank in range(max(len(p1), len(p2))):
        left = right = ""
        if rank < len(p1):
            pid, score = p1[rank]
            left = f"{score:6.2f} {titles.get(pid, '')[:width]}"
        moved = ""
        if rank < len(p2):
            pid, score = p2[rank]
            right = f"{score:6.2f} {titles.get(pid, '')[:width]}"
            moved = f"#{rank_in_bm25[pid]}" if pid in rank_in_bm25 else "new"
        print(f"{rank + 1:2}. {left:44} | {right:44} {moved:>4}")
    if p2:
        print(f"    BM25F #1 breakdown: {format_parts(b2[0][1])}")


# ---------------------------------------------------------------- toy checks

def hand_bm25(index, term, doc_id, k1, b):
    """BM25 for one term in one doc, straight from the raw index structures."""
    n = len(index)
    df = len(index.postings.get(term, {}))
    tf = sum(len(positions) for positions in index.postings[term][doc_id].values())
    dl = sum(index.field_lengths[doc_id].values())
    avgdl = sum(sum(lengths.values()) for lengths in index.field_lengths.values()) / n
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def hand_bm25f(index, term, doc_id, scorer):
    """BM25F for one term in one doc: per-field normalised tf, weighted sum, saturate once."""
    n = len(index)
    df = len(index.postings.get(term, {}))
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    tf = 0.0
    for field, positions in index.postings[term][doc_id].items():
        lengths = [index.field_lengths[d][field] for d in index.field_lengths if index.field_lengths[d].get(field)]
        avg = sum(lengths) / len(lengths)
        b = scorer.b[field]
        tf += scorer.w[field] * len(positions) / (1 - b + b * index.field_lengths[doc_id][field] / avg)
    return idf * tf * (scorer.k1 + 1) / (tf + scorer.k1)


def toy_profile(profile_id, title=None, past_titles=(), description=None):
    """A minimal profile: current job (with optional description) plus older jobs."""
    jobs = []
    if title:
        jobs.append({"title": title, "start_date": "01/2022", "end_date": None, "description": description})
    for year, past in enumerate(past_titles, start=2010):
        jobs.append({"title": past, "start_date": f"01/{year}", "end_date": f"12/{year}"})
    return {"profile_id": profile_id, "profile": {"user": {"bio": "", "skills": []}, "experience": jobs}}


PHRASE_PROFILES = [
    toy_profile("p_match", "Senior Software Engineer"),
    toy_profile("p_order", "Engineer Software"),
    toy_profile("p_gap", past_titles=["Software", "Engineering Lead"]),
    toy_profile("p_fields", "Software Lead", description="A great engineer."),
    toy_profile("p_head_of", "Head of Engineering"),
    toy_profile("p_head", "Head Engineer"),
    toy_profile("p_fullstack", "Full-Stack Engineer"),
    toy_profile("p_nurse", "Nurse"),
]

# phrase -> expected profile ids (None = phrase made only of stopwords)
PHRASE_EXPECTED = {
    "software engineer": ["p_match"],         # not wrong order, not across titles, not across fields
    "head of engineering": ["p_head_of"],     # the stopword keeps its slot
    "full-stack engineer": ["p_fullstack"],   # stacked hyphen tokens
    "nurse": ["p_nurse"],                     # one-word phrase
    "zzz unknown": [],                        # unknown word
    "of the": None,                           # only stopwords
}


def run_toy_phrases():
    index = Index(PHRASE_PROFILES)
    ok = True
    print("\n=== phrase_docs on the phrase toy index ===")
    for phrase, expected in PHRASE_EXPECTED.items():
        docs = index.phrase_docs(phrase)
        got = None if docs is None else sorted(index.read_doc(d)["profile_id"] for d in docs)
        mark = "ok " if got == expected else "BAD"
        ok = ok and got == expected
        print(f"{mark} {phrase!r:24} -> {got}   (expected {expected})")
    return ok


def run_toy(k1, b):
    index = Index(PROFILES)
    doc_of = {index.read_doc(d)["profile_id"]: d for d in range(len(index))}
    scorers = [("BM25", BM25(index, k1=k1, b=b)), ("BM25F", BM25F(index))]
    ok = True
    for name, scorer in scorers:
        for query in ["nurse", "python", "icu nurse", "c++ python"]:
            profiles, breakdowns, seconds = run_query(scorer, query, 10, None)
            print_results(name, query, profiles, breakdowns, seconds)
            for (profile_id, score), (_, parts) in zip(profiles, breakdowns):
                for term, value in parts.items():
                    if name == "BM25":
                        expected = hand_bm25(index, term, doc_of[profile_id], k1, b)
                    else:
                        expected = hand_bm25f(index, term, doc_of[profile_id], scorer)
                    if abs(expected - term_total(value)) > 1e-9:
                        ok = False
                        print(f"   MISMATCH {term} in {profile_id}: code {term_total(value)} vs hand {expected}")
                total = sum(term_total(value) for value in parts.values())
                if abs(score - total) > 1e-9:
                    ok = False
                    print(f"   MISMATCH total of {profile_id}: {score} vs sum of parts {total}")

        empty, _ = scorer.search("nurse", k=10, allowed=set())
        unknown, _ = scorer.search("zzzunknownterm", k=10)
        only_ohio, _ = scorer.search("python", k=10, allowed=allowed_docs(index, {"state": "OH"}, False))
        print(f"\n{name} empty allowed set -> {empty}")
        print(f"{name} unknown term       -> {unknown}")
        print(f"{name} python, state=OH   -> {only_ohio}")
        ok = ok and empty == [] and unknown == [] and [p for p, _ in only_ohio] == ["toy_career_changer"]

    ok = run_toy_phrases() and ok
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Run BM25 / BM25F queries on the saved index.")
    parser.add_argument("queries", nargs="*", help="queries to run (default: a small example set)")
    parser.add_argument("--method", choices=["bm25", "bm25f", "both"], default="bm25f",
                        help="which scorer to run (default bm25f); both = side-by-side comparison")
    parser.add_argument("--phrase", nargs="+", default=[], help="phrases every result must contain (all queries)")
    parser.add_argument("--not-phrase", nargs="+", default=[], help="phrases no result may contain (all queries)")
    parser.add_argument("--k", type=int, default=10, help="results per query (default 10)")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 only")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 only")
    parser.add_argument("--filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only docs whose doc-store value matches, e.g. state=CA seniority=senior")
    parser.add_argument("--judged", action="store_true", help="only rank the 1,000 judged profiles")
    parser.add_argument("--no-titles", action="store_true", help="do not print current titles (skips reading the corpus)")
    parser.add_argument("--toy", action="store_true", help="hand checks on toy indexes")
    args = parser.parse_args()

    if args.toy:
        run_toy(args.k1, args.b)
        return

    if not INDEX_PATH.exists():
        raise SystemExit(f"{INDEX_PATH} not found. Build it with: python -m scripts.try_index")
    start = time.perf_counter()
    index = Index.load(INDEX_PATH)
    print(f"Loaded {INDEX_PATH} ({len(index)} docs) in {time.perf_counter() - start:.1f} s")

    filters = dict(item.split("=", 1) for item in args.filter)
    allowed = allowed_docs(index, filters, args.judged)
    if allowed is not None:
        print(f"Filters {filters or ''}{' + judged only' if args.judged else ''}: {len(allowed)} docs allowed")

    titles = None if (args.no_titles and args.method != "both") else load_titles()
    bm25 = BM25(index, k1=args.k1, b=args.b)
    bm25f = BM25F(index)
    print(f"BM25:  k1={bm25.k1}  b={bm25.b}  avgdl={bm25.avgdl:.2f}")
    print(f"BM25F: k1={bm25f.k1}  w={bm25f.w}")
    print(f"       b={bm25f.b}")

    for query in args.queries or DEFAULT_QUERIES:
        words, must, must_not = parse_query(query)
        must += args.phrase
        must_not += args.not_phrase
        start = time.perf_counter()
        query_allowed, report = apply_phrases(index, allowed, must, must_not)
        phrase_ms = (time.perf_counter() - start) * 1000
        print(f"\n##### {query!r}  ->  scored words {words!r}")
        if report:
            allowed_text = "all" if query_allowed is None else len(query_allowed)
            print(f"      phrases ({phrase_ms:.1f} ms): {';  '.join(report)}  ->  {allowed_text} docs allowed")
        if not words.strip():
            print("      nothing left to score")
            continue
        if args.method == "both":
            print_comparison(words, args.k, run_query(bm25, words, args.k, query_allowed),
                             run_query(bm25f, words, args.k, query_allowed), titles)
        else:
            name, scorer = ("BM25", bm25) if args.method == "bm25" else ("BM25F", bm25f)
            print_results(name, words, *run_query(scorer, words, args.k, query_allowed), titles)


if __name__ == "__main__":
    main()
