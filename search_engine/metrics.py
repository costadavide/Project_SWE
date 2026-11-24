import math

RELEVANT = 2    

def precision_at_k(grades, k):
    """Share of the top k results with grade > 0 (useful at all). Always divides by k,
    so a list shorter than k counts its missing ranks as not useful."""
    return sum(grades[i] > 0 for i in range(min(k, len(grades)))) / k

def recall_at_k(grades, good_results, k):
    """Share of all RELEVANT (grade 2) profiles that appear in the top k.
    good_results = how many grade-2 profiles exist for the query (0 gives 0.0)."""
    return sum(grades[i] == RELEVANT for i in range(min(k, len(grades)))) / good_results if good_results else 0.0

def rr_at_k(grades, k):
    """Reciprocal rank: 1 / rank of the first RELEVANT result in the top k, 0.0 if none.
    Averaged over queries (mean_metrics) it becomes MRR."""
    for i in range(min(k, len(grades))):
        if grades[i] == RELEVANT:
            return 1 / (i+1)
    return 0.0

def dcg_at_k(grades, k):
    """Discounted cumulative gain of the top k: sum of (2^grade - 1) / log2(rank + 1),
    with rank starting at 1."""
    gain = 0
    for i in range(min(k, len(grades))):
        gain += (2**grades[i] - 1) / math.log2(i + 1 + 1)
    return gain 

def ndcg_at_k(grades, ideal_order, k):
    """DCG of the ranking divided by the DCG of the best possible ranking, in [0, 1].
    ideal_order = the query's judged grades, in any order (sorted here); 0.0 if all are 0."""
    ideal = sorted(ideal_order, reverse=True)
    idcg = dcg_at_k(ideal, k)
    gain = dcg_at_k(grades, k)
    return gain / idcg if idcg else 0.0


def grades_for(ranked_ids, judgments):
    """Turn a ranked list of profile_ids into their grades, in the same order.
    judgments = {profile_id: grade}; profiles not listed get grade 0."""
    return [judgments.get(ids, 0) for ids in ranked_ids]

def score_query(ranked_ids, judgments, k=10, precision_k=5):
    """All metrics for one query: NDCG, RR and recall at k, precision at precision_k.
    ranked_ids = profile_ids in rank order, judgments = {profile_id: grade}."""
    grades = grades_for(ranked_ids, judgments)
    ideal = sorted(judgments.values(), reverse=True)[:k]
    good_res = sum(judgement == RELEVANT for judgement in judgments.values())

    return {
        f"precision@{precision_k}": precision_at_k(grades, precision_k),
        f"recall@{k}": recall_at_k(grades, good_res, k),
        f"rr@{k}": rr_at_k(grades, k),
        f"NDCG@{k}": ndcg_at_k(grades, ideal, k),
    }

def mean_metrics(queries):
    """Average each metric over a list of score_query dicts (one per query).
    Returns {} for an empty list."""
    res = {}
    for metrics in queries:
        for key, value in metrics.items():
            res[key] = res.get(key, 0.0) + value

    return {key: val/len(queries) for key, val in res.items()} if len(queries) else {}
    