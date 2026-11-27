"""Ranking metrics for graded judgments (2 = relevant, 1 = partly, 0 = not).

score_query gives precision@5, recall@10, RR@10 and NDCG@10 for one query
(gain 2^grade - 1, discount log2(rank + 1)); mean_metrics averages them over
queries. Profiles without a judgment count as grade 0.
"""

import math

RELEVANT = 2    

def precision_at_k(grades, k):
    """Share of the top k with grade > 0; missing ranks count as not useful."""
    return sum(grades[i] > 0 for i in range(min(k, len(grades)))) / k

def recall_at_k(grades, good_results, k):
    """Share of the grade-2 profiles (good_results of them) found in the top k."""
    return sum(grades[i] == RELEVANT for i in range(min(k, len(grades)))) / good_results if good_results else 0.0

def rr_at_k(grades, k):
    """1 / rank of the first grade-2 result in the top k, 0.0 if none (mean = MRR)."""
    for i in range(min(k, len(grades))):
        if grades[i] == RELEVANT:
            return 1 / (i+1)
    return 0.0

def dcg_at_k(grades, k):
    """Discounted cumulative gain of the top k."""
    gain = 0
    for i in range(min(k, len(grades))):
        gain += (2**grades[i] - 1) / math.log2(i + 1 + 1)
    return gain 

def ndcg_at_k(grades, ideal_order, k):
    """DCG divided by the DCG of the ideal ordering of ideal_order, in [0, 1]."""
    ideal = sorted(ideal_order, reverse=True)
    idcg = dcg_at_k(ideal, k)
    gain = dcg_at_k(grades, k)
    return gain / idcg if idcg else 0.0


def grades_for(ranked_ids, judgments):
    """Grades of the ranked profile_ids, in rank order (unjudged = 0)."""
    return [judgments.get(ids, 0) for ids in ranked_ids]

def score_query(ranked_ids, judgments, k=10, precision_k=5):
    """All metrics for one ranked list against judgments = {profile_id: grade}."""
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
    """Average of each metric over a list of score_query results ({} if empty)."""
    res = {}
    for metrics in queries:
        for key, value in metrics.items():
            res[key] = res.get(key, 0.0) + value

    return {key: val/len(queries) for key, val in res.items()} if len(queries) else {}
    