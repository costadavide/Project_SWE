"""Reciprocal rank fusion (RRF) of searchers that share the search() interface.

Every searcher returns its top `depth` profiles for the same query and allowed
set, and a profile scores the sum over searchers of 1 / (rrf_k + rank).
Ties go to the better semantic rank, then to the lower profile_id.
"""

from collections import defaultdict

class Fusion:
    """searchers = {name: searcher}; only method="rrf" is supported."""
    def __init__(self, searchers, method="rrf", rrf_k=60, depth=200):
        self.method = method
        if self.method != "rrf":
            raise ValueError("Method must be rrf (reciprocal rank fusion)")
        self.rrf_k = rrf_k
        self.depth = depth
        self.searchers = searchers

    def search(self, query, k, allowed=None):
        """Top k fused profiles; breakdown = {searcher: (rank, contribution)} for each profile."""
        res = {}
        for searcher, obj in self.searchers.items():
            res[searcher] = obj.search(query=query, k=self.depth, allowed=allowed)[0]

        if self.method == "rrf":
            docs = defaultdict(float)
            breakdowns = defaultdict(dict)

            for searcher, results in res.items(): 
                for rank, (doc, _) in enumerate(results, start=1):
                    contr = 1/(self.rrf_k + rank)
                    docs[doc] += contr
                    breakdowns[doc][searcher] = (rank, contr) 

            sort_docs = sorted(docs.items(), key=lambda item: (-item[1], breakdowns[item[0]].get("semantic", (float("inf"), 0))[0], item[0]))
            profiles = sort_docs[:k]
            breakdown = [(doc, breakdowns[doc]) for doc, _ in profiles]
        return profiles, breakdown

