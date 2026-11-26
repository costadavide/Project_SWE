from collections import defaultdict

class Fusion:
    def __init__(self, searchers, method="rrf", rrf_k=60, depth=200):
        self.method = method
        if self.method != "rrf":
            raise ValueError("Method must be rrf (reciprocal rank fusion)")
        self.rrf_k = rrf_k
        self.depth = depth
        self.searchers = searchers

    def search(self, query, k, allowed=None):
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

