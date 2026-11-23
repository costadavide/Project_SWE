from search_engine.analyzer import analyze
from collections import defaultdict
import heapq

def new_score_by_term():
    return defaultdict(int)

class BM25:
    def __init__(self, index, k1=1.2, b=0.75):
        self.index = index
        self.b = b
        self.k1 = k1
        self.avgdl = self.index.avg_doc_length()

    def search(self, query, k, allowed=None):
        """
        search in the index for docs that match the query and in the allowed set (if filters are given)
        and then return the k with the highest scores
        """
        term_pos = analyze(query, stem=True, query=True)
        unique_terms = set()
        for term, _ in term_pos:
            unique_terms.add(term)

        doc_to_score = defaultdict(int)
        score_breakdown = defaultdict(new_score_by_term)
        
        for term in unique_terms:
            idf = self.index.idf(term)
            postings = self.index.read_term(term)
            for doc in postings:
                if allowed is not None and doc not in allowed:
                    continue

                tf = self.index.term_freq(term, doc)
                dl = self.index.doc_length(doc)
                score = idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                doc_to_score[doc] += score
                score_breakdown[doc][term] = score
        
        best_k = heapq.nlargest(k, doc_to_score.items(), key=lambda item: (item[1], -item[0]))   

        best_k_breakdown = []
        best_k_profiles = []
        for i in range(len(best_k)):
            doc, score = best_k[i]
            profile_id = self.index.read_doc(doc)["profile_id"]
            best_k_profiles.append((profile_id, score))
            best_k_breakdown.append((profile_id, dict(score_breakdown[doc])))

        return best_k_profiles, best_k_breakdown

