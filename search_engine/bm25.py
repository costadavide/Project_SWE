"""BM25 and BM25F ranking over the Index.

Both expose search(query, k, allowed=None) -> (profiles, breakdowns):
profiles = [(profile_id, score)] best first, breakdowns = the per-term parts
of each score. allowed (a set of doc_ids) limits scoring to pre-filtered docs.

BM25:  tf summed over all fields, one length normalization per doc.
BM25F: tf normalized per field (own b), weighted per field (title highest),
       summed and saturated once with k1.
"""

from search_engine.analyzer import analyze
from collections import defaultdict
import heapq

def new_score_by_term():
    """defaultdict factory for the score breakdowns."""
    return defaultdict(int)

def new_score_by_term_field():
    """defaultdict factory for the score breakdowns."""
    return defaultdict(new_score_by_term)

class BM25:
    """Classic BM25 (k1=1.2, b=0.75), with all fields treated as one bag of words."""
    def __init__(self, index, k1=1.2, b=0.75):
        self.index = index
        self.b = b
        self.k1 = k1
        self.avgdl = self.index.avg_doc_length()

    def search(self, query, k, allowed=None):
        """Top k docs among allowed (None = all); breakdown = {term: score}."""
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


class BM25F:
    """BM25F with per-field weights w and length normalization b (defaults below)."""
    def __init__(self, index, k1=1.2, b=None, w=None):
        self.index = index
        self.b = b
        self.k1 = k1
        self.w = w

        if self.b is None:
            self.b = {
                "current_title": 0.3, "skills": 0.5,
                "past_titles": 0.5, "certifications": 0.75,
                "education": 0.5, "summary": 0.75
            }
        if self.w is None:
            self.w = {
                "current_title": 3.0, "skills": 2.0,
                "past_titles": 1.5, "certifications": 1.5,
                "education": 1.0, "summary": 1.0
            }
        self.fields = list(self.index.n_docs_per_field.keys())
        self.avg_fl = {}
        for field in self.fields:
            self.avg_fl[field] = self.index.avg_field_length(field)

    def search(self, query, k, allowed=None):
        """Top k docs among allowed (None = all); breakdown = {term: {field: share of the term's score}}."""
        term_pos = analyze(query, stem=True, query=True)
        unique_terms = set()
        for term, _ in term_pos:
            unique_terms.add(term)

        doc_to_score = defaultdict(int)
        score_breakdown = defaultdict(new_score_by_term_field)
        
        for term in unique_terms:
            idf = self.index.idf(term)
            postings = self.index.read_term(term)
            for doc in postings:
                if allowed is not None and doc not in allowed:
                    continue
                tf = 0
                weighted_tf_by_field = {}
                for field in postings[doc]:
                    tf_f = len(postings[doc][field])
                    lf = self.index.read_field_length(doc, field)

                    tf_norm = tf_f / (1 - self.b[field] + self.b[field] * lf / self.avg_fl[field])
                    weighted_tf_by_field[field] = tf_norm * self.w[field]
                    tf += weighted_tf_by_field[field]

                score = idf * tf * (self.k1 + 1) / (tf + self.k1)
                doc_to_score[doc] += score

                # split the term's score across fields in proportion to their weighted tf,
                # so the parts of one term add up to that term's score
                for field in weighted_tf_by_field:
                    score_breakdown[doc][term][field] = score * weighted_tf_by_field[field] / tf
                
        best_k = heapq.nlargest(k, doc_to_score.items(), key=lambda item: (item[1], -item[0]))   

        best_k_breakdown = []
        best_k_profiles = []
        for i in range(len(best_k)):
            doc, score = best_k[i]
            profile_id = self.index.read_doc(doc)["profile_id"]
            best_k_profiles.append((profile_id, score))
            breakdown = {}
            for term in score_breakdown[doc]:
                breakdown[term] = dict(score_breakdown[doc][term])
            best_k_breakdown.append((profile_id, breakdown))

        return best_k_profiles, best_k_breakdown

