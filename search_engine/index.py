from collections import defaultdict
from search_engine.fields import profile_to_dict
from search_engine.analyzer import analyze
import pickle
from pathlib import Path
import math

# defaultdict needs a function that creates the missing value, not a value.
# These named functions build one nesting level each:
#   postings[term][doc_id][field] -> list of positions
#   field_lengths[doc_id][field]  -> int
# They are named (not lambdas) so that pickle can save the index.
def new_positions_by_field():
    return defaultdict(list)


def new_fields_by_doc():
    return defaultdict(new_positions_by_field)


def new_length_by_field():
    return defaultdict(int)


# Per-field stemming. The query side must use the same setting for each field,
# so it lives here in one place. Set a field to False to index it unstemmed.
FIELD_STEM = {
    "current_title": True,
    "past_titles": True,
    "skills": True,
    "summary": True,
    "education": True,
    "certifications": True,
}

class Index:
    def __init__(self, docs):
        self.postings = defaultdict(new_fields_by_doc)
        self.doc_store = []
        self.field_lengths = defaultdict(new_length_by_field)
        self.n_docs = 0
        self.n_docs_per_field = defaultdict(int)
        self.sum_length_field = defaultdict(int)

        self.build_index(docs)

    def build_index(self, docs):
        if not docs:
            return None

        for doc in docs:
            self.add_doc(doc)

    def add_doc(self, document):
        """
        Assume the document containes a full row of the database as a dict
        """
        profile_id = document.get("profile_id", None)

        if not document.get("profile", None):
            return None
        
        prof = {"profile_id": profile_id}
        keys = ["job_family", "seniority", "city", "state", "experience_years"]
        for key in keys:
            if document.get(key, None) is not None:
                prof[key] = document[key]

        self.doc_store.append(prof)
        self.n_docs += 1
        doc_id = len(self.doc_store) - 1

        profile = profile_to_dict(document["profile"])
        for field in profile:
            if not profile[field]:
                continue
            self.add_field(field, profile[field], doc_id)

    def add_field(self, field, values, doc_id):
        offset = 0
        cnt = 0
        for text in values:
            term_pos = analyze(text, query=False, stem=FIELD_STEM.get(field, True))
            last_pos = -1
            for term, pos in term_pos:
                self.postings[term][doc_id][field].append(pos + offset)
                if last_pos != pos:
                    last_pos = pos
                    cnt += 1
            if term_pos:
                offset += term_pos[-1][1] + 1 + 100
        self.field_lengths[doc_id][field] = cnt
        self.sum_length_field[field] += cnt
        if cnt:
            self.n_docs_per_field[field] += 1
    
    def read_term(self, term):
        if term not in self.postings:
            return {}
        return self.postings[term]

    def __len__(self):
        return self.n_docs

    def read_field_length(self, doc_id, field):
        if not (0 <= doc_id < self.n_docs):
            raise KeyError("Document ID not in the index")

        if doc_id not in self.field_lengths:
            return 0
        
        if field not in self.field_lengths[doc_id]:
            return 0

        return self.field_lengths[doc_id][field]

    def avg_field_length(self, field):
        if field not in self.sum_length_field:
            return 0.0
        if self.n_docs_per_field[field] != 0:
            return self.sum_length_field[field] / self.n_docs_per_field[field]
        else:
            return 0.0

    def doc_freq(self, term):
        return len(self.read_term(term))

    def read_doc(self, doc_id):
        if not (0 <= doc_id < self.n_docs):
            raise KeyError("Document ID not in the index")

        return self.doc_store[doc_id]

    def term_freq(self, term, doc_id):
        if not (0 <= doc_id < self.n_docs):
            raise KeyError("Document ID not in the index")
        
        postings = self.read_term(term)
        if not postings:
            return 0

        if doc_id not in postings:
            return 0
        
        tf = 0
        for field in postings[doc_id]:
            tf += len(postings[doc_id][field])

        return tf

    def doc_length(self, doc_id):
        if not (0 <= doc_id < self.n_docs):
            raise KeyError("Document ID not in the index")
        
        if doc_id not in self.field_lengths:
            return 0
        
        dl = 0
        for fl in self.field_lengths[doc_id].values():
            dl += fl

        return dl

    def avg_doc_length(self):
        avg = 0
        for field in self.sum_length_field.values():
            avg += field

        return avg / self.n_docs

    def idf(self, term):
        df = self.doc_freq(term)
        
        return math.log(1 + (self.n_docs - df  + 0.5) / (df + 0.5))

    def save(self, path):
        """Save the whole index to a pickle file (creates the folder if needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.field_stem = dict(FIELD_STEM)
        with path.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        """Load an index saved with save()."""
        with Path(path).open("rb") as file:
            index = pickle.load(file)
        if not isinstance(index, Index):
            raise TypeError(f"{path} does not contain an Index")
        if getattr(index, "field_stem", None) != FIELD_STEM:
            print(f"Warning: {path} was built with different stem settings; rebuild the index.")
        return index
            