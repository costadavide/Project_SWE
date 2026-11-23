"""Turn one string into the ordered list of (term, position) pairs that the index stores.

The same analyzer runs on documents (every value of every field, at index
time) and on queries (the free-text words, at search time), so the same
string always gives the same terms. A query can only match a document if
both were analyzed the same way. The one deliberate exception is described
under "Acronyms that collide with stopwords" below.

Contract
--------
analyze(text, stem=True, query=False) -> list of (term, position)

- text: one string: one job title, one skill, one bio, one query.
- stem: False turns off stemming, so a field such as skills can be indexed
  unstemmed if NDCG shows that works better. The index builder decides this
  per field; the analyzer only takes the flag.
- query: True when analyzing a user's query instead of a document (see
  acronyms below).
- Positions start at 0 for every string and count tokens in the original
  text, including stopwords that are later removed.
- The analyzer knows nothing about fields, profiles or the index. The gap of
  100 positions between the values of a field is added by the index builder.
- Deterministic: no randomness, no state between calls.
- Empty strings or text with no letters or digits return [].

Pipeline (in this order)
------------------------
1. Character normalization: Unicode NFKC, accents folded (José -> jose),
   curly quotes and the different dash characters turned into plain ASCII.
2. Tokenization, on the original-case text:
   a. Protected tokens are matched first and kept whole: tokens that
      splitting would break or make ambiguous, such as c++, c#, f#, .net,
      node.js, ci/cd, a/b. The list comes from the skills column (words
      inside skills containing + # . / &, used by at least 20 profiles).
   b. The rest is split on any character that is not a letter, a digit or
      a hyphen between two letters/digits.
   c. Every token gets the next position here, before anything is removed.
3. Hyphenated words become stacked tokens: the parts at consecutive
   positions, plus the joined form at the same position as the first part.
   "full-stack engineer" -> full 0, fullstack 0, stack 1, engin 2.
   So "full stack", "full-stack" and "fullstack" all match, and phrases still
   work. Only the parts count towards the field length; the joined form sits
   on a position that is already counted.
4. Lowercasing. Acronyms are recognised before this step (see below).
5. Stopword removal: a standard English list. Removed words keep their
   position, so "head of engineering" -> head 0, engin 2. The list holds
   unstemmed words, so this runs before stemming.
6. Stemming with Snowball English (snowballstemmer), unless stem=False.
   Protected tokens and acronyms are never stemmed: Snowball would turn
   aws into "aw" and node.js into "node.j".

Acronyms
--------
Most acronyms (AWS, SQL, ICU, SAS) are simply lowercased and protected from
stemming; they do not collide with English words.

A few acronyms are also stopwords: IT, OR (operating room), US, AS, and the
single-letter languages R and C if one-letter tokens would otherwise be
dropped. For these:
- in documents, the term is kept only when the original token is written in
  all capitals ("IT support" -> it). Any other casing ("it", "It") is the
  ordinary word and is removed as a stopword. "I love it" adds nothing, so
  a bio's pronoun never matches a search for IT people.
- in queries (query=True), these words are never removed, because users type
  in lowercase ("it support"). Since documents only kept them where they
  were acronyms, the query term matches exactly the IT people.
This is the one place where documents and queries are analyzed differently,
on purpose.

Examples (documents)
--------------------
"Senior Full-Stack Engineer (Node.js, C++, AWS)"
    senior 0, full 1, fullstack 1, stack 2, engin 3, node.js 4, c++ 5, aws 6
"Head of IT Operations"
    head 0, it 2, oper 3            ("of" removed, keeps position 1)
"Registered Nurse, ICU / Critical Care"
    regist 0, nurs 1, icu 2, critic 3, care 4
"I love it"
    love 1                          ("i" and "it" are stopwords here)
Query "head of it operations" -> head 0, it 2, oper 3 (same as the document).
"""
import snowballstemmer
import unicodedata
from pathlib import Path
import re

stemmer = snowballstemmer.stemmer("english")
stem_cache = {}

def build_set_from_txt(path):
    f = path.read_text(encoding="utf-8")
    lines = f.splitlines()

    res = set()
    for line in lines:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        res.add(line)
    return res

project_root = Path(__file__).resolve().parent
PROTECTED_TOKENS = build_set_from_txt(project_root / "resources" / "protected_tokens.txt")
ACRONYMS = build_set_from_txt(project_root / "resources" / "acronym_stopwords.txt")
STOPWORDS = build_set_from_txt(project_root / "resources" / "stopwords.txt")

table = {
        "’": "'",
        "‘": "'",
        "“": "\"",
        "”": "\"",
        "–": "-",
        "—": "-",
        "−": "-"
    }
TABLE = str.maketrans(table)

CHARS_TO_STRIP = "()[]{}\"',;:!?"

def normalize(text):
    if text == "":
        return text
    text = unicodedata.normalize("NFKD", text)
    norm = []
    for c in text:
        if unicodedata.combining(c) == 0:
            norm.append(c)
    norm = "".join(norm)

    norm = norm.translate(TABLE)    
    return norm
        

def check_chunk(chunk, chars_to_strip=CHARS_TO_STRIP, acronyms=ACRONYMS, protected_tokens=PROTECTED_TOKENS):
    chunk = chunk.strip(chars_to_strip)
    chunk = chunk.rstrip(".")
    if not chunk:
        return []
    
    if chunk in acronyms:
        return [(chunk.lower(), True)]
        
    if chunk.lower() in protected_tokens:
        return [(chunk.lower(), True)]

    split_on_bar = chunk.split("/")

    if len(split_on_bar) == 1:
        pieces = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", chunk.lower())
        res = []
        for piece in pieces:
            res.append((piece, piece in protected_tokens))
        return res
    
    res = []
    for text in split_on_bar:
        res.extend(check_chunk(text, chars_to_strip, acronyms, protected_tokens))
    return res


def split_text(text, chars_to_strip=CHARS_TO_STRIP, acronyms=ACRONYMS, protected_tokens=PROTECTED_TOKENS):
    splitted = text.split()
    res = []
    for split in splitted:
        res.extend(check_chunk(split, chars_to_strip, acronyms, protected_tokens))
    return res

def stem_word(word, stem=True):
    if stem:
        if word in stem_cache:
            return stem_cache[word]
        stem_cache[word] = stemmer.stemWord(word)
        return stem_cache[word]
    return word


def is_removed(word, query=False, stopwords=STOPWORDS, acronyms=ACRONYMS):
    # In queries, acronyms that are also stopwords ("it support") are never removed:
    # users type in lowercase, and documents only kept them when written in capitals.
    if query and word.upper() in acronyms:
        return False
    return word in stopwords


def combine_terms_positions(split, stem=True, query=False, stopwords=STOPWORDS, acronyms=ACRONYMS):
    term_pos_pairs = []
    i = 0
    for text, keep in split:
        if keep:
            term_pos_pairs.append((text, i))
            i += 1
            continue
        splitted = text.split("-")

        if len(splitted) == 1:
            if not is_removed(splitted[0], query, stopwords, acronyms):
                term_pos_pairs.append((stem_word(splitted[0], stem), i))
            i += 1
            continue

        joined = "".join(splitted)
        if not is_removed(joined, query, stopwords, acronyms):
            term_pos_pairs.append((stem_word(joined, stem), i))
        for split in splitted:
            if not is_removed(split, query, stopwords, acronyms):
                term_pos_pairs.append((stem_word(split, stem), i))
            i += 1

    return term_pos_pairs


def analyze(text, stem=True, query=False):
    """Turn one string into a list of (term, position) pairs; see the module docstring."""
    if not text:
        return []
    text = normalize(text)
    split = split_text(text, CHARS_TO_STRIP, ACRONYMS, PROTECTED_TOKENS)
    return combine_terms_positions(split, stem, query, STOPWORDS, ACRONYMS)

