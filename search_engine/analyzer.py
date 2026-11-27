"""Text analyzer: turns one string into a list of (term, position) pairs.

The same analyzer runs on documents and on queries, so the same text always
gives the same terms. Steps:
1. normalize: strip accents, turn curly quotes and dashes into ASCII;
2. tokenize: protected tokens (c++, node.js, ci/cd, from resources/) and
   acronyms stay whole, "/" splits, everything else is split on
   non-alphanumeric characters (hyphens inside a word are kept);
3. hyphenated words are stacked: "full-stack" -> full, fullstack (same
   position), stack, so "full stack", "full-stack" and "fullstack" all match;
4. lowercase and drop stopwords, keeping their positions so phrases still work;
5. Snowball stemming (protected tokens and acronyms are not stemmed).

Acronyms that are also stopwords (IT, OR, US, ...) are kept in documents only
when written in capitals, and always kept in queries (users type "it support").

Example: "Head of IT Operations" -> [("head", 0), ("it", 2), ("oper", 3)]
"""
import snowballstemmer
import unicodedata
from pathlib import Path
import re

stemmer = snowballstemmer.stemmer("english")
stem_cache = {}

def build_set_from_txt(path):
    """Read a word list (one entry per line, # for comments) into a set."""
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
    """Strip accents and turn curly quotes and dashes into plain ASCII."""
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
    """Split one whitespace-separated chunk into (token, protected) pairs."""
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
    """Split a whole string into (token, protected) pairs, chunk by chunk."""
    splitted = text.split()
    res = []
    for split in splitted:
        res.extend(check_chunk(split, chars_to_strip, acronyms, protected_tokens))
    return res

def stem_word(word, stem=True):
    """Snowball stem of word (cached); the word itself if stem is False."""
    if stem:
        if word in stem_cache:
            return stem_cache[word]
        stem_cache[word] = stemmer.stemWord(word)
        return stem_cache[word]
    return word


def is_removed(word, query=False, stopwords=STOPWORDS, acronyms=ACRONYMS):
    """True if word is a stopword; in queries, stopword-acronyms (it, or, us) are kept."""
    if query and word.upper() in acronyms:
        return False
    return word in stopwords


def combine_terms_positions(split, stem=True, query=False, stopwords=STOPWORDS, acronyms=ACRONYMS):
    """Give each token its position, drop stopwords, stack hyphenated words and stem."""
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
    """Turn one string into [(term, position)]. stem=False skips stemming;
    query=True keeps stopword-acronyms.
    """
    if not text:
        return []
    text = normalize(text)
    split = split_text(text, CHARS_TO_STRIP, ACRONYMS, PROTECTED_TOKENS)
    return combine_terms_positions(split, stem, query, STOPWORDS, ACRONYMS)

