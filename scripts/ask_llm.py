"""
Ask a question to the local LLM served by LM Studio.

Run from the project root, with the LM Studio server on:

    python -m scripts.ask_llm "What is cosine similarity?"   # one question
    python -m scripts.ask_llm        # keeps asking; type "exit" to stop

Each question is independent: the model does not remember earlier ones.
"""

import copy
import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-1.7b"  # slower but more accurate: "qwen3.8-27b"
TIMEOUT_SECONDS = 600

SYSTEM_PROMPT = """
You are a query understanding component for a semantic people search engine.

Your task is to transform natural-language requests into structured search representations
that will be used for candidate retrieval, filtering, and ranking.

You must produce a JSON object containing:
1. A rewritten semantic search query.
2. Explicit hard constraints.
3. Soft preferences.

### 1. SEMANTIC QUERY REWRITING

Rewrite the user's request into a concise, information-rich description of the ideal candidate.

Rules:
- Preserve relevant professional information, including roles, skills, experience, industries, and expertise.
- Use standardized professional terminology when appropriate.
- Remove conversational language, filler words, and irrelevant details.
- Preserve the original meaning and intent.
- Do not invent qualifications, skills, or requirements.
- Do not add synonyms or related skills unless clearly implied by the request.
- Keep relevant professional attributes in the semantic query even when they also appear in structured constraints.
- Leave exclusions out of the semantic query, such as companies, skills or places the user does not want. You must put them only in hard_constraints.
- Write the rewritten query in English, regardless of the input language.

### 2. HARD CONSTRAINT EXTRACTION

Extract requirements that explicitly restrict which candidates are eligible.

Supported constraints:
- location: Required geographic location.
- excluded_locations: Places explicitly excluded.
- current_role: Required current job title or position.
- past_role: Required previous job title or position.
- min_years_experience: Minimum years of professional experience.
- max_years_experience: Maximum years of professional experience.
- required_skills: Skills explicitly required.
- excluded_skills: Skills explicitly prohibited.
- industry: Required industry.
- current_company: Required current employer.
- excluded_companies: Employers explicitly excluded.

Rules:
- Extract only constraints explicitly stated or unambiguously implied.
- Never infer constraints from stereotypes or assumptions.
- Distinguish current experience from past experience.
- Do not confuse years of experience with years since graduation.
- Do not treat preferences as mandatory requirements.
- Preserve the meaning of negations.
- Use null for unspecified scalar constraints and empty arrays for unspecified list constraints.
- Never invent missing values.
- Do not assume a requested role must be the candidate's current role unless explicitly specified.
- requested_role is the job the user is asking for. Fill current_role only when the request says "currently", "now" or similar.
- "former", "ex-", "used to be" and "previously" mean past_role. If the request also says what the person does now, fill both past_role and current_role.
- "at least N" and "N+" set only min_years_experience. "up to N" and "no more than N" set only max_years_experience. "between N and M" sets both. Write numbers without quotes.
- "not from X" means excluded_companies when X is a company. "without X" or "but not X" means excluded_skills when X is a skill.
- "who speaks X", "fluent in X" and "must know X" mean required_skills, unless a soft preference word introduces them.
- "not in X", "not currently in X" and "outside X" mean excluded_locations when X is a place. Never put an excluded place in location.
- Fill industry only when the request names an industry.
- Write all values in English, regardless of the input language.

### 3. SOFT PREFERENCE EXTRACTION

Identify desirable characteristics that should influence ranking without excluding candidates.

Rules:
- Words such as "ideally", "preferably", "bonus if", "would be nice", "nice to have", "a plus" and "optional" mark a soft preference. Never put it in required_skills.
- If the request states no preference, leave soft_preferences out.
- Write only the preference itself, without the introducing words.
- Preserve soft preferences as concise English strings.
- Do not promote soft preferences to hard constraints.
- Do not duplicate hard constraints as soft preferences.

### 4. AMBIGUITY HANDLING

When a requirement is ambiguous, preserve its meaning without inventing a precise interpretation.

Examples:
- "Senior Software engineer" keeps the word "Senior" in requested_role and semantic_query, but does not set min_years_experience.
- "Experienced in Python" does not imply a specific number of years.
- "Near Milan" should not be converted into an arbitrary distance radius.
- "Backend engineer" does not necessarily imply a current job title.

### 5. OUTPUT FORMAT

Return ONLY valid JSON matching the following structure:

{
  "requested_role": "string",
  "hard_constraints": {
    "location": null,
    "excluded_locations": [],
    "current_role": null,
    "past_role": null,
    "min_years_experience": null,
    "max_years_experience": null,
    "required_skills": [],
    "excluded_skills": [],
    "industry": null,
    "current_company": null,
    "excluded_companies": []
  },
  "semantic_query": "string",
  "soft_preferences": []
}

Output requirements:
- Return valid JSON only.
- Do not include Markdown code fences.
- Do not include explanations or comments.
- Do not include additional fields.
- Preserve the exact JSON field names and data types.

Additional output rules (these override the rules above):
- Omit every field whose value would be null or an empty array. Always include requested_role (the role asked for) and semantic_query. Omit hard_constraints if it would be empty.

### 6. EXAMPLES

These examples only show the format. Never copy their values into other answers.

Request: Former journalist who now works as a press officer in Vienna
Answer: {"requested_role": "press officer", "semantic_query": "Press officer in Vienna with previous experience as a journalist", "hard_constraints": {"location": "Vienna", "current_role": "press officer", "past_role": "journalist"}}

Request: QA tester, not from Nokia, no more than 9 years of experience
Answer: {"requested_role": "QA tester", "semantic_query": "QA tester with up to 9 years of experience", "hard_constraints": {"max_years_experience": 9, "excluded_companies": ["Nokia"]}}

Request: Receptionist fluent in Korean, not in Oslo, knowing Excel would be a plus
Answer: {"requested_role": "receptionist", "semantic_query": "Receptionist fluent in Korean", "hard_constraints": {"required_skills": ["Korean"], "excluded_locations": ["Oslo"]}, "soft_preferences": ["Excel"]}

Request: Gardener who must know irrigation but not pesticides, at least 2 years of experience
Answer: {"requested_role": "gardener", "semantic_query": "Gardener with irrigation skills and at least 2 years of experience", "hard_constraints": {"min_years_experience": 2, "required_skills": ["irrigation"], "excluded_skills": ["pesticides"]}}

"""

# The model may leave out empty fields (shorter answers are faster);
# fill_missing_fields() puts them back.
EMPTY_CONSTRAINTS = {
    "location": None,
    "excluded_locations": [],
    "current_role": None,
    "past_role": None,
    "min_years_experience": None,
    "max_years_experience": None,
    "required_skills": [],
    "excluded_skills": [],
    "industry": None,
    "current_company": None,
    "excluded_companies": [],
}


def call_model(question, **options):
    """Send one question to LM Studio and return the model's raw text."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "reasoning_effort": "none",
        "temperature": 0,  # same question, (almost) same answer
        **options,
    }
    request = Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        result = json.load(response)

    return result["choices"][0]["message"]["content"].strip()


# Words that start an exclusion, and filler words that may come just before one.
NEGATIONS = r"(?:not|never|no|without|except|excluding|outside)"
LEAD_INS = r"(?:but|and|with|who|that|is|are|was|has|have|had|does|do|did)"


def remove_exclusions(text, constraints):
    """
    Remove phrases such as "not from Oracle" from the semantic query.

    Embeddings handle negation poorly: naming an excluded company, skill or
    place would pull those people closer. Only names the model extracted as
    exclusions are removed, and only when a negation comes shortly before them.
    """
    names = []
    for field in ("excluded_companies", "excluded_skills", "excluded_locations"):
        value = constraints[field]
        names += [value] if isinstance(value, str) else value

    names = [re.escape(name) for name in names if isinstance(name, str) and name.strip()]
    if not names:
        return text

    name = rf"(?:{'|'.join(sorted(names, key=len, reverse=True))})(?!\w)"
    phrase = (
        rf"[,;]?\s*(?:\b{LEAD_INS}\s+)*"
        rf"\b{NEGATIONS}\b"
        rf"(?:\s+(?!{NEGATIONS}\b)[\w'/-]+){{0,4}}?"  # e.g. "worked for"
        rf"\s+{name}(?:(?:\s*,\s*|\s+)(?:(?:or|and|nor)\s+)?{name})*"
        rf"(?:\s+(?:experience|skills?|knowledge|background)\b)?"
    )
    cleaned = re.sub(phrase, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")

    return cleaned or text


def fill_missing_fields(answer):
    constraints = copy.deepcopy(EMPTY_CONSTRAINTS)

    for name, value in (answer.get("hard_constraints") or {}).items():
        if value is not None:
            constraints[name] = value

    return {
        "requested_role": answer.get("requested_role"),
        "semantic_query": remove_exclusions(answer["semantic_query"], constraints),
        "hard_constraints": constraints,
        "soft_preferences": answer.get("soft_preferences") or [],
    }


def ask(question):
    """Return the structured search request for a question, as a dict."""
    text = call_model(question)

    try:
        return fill_missing_fields(json.loads(text))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"The model did not return the expected JSON:\n{text}") from error


def warm_up():
    """Load the model and cache the system prompt, so the first question is fast."""
    try:
        call_model("Hello", max_tokens=1)
    except OSError:
        pass  # the first question will report the problem


def print_answer(question):
    """Print the model's answer, or the error. Returns True on success."""
    try:
        print(json.dumps(ask(question), indent=2, ensure_ascii=False))
        return True
    except HTTPError as error:
        print(f"LM Studio returned an error: {error.read().decode()}")
    except URLError as error:
        print(f"Could not reach LM Studio at {URL}: {error.reason}")
    except TimeoutError:
        print(f"No answer from LM Studio after {TIMEOUT_SECONDS} seconds.")
    except ValueError as error:
        print(error)
    return False


def main():
    if len(sys.argv) > 1:
        sys.exit(0 if print_answer(" ".join(sys.argv[1:])) else 1)

    print(f"Warming up {MODEL}...")
    warm_up()

    while True:
        try:
            question = input("\nQuestion (or 'exit' to stop): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() == "exit":
            break

        if question:
            print_answer(question)


if __name__ == "__main__":
    main()
