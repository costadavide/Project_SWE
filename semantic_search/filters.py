import json

STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "district of columbia": "DC", "florida": "FL",
    "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def load_metadata(path):
    """Return one {city, state, experience_years} per profile, in file order.
    Position i in the list is row i of the embeddings."""
    metadata = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            metadata.append({
                "city": row["city"],
                "state": row["state"],
                "experience_years": row["experience_years"],
            })
    return metadata


def filter_by_years(metadata, minimum=None, maximum=None):

    if minimum is None and maximum is None:
        return None 

    allowed = set()

    for row, person in enumerate(metadata):
        years = person["experience_years"]
        if minimum is not None and years < minimum:
            continue 
        if maximum is not None and years > maximum:
            continue 
        allowed.add(row)

    return allowed


def find_state(text=None):
    if text is None:
        return None 

    state = text.strip().lower()

    if state in STATES:
        return STATES[state]

    code = state.upper()
    if code in STATES.values():
        return code 

    return None


def find_place(text):
    """Return (city, state) for a place such as "Elizabeth, New Jersey".
    Either part is None when it is not given."""
    if text is None:
        return None, None

    # only a state: "Kansas"
    state = find_state(text)
    if state is not None:
        return None, state

    # city and state: "Elizabeth, New Jersey"
    if "," in text:
        city, state_text = text.rsplit(",", 1)
        return city.strip().lower(), find_state(state_text)

    # only a city: "San Francisco"
    return text.strip().lower(), None


def filter_by_place(metadata, city=None, state=None):

    if city is None and state is None:
        return None 

    allowed = set()

    for row, person in enumerate(metadata):
        if state is not None and person["state"] != state:
            continue 
        if city is not None and person["city"].lower() != city:
            continue 

        allowed.add(row)

    return allowed 


def build_allowed(metadata, constraints):
    min_years = max_years = None
    if constraints["min_years_experience"] is not None:
        min_years = int(constraints["min_years_experience"])
    if constraints["max_years_experience"] is not None:
        max_years = int(constraints["max_years_experience"])

    allowed_years = filter_by_years(metadata, min_years, max_years)

    city, state = find_place(constraints["location"])
    allowed_place = filter_by_place(metadata, city, state)

    if allowed_years is None:
        return allowed_place
    if allowed_place is None:
        return allowed_years
    return allowed_years & allowed_place
