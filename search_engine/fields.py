"""Turn a profile JSON into the six text fields that the index stores.

profile_to_dict(profile) -> {field: [strings]}, one string per value:
    current_title    most recent job title
    past_titles      the other job titles, newest first
    skills           user.skills
    summary          bio, job descriptions and project names
    education        "degree in field_of_study"
    certifications   certification names
Never indexed: protected attributes, companies, locations, dates, cohort and
seed_role. Filters come from the database columns, not from here.
"""


def _clean_list(values):
    """Strip values, drop empty and non-string ones, remove case-insensitive duplicates."""
    seen = set()
    result = []
    for value in values or []:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            result.append(value)
    return result


def _month_key(date):
    """'MM/YYYY' -> (year, month) for sorting; None if missing or malformed."""
    try:
        month, year = date.split("/")
        return int(year), int(month)
    except (AttributeError, ValueError):
        return None


def _sorted_jobs(experiences):
    """Jobs newest first (ongoing before finished, then latest start date); undated jobs last."""
    jobs = [
        job for job in experiences or []
        if isinstance(job, dict) and (job.get("title") or "").strip()
    ]
    dated = [job for job in jobs if _month_key(job.get("start_date"))]
    undated = [job for job in jobs if not _month_key(job.get("start_date"))]
    dated.sort(
        key=lambda job: (job.get("end_date") is None, _month_key(job["start_date"])),
        reverse=True,
    )
    return dated + undated


def _job_title(job):
    """The job title, marked "(internship)" when it is one."""
    title = job["title"].strip()
    return f"{title} (internship)" if job.get("is_internship") else title


def _education_entry(entry):
    """"Degree in field", or whichever of the two is given."""
    degree = entry.get("degree").strip() if entry.get("degree") else None
    field = entry.get("field_of_study").strip() if entry.get("field_of_study") else None
    if degree and field and field.lower() not in degree.lower():
        return f"{degree} in {field}"
    return degree or field


def profile_to_dict(profile):
    """The profile as {field: [non-empty strings]}; a missing field is []."""
    user = profile.get("user") or {}
    res = {"current_title": [], "past_titles":[], "skills":[], "summary":[], "education":[], "certifications":[]}

    jobs = _sorted_jobs(profile.get("experience"))
    if jobs:
        latest_title = _job_title(jobs[0])
        res["current_title"] = [latest_title]

        previous = _clean_list(
            _job_title(job) for job in jobs[1:]
            if _job_title(job).lower() != latest_title.lower()
        )
        if previous:
            res["past_titles"] = previous

    if user.get("bio"):
        if user.get("bio").strip():
            res["summary"].append(user.get("bio").strip())

    if jobs:
        description = _clean_list([job.get("description").strip() if job.get("description") else None for job in jobs])
        res["summary"].extend(description)
            
    skills = _clean_list(user.get("skills"))
    if skills:
        res["skills"].extend(skills)

    certifications = _clean_list(
        cert.get("name") for cert in profile.get("certifications") or []
        if isinstance(cert, dict)
    )
    if certifications:
        res["certifications"].extend(certifications)

    education = _clean_list([_education_entry(entry) for entry in profile.get("education") or []
        if isinstance(entry, dict)])

    if education:
        res["education"].extend(education)

    projects = _clean_list(
        project.get("name") for project in profile.get("projects") or []
        if isinstance(project, dict)
    )
    if projects:
        res["summary"].extend(projects)

    return res
