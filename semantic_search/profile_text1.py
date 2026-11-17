"""Turn a structured profile into the text that the embedding model represents.

Qualifications come first, then the user's city and state near the end.
Sensitive or filter-style fields (age, ethnicity, legal status, sponsorship,
desired salary), company names, job locations and work location preferences
are left out on purpose: protected attributes should never influence the
ranking, and company names add noise.
Location in the embedding only nudges the ranking; strict location
requirements still need a database filter.

experience_to_text() builds a second text from the job history alone; see
semantic_search/scoring.py for how the two similarities are combined.
"""

MAX_PREVIOUS_ROLES = 3


def _clean_list(values):
    """Strip, drop empties and non-strings, and remove case-insensitive duplicates."""
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
    """Convert 'MM/YYYY' to (year, month) for sorting; None if missing or malformed."""
    try:
        month, year = date.split("/")
        return int(year), int(month)
    except (AttributeError, ValueError):
        return None


def _sorted_jobs(experiences):
    """Return jobs newest first; undated jobs keep their original order at the end.

    Ongoing jobs (no end date) come before finished ones, then later start
    dates first. The source data is not reliably ordered: in a 100-profile sample,
    14 of 100 profiles listed an older job first.
    """
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
    title = job["title"].strip()
    return f"{title} (internship)" if job.get("is_internship") else title


def _education_entry(entry):
    degree = (entry.get("degree") or "").strip()
    field = (entry.get("field_of_study") or "").strip()
    if degree and field and field.lower() not in degree.lower():
        return f"{degree} in {field}"
    return degree or field


def profile_to_text(profile):
    user = profile.get("user") or {}
    lines = []

    jobs = _sorted_jobs(profile.get("experience"))
    if jobs:
        latest_title = _job_title(jobs[0])
        lines.append(f"Most recent role: {latest_title}")

        previous = _clean_list(
            _job_title(job) for job in jobs[1:]
            if _job_title(job).lower() != latest_title.lower()
        )
        if previous:
            lines.append(f"Previous roles: {', '.join(previous[:MAX_PREVIOUS_ROLES])}")

    bio = (user.get("bio") or "").strip()
    if bio:
        lines.append(f"Summary: {bio}")

    if jobs:
        description = (jobs[0].get("description") or "").strip()
        if description:
            lines.append(f"Most recent role description: {description}")

    experience_years = user.get("experience_years")
    if experience_years is not None:
        lines.append(f"Years of experience: {experience_years}")

    skills = _clean_list(user.get("skills"))
    if skills:
        lines.append(f"Skills: {', '.join(skills)}")

    certifications = _clean_list(
        cert.get("name") for cert in profile.get("certifications") or []
        if isinstance(cert, dict)
    )
    if certifications:
        lines.append(f"Certifications: {', '.join(certifications)}")

    education = _clean_list(
        _education_entry(entry) for entry in profile.get("education") or []
        if isinstance(entry, dict)
    )
    if education:
        lines.append(f"Education: {', '.join(education)}")

    projects = _clean_list(
        project.get("name") for project in profile.get("projects") or []
        if isinstance(project, dict)
    )
    if projects:
        lines.append(f"Projects: {', '.join(projects)}")

    languages = _clean_list(user.get("languages"))
    if languages:
        lines.append(f"Languages: {', '.join(languages)}")

    # The user's own city and state; job locations and work preferences are left out.
    location = ", ".join(
        part for part in ((user.get("city") or "").strip(), (user.get("state") or "").strip()) if part
    )
    if location:
        lines.append(f"Location: {location}")

    # career_interests are left out: they say what the person wants to do,
    # not what they have done, and pushed career changers above people
    # already doing the job.

    return "\n".join(lines)


def experience_to_text(profile):
    """Only the jobs the person has held: titles and descriptions, newest first.

    Embedded separately from profile_to_text() so that ranking can reward
    actual work experience, not just matching skills or a matching summary.
    Skills, bio, education and interests are left out on purpose.
    """
    lines = []
    for position, job in enumerate(_sorted_jobs(profile.get("experience"))[: MAX_PREVIOUS_ROLES + 1]):
        label = "Most recent role" if position == 0 else "Previous role"
        lines.append(f"{label}: {_job_title(job)}")
        description = (job.get("description") or "").strip()
        if description:
            lines.append(f"Description: {description}")
    return "\n".join(lines)
