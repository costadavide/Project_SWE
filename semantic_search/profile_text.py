import json 

def text_generation(row, include_seed_role=True, include_bio=True):
    """
    Generate the text that will be then used to generate the embedding for the user
    Generates 3 texts: 
        1. main text
        2. projects 
        3. career interests
    """
    profile = row["profile"]

    if isinstance(profile, str):
        profile = json.loads(profile)

    user = profile.get("user") or {}

    def clean(value):
        return str(value).strip() if value is not None else ""

    def add(parts, label, value):
        value = clean(value)
        if value:
            parts.append(f"{label}: {value}")

    def add_list_section(parts, title, items):
        items = [item.strip() for item in items if item.strip()]

        if items: 
            parts.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))

    # 1. main text 
    core = []
    if include_seed_role:
        add(core, "Occupation Category", row.get("seed_role"))

    # Professional field
    add(
        core,
        "Professional field",
        clean(row.get("job_family")).replace("_", " ")
    )
    # skills
    skills = row.get("skills")

    if skills is None:
        skills = user.get("skills") or []

    if len(skills) > 0:
        add(core, "Skills", ", ".join(skills))

     # Professional summary
    if include_bio:
        add(core, "Professional Summary", user.get("bio"))
    

    # experiences (title, description)
    experiences = []

    experience_data = profile.get("experience")
    if experience_data is None:
        experience_data = []
    sorted_experiences = sorted(
    experience_data,
    key=lambda exp: exp.get("start_date") or "",
    reverse=True) # sort experiences from currest/latest to first one

    for exp in sorted_experiences:
        title = clean(exp.get("title"))
        description = clean(exp.get("description"))

        if title or description:
            experiences.append(f"{title}: {description}")
    add_list_section(core, "Work experience", experiences)

    # education
    education = []
    edu_data = profile.get("education")
    if edu_data is None:
        edu_data = []

    for edu in edu_data:
        degree = clean(edu.get("degree"))
        study_field = clean(edu.get("field_of_study"))

        if degree and study_field:
            education.append(f"{degree} in {study_field}") 
        elif degree or study_field:
            education.append(degree or study_field)
    add_list_section(core, "Education", education)
    # certifications 
    cert_data = profile.get("certifications")
    if cert_data is None:
        cert_data = []
    certifications = [ clean(cert.get("name")) for cert in cert_data]
    add_list_section(core, "Certifications", certifications)

    # 2. projects
    # name
    # description
    projects = []
    projects_data = profile.get("projects")
    if projects_data is None:
        projects_data = []

    for proj in projects_data:
        name = clean(proj.get("name"))
        description = clean(proj.get("description"))

        if name and description:
            projects.append(f"{name}: {description}")
        elif name or description:
            projects.append(name or description)

    projects_text = []
    add_list_section(projects_text, "Projects", projects)

    #3. career interests
    interests = user.get("career_interests")
    if interests is None:
        interests = []
    career_interests = []
    add_list_section(career_interests, "Desired Career Roles", interests)


    return {
    "core_text": "\n".join(core),
    "projects_text": "\n".join(projects_text),
    "interests_text": "\n".join(career_interests)
    }

