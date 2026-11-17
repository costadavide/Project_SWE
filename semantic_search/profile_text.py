import json 

def text_generation(row):
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
    # seed_role
    add(core, "Occupation Category", row.get("seed_role"))
    # job_family
    add(core, "Professional field", clean(row.get("job_family")).replace("_", " "))
    # skills
    skills = row.get("skills") or user.get("skills") or []
    if skills:
        add(core, "Skills", ", ".join(skills))
    # bio
    add(core, "Professional Summary", user.get("bio"))
    # experiences (title, description)
    experiences = []

    sorted_experiences = sorted(
    profile.get("experience") or [],
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
    for edu in profile.get("education") or []:
        degree = clean(edu.get("degree"))
        study_field = clean(edu.get("field_of_study"))

        if degree or study_field:
            education.append(f"{degree} in {study_field}") 
        elif degree or study_field:
            education.append(degree or study_field)
    add_list_section(core, "Education", education)
    # certifications 
    certifications = [ clean(cert.get("name")) for cert in profile.get("certifications") or []]
    add_list_section(core, "Certifications", certifications)

    # 2. projects
    # name
    # description
    projects = []

    for proj in profile.get("projects") or []:
        name = clean(proj.get("name"))
        description = clean(proj.get("description"))

        if name and description:
            projects.append(f"{name}: {description}")
        elif name or description:
            projects.append(name or description)

    projects_text = []
    add_list_section(projects_text, "Projects", projects)

    #3. career interests
    interests = user.get("career_interests") or []
    career_interests = []
    add_list_section(career_interests, "Desired Career Roles", interests)


    return {
    "core_text": "\n".join(core),
    "projects_text": "\n".join(projects_text),
    "interests_text": "\n".join(career_interests)
    }

