def tag_latest_runs(api, project: str, model_name: str | None = None) -> dict:
    """Keep one latest tag per config.name, preserving logs and unrelated tags."""
    filters = {"config.name": model_name} if model_name else {}
    seen = set()
    latest = {}
    updated = 0
    for run in api.runs(project, filters=filters, order="-created_at"):
        name = run.config.get("name")
        if not isinstance(name, str) or not name:
            continue
        tag = "latest" if name not in seen else "historical"
        seen.add(name)
        if tag == "latest":
            latest[name] = run.id
        tags = [value for value in (run.tags or []) if value not in {"latest", "historical"}]
        tags.append(tag)
        if set(tags) != set(run.tags or []):
            run.tags = tags
            run.update()
            updated += 1
    return {"latest": latest, "updated": updated}
