from pathlib import Path

import yaml


def load_comparison(path: str | Path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError("Comparison config must be a YAML mapping")
    shared = {key: raw[key] for key in ("seed", "data", "train", "hardware", "output")}
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    names = set()
    experiments = []
    for model in models:
        if (not isinstance(model, dict) or not {"name", "config"} <= model.keys()
                or set(model) - {"name", "config"}):
            raise ValueError("Each model needs only name and config")
        name = model["name"]
        if not isinstance(name, str) or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Model name must contain only letters, digits, '-' or '_': {name}")
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name)
        if not Path(model["config"]).is_file():
            raise ValueError(f"Model config does not exist: {model['config']}")
        model_config = {"config": model["config"], "pretrained": None}
        experiments.append({"name": name, **shared, "model": model_config})
    return experiments
