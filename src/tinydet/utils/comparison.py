from pathlib import Path

import yaml


def load_comparison(path: str | Path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError("Comparison config must be a YAML mapping")
    if ("seed" in raw) == ("seeds" in raw):
        raise ValueError("Comparison config needs exactly one of seed or seeds")
    seeds = raw["seeds"] if "seeds" in raw else [raw["seed"]]
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a nonempty list of unique nonnegative integers")
    shared = {key: raw[key] for key in ("data", "train", "hardware", "output")}
    if "diagnostics" in raw:
        shared["diagnostics"] = raw["diagnostics"]
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    names = set()
    experiments = []
    for model in models:
        if (not isinstance(model, dict) or "name" not in model
                or set(model) - {"name", "config", "pretrained", "run"}):
            raise ValueError("Each model needs name and either config or run")
        name = model["name"]
        if not isinstance(name, str) or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Model name must contain only letters, digits, '-' or '_': {name}")
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name)
        if "run" in model:
            if set(model) != {"name", "run"}:
                raise ValueError(f"Existing run cannot also be trained: {name}")
            run = model["run"]
            parts = Path(run).parts if isinstance(run, str) else ()
            if len(parts) != 2 or any(part in {".", ".."} for part in parts):
                raise ValueError(f"run must be experiment/run-id: {run!r}")
            experiments.append({"name": name, "run": run})
            continue
        if "config" not in model:
            raise ValueError(f"Model needs config or run: {name}")
        if not Path(model["config"]).is_file():
            raise ValueError(f"Model config does not exist: {model['config']}")
        pretrained = model.get("pretrained")
        if pretrained is not None and not isinstance(pretrained, str):
            raise ValueError(f"pretrained must be a path string: {pretrained}")
        model_config = {"config": model["config"], "pretrained": pretrained}
        for seed in seeds:
            experiments.append({
                "name": name,
                "seed": seed,
                **shared,
                "model": model_config.copy(),
            })
    comparisons = raw.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise TypeError("comparisons must be a list")
    comparison_names = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise TypeError("Each comparison must be a mapping")
        name = comparison.get("name")
        control = comparison.get("control")
        treatment = comparison.get("treatment")
        if not all(isinstance(value, str) and value for value in (name, control, treatment)):
            raise ValueError("Each comparison needs name, control and treatment")
        if name in comparison_names:
            raise ValueError(f"Duplicate comparison name: {name}")
        comparison_names.add(name)
        unknown = {control, treatment} - names
        if unknown:
            raise ValueError(f"Unknown comparison arm(s): {', '.join(sorted(unknown))}")
    return experiments
