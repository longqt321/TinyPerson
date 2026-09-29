import argparse

import torch

from tinydet.models.registry import build_model
from tinydet.utils.config import load_config
from tinydet.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU config and model forward check")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.seed, config.train["deterministic"])
    model = build_model({**config.model, "pretrained": None})
    size = 64
    with torch.no_grad():
        result = model.predict(torch.zeros(1, 3, size, size), device="cpu", verbose=False)
    assert len(result) == 1 and result[0].orig_shape == (size, size)
    print(f"CPU sanity passed: {config.name}")


if __name__ == "__main__":
    main()
