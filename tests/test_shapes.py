import torch

from tinydet.models.registry import build_model


def test_baseline_forward():
    model = build_model({"config": "configs/model/yolo26n.yaml", "pretrained": None})
    result = model.predict(torch.zeros(1, 3, 64, 64), device="cpu", verbose=False)
    assert result[0].orig_shape == (64, 64)
