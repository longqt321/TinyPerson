"""Tests for the pipeline-native DetectGaussSplit integration.

These are the checks that decide whether the experiment is valid at all:

* the head must not change the detector's output (identity property),
* the baseline checkpoint must still load (no shifted state-dict keys),
* the auxiliary loss must produce gradients on the head and only the head,
* the inference path must emit second boxes only above the threshold.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tinydet.models.registry import build_model
from tinydet.modules.gauss_split_detect import (
    DetectGaussSplit,
    find_gauss_split_head,
)
from tinydet.modules.pair_split import PairSplit

BASE_YAML = ROOT / "configs/model/yolo26n_tiny_A2_p2_tinyperson.yaml"
GAUSS_YAML = ROOT / "configs/model/yolo26n_tiny_A2_p2_gauss_split.yaml"


def _build(yaml_path: Path):
    return build_model({"config": str(yaml_path)}).model


def test_head_does_not_change_detector_output() -> None:
    """Same weights in, same boxes out: the head must be a pure add-on."""
    base = _build(BASE_YAML)
    gauss = _build(GAUSS_YAML)
    missing, unexpected = gauss.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected[:5]
    assert all(".head." in k for k in missing), missing[:5]

    base.eval()
    gauss.eval()
    x = torch.randn(1, 3, 256, 320)
    with torch.no_grad():
        y_base = base(x)[0]
        y_gauss = gauss(x)[0]
    assert torch.equal(y_base, y_gauss)


def test_state_dict_keys_match_baseline() -> None:
    """Every baseline key must exist in the GaussSplit model (checkpoint compat)."""
    base = set(_build(BASE_YAML).state_dict())
    gauss = set(_build(GAUSS_YAML).state_dict())
    assert base <= gauss, sorted(base - gauss)[:5]
    extra = gauss - base
    assert extra and all(".head." in k for k in extra), sorted(extra)[:5]


def test_head_is_registered_as_detect_subclass() -> None:
    """Ultralytics reads model.model[-1] as the head, so it must stay a Detect."""
    from ultralytics.nn.modules.head import Detect

    gauss = _build(GAUSS_YAML)
    assert isinstance(gauss.model[-1], DetectGaussSplit)
    assert isinstance(gauss.model[-1], Detect)
    assert gauss.model[-1].i == len(gauss.model) - 1


def test_parameter_counts() -> None:
    """23,690 head parameters on top of the 2,515,976-parameter baseline."""
    base = _build(BASE_YAML)
    gauss = _build(GAUSS_YAML)
    n_base = sum(p.numel() for p in base.parameters())
    n_gauss = sum(p.numel() for p in gauss.parameters())
    head = find_gauss_split_head(gauss)
    n_head = sum(p.numel() for p in head.head.parameters())
    assert n_head == 23_690, n_head
    assert n_gauss - n_base == n_head


def test_aux_loss_gradients_reach_only_the_head() -> None:
    """The auxiliary loss must train the head and leave the detector alone.

    ``iou_thr=0`` is forced because a randomly initialised detector emits no box
    that matches a GT at IoU 0.5, so every row would be invalid and the loss
    would be identically zero (that case is covered separately).
    """
    gauss = _build(GAUSS_YAML)
    gauss.train()
    head = find_gauss_split_head(gauss)
    head.iou_thr = 0.0
    img = torch.rand(2, 3, 256, 320)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.zeros(3, 1),
        "bboxes": torch.tensor(
            [[0.30, 0.40, 0.02, 0.05], [0.315, 0.40, 0.02, 0.05], [0.5, 0.5, 0.03, 0.06]]
        ),
    }
    preds = gauss(img)
    aux = head.aux_loss(preds, batch)
    assert aux.ndim == 0
    assert float(aux.detach()) > 0
    aux.backward()
    head_grads = [p.grad for p in head.head.parameters() if p.grad is not None]
    assert head_grads, "no gradient reached the head"
    assert any(g.abs().sum() > 0 for g in head_grads)
    assert all(p.grad is None for p in gauss.model[-1].cv2.parameters())


def test_aux_loss_is_zero_without_positive_targets() -> None:
    """A batch whose GTs are far apart must give a finite, small loss."""
    gauss = _build(GAUSS_YAML)
    gauss.train()
    head = find_gauss_split_head(gauss)
    img = torch.rand(1, 3, 256, 320)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.02, 0.04]]),
    }
    aux = head.aux_loss(gauss(img), batch)
    assert torch.isfinite(aux)
    assert head.last_stats["pos_rate"] == 0.0


def test_add_second_boxes_respects_threshold() -> None:
    """At init p(second)=0.119: nothing at 0.5, and a pushed-away box at 0.05.

    The init offsets are zero, so the extra box would be an exact copy of the
    source and the NMS at 0.5 removes it — that is the "no duplicate cloud"
    guard G3 working as intended. To see the extra box survive, the head is
    pushed to emit an offset of one source width.
    """
    gauss = _build(GAUSS_YAML)
    gauss.eval()
    head = find_gauss_split_head(gauss)
    with torch.no_grad():
        gauss(torch.randn(1, 3, 256, 320))
        head.head.fc3.bias[1] = 1.0  # dx = +1 width: the second box no longer overlaps
    preds = [torch.tensor([[10.0, 10.0, 18.0, 22.0, 0.9, 0.0]])]

    head.threshold = 0.5
    assert head.add_second_boxes(preds)[0].shape[0] == 1

    head.threshold = 0.05
    out = head.add_second_boxes(preds)[0]
    assert out.shape[0] == 2
    assert out[1, 0] > out[0, 0]  # shifted right, so it survives the 0.5 NMS
    assert out[1, 4] < out[0, 4]  # scored below its source


def test_decode_with_cols_filters_the_same_rows() -> None:
    """``cols`` must be filtered by the acceptance mask, not by position."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    score = torch.tensor([0.9, 0.8])
    out = torch.tensor([[8.0, 0.0, 0.0, 0.0], [-8.0, 0.0, 0.0, 0.0]])  # only row 0 accepted
    cols = torch.tensor([[1.0], [2.0]])
    extra, extra_score, extra_cols = PairSplit.decode(boxes, score, out, threshold=0.5, cols=cols)
    assert extra.shape[0] == 1
    assert float(extra_cols[0, 0]) == 1.0
    assert float(extra_score[0]) < 0.9


def test_decode_without_cols_keeps_two_return_values() -> None:
    """The old two-value contract must still hold for existing callers."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    score = torch.tensor([0.9])
    out = torch.tensor([[8.0, 0.0, 0.0, 0.0]])
    result = PairSplit.decode(boxes, score, out, threshold=0.5)
    assert len(result) == 2


# --------------------------------------------------------------------- step 3
def test_callback_freezes_detector_and_keeps_bn_in_eval() -> None:
    """Only ``*.head.*`` may train, and every BN must stay in eval mode.

    Ultralytics' ``freeze=`` is index-based and cannot express "this layer except
    its ``head`` child", so the callback does it by name. If BN drifted, the
    "frozen detector" premise would silently break.
    """
    from torch import nn

    from tinydet.training.gauss_split_callbacks import GaussSplitCallback

    gauss = _build(GAUSS_YAML)
    cb = GaussSplitCallback()
    frozen, trainable = cb._freeze_detector(gauss)
    head = find_gauss_split_head(gauss)

    assert trainable == 23_690, trainable
    assert frozen == sum(p.numel() for p in gauss.parameters()) - trainable
    assert all(p.requires_grad for p in head.head.parameters())
    assert not any(p.requires_grad for p in gauss.model[0].parameters())

    gauss.train()
    n_bn = cb._freeze_bn(gauss)
    assert n_bn > 0
    assert all(not m.training for m in gauss.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))


def test_bn_eval_survives_ultralytics_model_train() -> None:
    """The BN guard must run after Ultralytics' ``_model_train``.

    Ultralytics calls ``self._model_train()`` *after* ``on_train_epoch_start``
    and only re-evals BN modules inside ``freeze_layer_names`` (layers
    0..freeze-1). With ``freeze=29`` the detect head (layer 29) went back to
    train mode, so its running statistics drifted to the tile distribution and
    the "frozen detector" premise broke silently. ``on_train_batch_start`` is
    the first hook that runs after ``_model_train``.
    """
    from types import SimpleNamespace

    import torch
    from torch import nn

    from tinydet.training.gauss_split_callbacks import GaussSplitCallback

    gauss = _build(GAUSS_YAML)
    cb = GaussSplitCallback()
    cb._freeze_detector(gauss)

    bn0 = gauss.model[0].bn
    assert isinstance(bn0, nn.BatchNorm2d)
    before = bn0.running_mean.clone()

    # Emulate Ultralytics: _model_train() puts every BN back into train mode.
    gauss.train()
    assert bn0.training

    cb.on_train_batch_start(SimpleNamespace(model=gauss))
    assert not any(m.training for m in gauss.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))

    with torch.no_grad():
        gauss.model[0](torch.randn(1, 3, 64, 64))
    assert torch.equal(bn0.running_mean, before), "frozen BN statistics drifted"


def test_callbacks_register_only_for_gauss_split_models() -> None:
    """A baseline model must be untouched (default off for the 150e campaign).

    ``model.callbacks`` is created by the ``YOLO`` wrapper (``Model.__init__``), so
    the test uses the wrapper returned by ``build_model``, exactly like the trainer.
    """
    from tinydet.models.registry import build_model
    from tinydet.training.gauss_split_callbacks import register_gauss_split_callbacks

    base = build_model({"config": str(BASE_YAML)})
    gauss = build_model({"config": str(GAUSS_YAML)})
    before = {k: len(v) for k, v in base.callbacks.items()}

    assert register_gauss_split_callbacks(base) is False
    assert {k: len(v) for k, v in base.callbacks.items()} == before

    assert register_gauss_split_callbacks(gauss) is True
    assert len(gauss.callbacks["on_train_start"]) == len(base.callbacks["on_train_start"]) + 1


def test_criterion_is_untouched_for_baseline_models() -> None:
    """No ``_tinydet_head`` on a baseline criterion => no auxiliary loss at all."""
    base = _build(BASE_YAML)
    gauss = _build(GAUSS_YAML)

    assert getattr(base.init_criterion(), "_tinydet_head", None) is None
    assert getattr(gauss.init_criterion(), "_tinydet_head", None) is find_gauss_split_head(gauss)


# --------------------------------------------------------------------- step 4
def _stub_predictor(model):
    predictor = DetectionPredictor.__new__(DetectionPredictor)
    predictor.model = types.SimpleNamespace(model=model, names={0: "person"})
    predictor.batch = [["a.jpg"]]
    return predictor


def test_predictor_hook_appends_second_boxes() -> None:
    """``DetectionPredictor.construct_results`` must route through the head."""
    import numpy as np

    gauss = _build(GAUSS_YAML)
    gauss.eval()
    head = find_gauss_split_head(gauss)
    with torch.no_grad():
        gauss(torch.randn(1, 3, 256, 320))
        head.head.fc3.bias[1] = 1.0  # dx = +1 width, so the copy survives NMS

    predictor = _stub_predictor(gauss)
    preds = [torch.tensor([[10.0, 10.0, 18.0, 22.0, 0.9, 0.0]])]
    img = torch.zeros(1, 3, 256, 320)
    orig = [np.zeros((64, 80, 3), dtype=np.uint8)]

    head.threshold = 0.5
    assert len(DetectionPredictor.construct_results(predictor, preds, img, orig)[0].boxes) == 1

    head.threshold = 0.05
    assert len(DetectionPredictor.construct_results(predictor, preds, img, orig)[0].boxes) == 2


def test_validator_hook_appends_second_boxes() -> None:
    """``DetectionValidator.postprocess`` must add the head's boxes to the dict.

    The input is cloned per call because ``non_max_suppression`` may modify it in
    place; the real validator calls it once per batch, so this only matters for
    the test itself. ``threshold=1.1`` accepts nothing (p <= 1).
    """
    from ultralytics.models.yolo.detect.val import DetectionValidator

    gauss = _build(GAUSS_YAML)
    gauss.eval()
    head = find_gauss_split_head(gauss)
    with torch.no_grad():
        y = gauss(torch.randn(1, 3, 256, 320))
        head.head.fc3.bias[1] = 1.0

    validator = DetectionValidator.__new__(DetectionValidator)
    validator._tinydet_head = head
    validator.args = types.SimpleNamespace(
        conf=0.001, iou=0.7, single_cls=True, agnostic_nms=False, max_det=3000, task="detect"
    )
    validator.nc = 1
    validator.end2end = getattr(gauss, "end2end", False)

    head.threshold = 1.1
    n_off = len(DetectionValidator.postprocess(validator, y[0].clone())[0]["bboxes"])
    head.threshold = 0.0
    out = DetectionValidator.postprocess(validator, y[0].clone())[0]
    assert len(out["bboxes"]) > n_off
    assert out["bboxes"].shape[0] == out["conf"].shape[0] == out["cls"].shape[0]


def test_add_second_boxes_is_identity_below_threshold() -> None:
    """With no accepted second box the output must be bit-identical to the input.

    Regression guard: an earlier version re-ran NMS at 0.5 over *all* boxes, which
    silently tightened the detector's own NMS (400 -> 248 boxes) even when the
    head emitted nothing.
    """
    gauss = _build(GAUSS_YAML)
    gauss.eval()
    head = find_gauss_split_head(gauss)
    with torch.no_grad():
        gauss(torch.randn(1, 3, 256, 320))

    preds = [torch.rand(40, 6) * torch.tensor([300.0, 200.0, 300.0, 200.0, 1.0, 1.0])]
    preds[0][:, 2:4] += preds[0][:, :2]  # make x2 > x1, y2 > y1
    head.threshold = 1.1
    assert torch.equal(head.add_second_boxes(preds)[0], preds[0])


def test_zero_detections_give_finite_stats() -> None:
    """No detections at all must not produce NaN diagnostics.

    ``mean()`` of an empty tensor is NaN; ``p_second_mean`` feeds the epoch
    average in the callback, so one empty batch would poison every later value.
    """
    gauss = _build(GAUSS_YAML)
    gauss.train()
    head = find_gauss_split_head(gauss)
    head.feat = torch.rand(1, 32, 8, 16)
    batch = {
        "img": torch.rand(1, 3, 32, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.05]]),
    }
    # conf 1.0 rejects every detection (scores < 1), so the NMS output is empty.
    head.conf = 1.0
    aux = head.aux_loss(gauss(batch["img"]), batch)
    assert torch.isfinite(aux)
    assert head.last_stats["n_det"] == 0.0
    assert head.last_stats["p_second_mean"] == 0.0
    assert head.last_stats["pos_rate"] == 0.0
