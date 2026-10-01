from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner

from tinydet.modules.nwd import NWDBboxLoss, nwd_similarity
from tinydet.modules.ultralytics_adapter import (
    ModuleDetectionModel,
    NWDTaskAlignedAssigner,
)
from tinydet.utils.comparison import load_comparison

CONFIG = 'configs/experiment/ablation_nwd.yaml'


def test_nwd_formula_pixels_and_gradients():
    box = torch.tensor([[0., 0., 4., 4.]], requires_grad=True)
    same = nwd_similarity(box, box.detach())
    assert same.item() == 1
    (1 - same).sum().backward()
    assert torch.isfinite(box.grad).all()
    near = torch.tensor([[3., 4., 7., 8.]])
    far = torch.tensor([[6., 8., 10., 12.]])
    larger = torch.tensor([[0., 0., 8., 8.]])
    assert torch.allclose(nwd_similarity(box, near, 5), torch.exp(torch.tensor([-1.])))
    assert nwd_similarity(box, near) > nwd_similarity(box, far)
    assert nwd_similarity(box, larger) < same
    assert torch.allclose(nwd_similarity(box, larger), nwd_similarity(larger, box))
    box.grad.zero_()
    (1 - nwd_similarity(box, near)).sum().backward()
    assert torch.isfinite(box.grad).all() and box.grad.abs().sum() > 0
    for stride in (8., 16., 32.):
        assert torch.allclose(nwd_similarity((box / stride) * stride, (near / stride) * stride),
                              nwd_similarity(box, near))


def test_nwd_tal_preserves_tal_and_uses_pixel_boxes():
    kwargs = {'topk': 2, 'topk2': 1, 'num_classes': 1, 'alpha': .5, 'beta': 6., 'stride': [8, 16]}
    baseline = TaskAlignedAssigner(**kwargs)
    tal = NWDTaskAlignedAssigner(**kwargs, constant=12.8)
    score = torch.tensor([[[.9], [.8], [.7], [.6]]])
    pred = torch.tensor([[[3., 3., 9., 9.], [9., 3., 15., 9.],
                          [3., 9., 9., 15.], [9., 9., 15., 15.]]])
    points = torch.tensor([[6., 6.], [12., 6.], [6., 12.], [12., 12.]])
    gt = torch.tensor([[[4., 4., 10., 10.]]])
    args = (score, pred, points, torch.zeros(1, 1, 1), gt, torch.ones(1, 1, 1))
    original = baseline(*args)
    assert all(torch.equal(a, b) for a, b in zip(original, TaskAlignedAssigner(**kwargs)(*args)))
    changed = tal(*args)
    candidate_mask = baseline.select_candidates_in_gts(points, gt, args[-1])
    baseline_metric, baseline_overlap = baseline.get_box_metrics(score, pred, args[3], gt, candidate_mask)
    tal_metric, tal_overlap = tal.get_box_metrics(score, pred, args[3], gt, candidate_mask)
    assert torch.allclose(tal_metric, score.transpose(1, 2).pow(.5) * tal_overlap.pow(6) * candidate_mask)
    assert not torch.allclose(baseline_overlap, tal_overlap)
    assert not torch.allclose(baseline_metric, tal_metric)
    assert all(a.shape == b.shape for a, b in zip(original, changed))
    assert torch.equal(original[3], changed[3]) and changed[3].any()
    assert all(torch.isfinite(x).all() for x in changed)
    assert (changed[2][changed[3]] > 0).all()
    assert (changed[2][changed[3]] < 1).all()  # quality-aware, not binary targets
    for stride in (8., 16.):
        assert torch.allclose(tal.iou_calculation(gt[0], pred[0, :1]),
                              nwd_similarity((gt[0] / stride) * stride,
                                             (pred[0, :1] / stride) * stride))
    assert tal.topk == baseline.topk and tal.topk2 == baseline.topk2
    assert tal.alpha == baseline.alpha and tal.beta == baseline.beta


@pytest.mark.parametrize('variant', range(4))
def test_ablation_loss_backward_and_checkpoint_options(variant, tmp_path):
    experiment = load_comparison(CONFIG)[variant]
    config = yaml.safe_load(Path(experiment['model']['config']).read_text())
    config['scale'] = 'n'
    if 'modules' in experiment['model']:
        config['tinydet_modules'] = experiment['model']['modules']
    torch.manual_seed(42)
    model = ModuleDetectionModel(config, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    batch = {'img': torch.rand(2, 3, 64, 64), 'batch_idx': torch.tensor([0., 1.]),
             'cls': torch.zeros(2, 1), 'bboxes': torch.tensor([[.13, .13, .04, .04], [.5, .5, .2, .2]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[-1].one2one_cv2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[-1].cv2.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / 'model.pt'
    torch.save(deepcopy(model).half(), path)
    loaded = torch.load(path, weights_only=False).float()
    assert loaded.yaml.get('tinydet_modules', {}) == experiment['model'].get('modules', {})
    criterion = loaded.init_criterion()
    for branch in (criterion.one2many, criterion.one2one):
        assert type(branch.assigner) is (NWDTaskAlignedAssigner if variant in (2, 3) else TaskAlignedAssigner)
        assert type(branch.bbox_loss) is (NWDBboxLoss if variant in (1, 3) else BboxLoss)


def test_ablation_factors_and_shared_settings():
    experiments = load_comparison(CONFIG)
    assert [e['name'] for e in experiments] == ['baseline', 'nwd_loss', 'nwd_tal', 'nwd_full']
    for key in ('data', 'train', 'hardware', 'output', 'seed'):
        assert all(e[key] == experiments[0][key] for e in experiments)
    assert len({e['model']['config'] for e in experiments}) == 1
    factors = [(bool(e['model'].get('modules', {}).get('nwd_tal', {}).get('enabled')),
                bool(e['model'].get('modules', {}).get('nwd', {}).get('enabled')))
               for e in experiments]
    assert factors == [(False, False), (False, True), (True, False), (True, True)]
    assert all(e['model'].get('pretrained') is None for e in experiments)


def test_empty_gt_backward():
    experiment = load_comparison(CONFIG)[3]
    config = yaml.safe_load(Path(experiment['model']['config']).read_text())
    config.update(scale='n', tinydet_modules=experiment['model']['modules'])
    model = ModuleDetectionModel(config, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    loss, _ = model({'img': torch.rand(2, 3, 64, 64), 'batch_idx': torch.empty(0),
                     'cls': torch.empty(0, 1), 'bboxes': torch.empty(0, 4)})
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
