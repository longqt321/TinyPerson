from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.loss import BboxLoss

from tinydet.modules.nwd import NWDBboxLoss, nwd_similarity
from tinydet.modules.rfla import RFLAAssigner, receptive_field_similarity
from tinydet.modules.ultralytics_adapter import ModuleDetectionModel
from tinydet.utils.comparison import load_comparison


def test_nwd_pixels_and_identical_box_gradients():
    pred = torch.tensor([[0., 0., 4., 4.]], requires_grad=True)
    same = nwd_similarity(pred, pred.detach())
    assert same.item() == 1
    (1 - same).sum().backward()
    assert torch.isfinite(pred.grad).all()
    shift = torch.tensor([[3., 4., 7., 8.]])
    assert torch.allclose(nwd_similarity(pred, shift, 5), torch.exp(torch.tensor([-1.])))
    pred.grad.zero_()
    (1 - nwd_similarity(pred, shift)).sum().backward()
    assert pred.grad.abs().sum() > 0
    # A fixed pixel shift must have the same loss at every stride.
    for stride in (8., 16., 32.):
        value = nwd_similarity((pred / stride) * stride, (shift / stride) * stride)
        assert torch.allclose(value, nwd_similarity(pred, shift))


def test_rfla_outside_gt_conflicts_and_empty_images():
    points = torch.tensor([[4., 4.], [12., 4.], [20., 4.]])
    boxes = torch.tensor([[[6., 2., 10., 6.], [5., 1., 11., 7.]],
                          [[0., 0., 0., 0.], [0., 0., 0., 0.]]])
    assigner = RFLAAssigner(1, topk=2, extra_topk=1, chunk_size=1)
    assigner.sigma = torch.ones(3) * 4
    result = assigner(torch.ones(2, 3, 1), torch.zeros(2, 3, 4), points,
                      torch.zeros(2, 2, 1), boxes, boxes.sum(-1, keepdim=True) > 0)
    assert result[3][0].sum() == 2
    assert not result[3][1].any()
    assert (result[4][0, result[3][0]] == 0).all()  # smaller GT wins collision
    assert result[2][0].sum() == 2  # nonzero supervision despite zero predicted IoU
    assigner = RFLAAssigner(1, one_to_one=True)
    assigner.sigma = torch.ones(3) * 4
    assert assigner(torch.ones(2, 3, 1), torch.zeros(2, 3, 4), points,
                    torch.zeros(2, 2, 1), boxes, boxes.sum(-1, keepdim=True) > 0)[3][0].sum() <= 2
    assert torch.allclose(receptive_field_similarity(torch.tensor([[2., 2.]]),
                          torch.tensor([2.]), torch.tensor([[0., 0., 4., 4.]])), torch.ones(1, 1))


@pytest.mark.parametrize('variant', range(4))
def test_p3_ablation_loss_backward_and_checkpoint_options(variant, tmp_path):
    experiment = load_comparison('configs/experiment/ablation_rfla_nwd.yaml')[variant]
    config = yaml.safe_load(Path(experiment['model']['config']).read_text())
    config['scale'] = 'n'
    config['tinydet_modules'] = experiment['model']['modules']
    torch.manual_seed(42)
    model = ModuleDetectionModel(config, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    batch = {'img': torch.rand(2, 3, 64, 64), 'batch_idx': torch.tensor([0., 1.]),
             'cls': torch.zeros(2, 1), 'bboxes': torch.tensor([[.13, .13, .04, .04], [.5, .5, .2, .2]])}
    params_before = sum(p.numel() for p in model.parameters())
    loss, _metrics = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[-1].one2one_cv2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[-1].cv2.parameters())
    assert sum(p.numel() for p in model.parameters()) == params_before == 144106
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Verify the options survive the same model serialization used by checkpoints.
    path = tmp_path / 'model.pt'
    torch.save(deepcopy(model).half(), path)
    loaded = torch.load(path, weights_only=False).float()
    assert loaded.yaml['tinydet_modules'] == experiment['model']['modules']
    criterion = loaded.init_criterion()
    assert type(criterion.one2many.bbox_loss) is (NWDBboxLoss if variant in (2, 3) else BboxLoss)


def test_ablation_changes_only_training_modules():
    experiments = load_comparison('configs/experiment/ablation_rfla_nwd.yaml')
    for key in ('data', 'train', 'hardware', 'output', 'seed'):
        assert all(e[key] == experiments[0][key] for e in experiments)
    assert len({e['model']['config'] for e in experiments}) == 1


def test_hla_second_stage_fills_only_unassigned_points():
    # Two levels: stage 1 matches sigma=2, shrink makes sigma=4 the exact match.
    assigner = RFLAAssigner(1, topk=1, extra_topk=1, shrink=.5)
    assigner.sigma = torch.tensor([2., 4.])
    points = torch.tensor([[2., 2.], [2., 2.]])
    result = assigner(torch.zeros(1, 2, 1), torch.zeros(1, 2, 4), points,
                      torch.zeros(1, 1, 1), torch.tensor([[[0., 0., 4., 4.]]]), torch.ones(1, 1, 1))
    assert result[3].sum() == 2
    one = RFLAAssigner(1, one_to_one=True)
    one.sigma = assigner.sigma
    result = one(torch.zeros(1, 2, 1), torch.zeros(1, 2, 4), points,
                 torch.zeros(1, 1, 1), torch.tensor([[[0., 0., 4., 4.]]]), torch.ones(1, 1, 1))
    assert result[3].sum() == 1


def test_empty_gt_backward():
    experiment = load_comparison('configs/experiment/ablation_rfla_nwd.yaml')[3]
    config = yaml.safe_load(Path(experiment['model']['config']).read_text())
    config['scale'] = 'n'
    config['tinydet_modules'] = experiment['model']['modules']
    model = ModuleDetectionModel(config, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    loss, _ = model({'img': torch.rand(2, 3, 64, 64), 'batch_idx': torch.empty(0),
                     'cls': torch.empty(0, 1), 'bboxes': torch.empty(0, 4)})
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('family', ['v8/yolov8.yaml', '11/yolo11.yaml'])
def test_dfl_yolo_adapters(family):
    import ultralytics

    config = yaml.safe_load((Path(ultralytics.__file__).parent / 'cfg/models' / family).read_text())
    config.update(scale='n', nc=1, tinydet_modules={
        'nwd': {'enabled': True}, 'rfla': {'enabled': True, 'inside_only': True}})
    model = ModuleDetectionModel(config, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    loss, _ = model({'img': torch.rand(2, 3, 64, 64), 'batch_idx': torch.tensor([0., 1.]),
                     'cls': torch.zeros(2, 1), 'bboxes': torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]])})
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.yaml['tinydet_modules']['rfla']['inside_only'] = False
    with pytest.raises(ValueError, match='signed box distances'):
        model.init_criterion()
