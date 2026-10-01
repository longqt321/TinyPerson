import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.models.yolo.detect import DetectionValidator

from tinydet.evaluation.evaluator import (
    PredictionCollector,
    finalize_predictions,
    metric_summary,
    preflight,
)


def test_deferred_metrics_equal_native_and_export_original_coordinates(tmp_path):
    # A TP, duplicate and FP exercise matching, precision and recall, not just zero AP.
    batch = {'batch_idx': torch.tensor([0]), 'cls': torch.tensor([[0.]]),
             'bboxes': torch.tensor([[.5, .5, .25, .25]]),
             'ori_shape': [(32, 32)], 'ratio_pad': [((2., 2.), (0., 0.))],
             'im_file': ['test/images/a.jpg'], 'img': torch.zeros(1, 3, 64, 64)}
    pred = {'bboxes': torch.tensor([[24., 24., 40., 40.], [24., 24., 40., 40.], [0., 0., 10., 10.]]),
            'conf': torch.tensor([.9, .8, .7]), 'cls': torch.zeros(3), 'extra': torch.empty(3, 0)}
    native = DetectionValidator(save_dir=tmp_path / 'native', args={'plots': False})
    native.device = torch.device('cpu')
    native.seen, native.is_custom_json, native.build_gdict = 0, False, False
    native.metrics.names = {0: 'person'}
    native.update_metrics([pred], batch)
    native.metrics.process(plot=False)
    collector = PredictionCollector(save_dir=tmp_path, args={'plots': False})
    collector.device, collector.names, collector.samples = torch.device('cpu'), {0: 'person'}, []
    collector.update_metrics([pred], batch)
    collector.get_stats()
    result = finalize_predictions(tmp_path, tmp_path / 'predictions.pt', 'test')
    assert result == metric_summary(native.metrics)
    assert result['AP50'] > .9
    exported = json.loads((tmp_path / 'predictions/test.json').read_text())['images'][0]
    assert exported['boxes'][0] == [12., 12., 20., 20.]
    assert exported['scores'] == pytest.approx([.9, .8, .7])
    assert np.isfinite(list(result.values())).all()


def test_test_split_directory_preflight(tmp_path):
    import yaml

    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/best.pt').touch()
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images/a.jpg').touch()
    (tmp_path / 'data.yaml').write_text(yaml.safe_dump({'path': str(tmp_path), 'test': 'images'}))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'data': {'config': str(tmp_path/'data.yaml')}}))
    assert preflight(tmp_path, 'test')['data']['config']
    with pytest.raises(ValueError, match='split'):
        preflight(tmp_path, 'train')


def test_compare_reuses_training_best_validation(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path('modal').resolve()))
    app = importlib.import_module('app')
    metrics = {'AP': .3, 'source': 'training_final_best'}
    calls = []
    monkeypatch.setattr(app, '_preflight', SimpleNamespace(
        remote=lambda *a, **kw: metrics if kw['reuse_training'] else None))
    monkeypatch.setattr(app, '_evaluate', SimpleNamespace(remote=lambda *a: calls.append('gpu') or 'artifact'))
    monkeypatch.setattr(app, '_finish_evaluation', SimpleNamespace(remote=lambda *a: calls.append('cpu') or metrics))
    assert app._evaluate_pipeline('run', 'val', reuse_training=True) == metrics
    assert calls == []
    assert app._evaluate_pipeline('run', 'test') == metrics
    assert calls == ['gpu', 'cpu']
