import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tinydet.evaluation.iou_metrics import ap_by_iou
from tinydet.training.iou_logging import log_ap_by_iou


def test_ap_by_iou_averages_classes_and_logs_epoch(tmp_path):
    box = SimpleNamespace(all_ap=np.array([[0.8 - i * 0.05 for i in range(10)],
                                            [0.6 - i * 0.05 for i in range(10)]]))
    values = ap_by_iou(box)
    assert len(values) == 10
    assert abs(values['AP0.50'] - 0.7) < 1e-12
    assert abs(values['AP0.95'] - 0.25) < 1e-12
    trainer = SimpleNamespace(validator=SimpleNamespace(metrics=SimpleNamespace(box=box)),
                              epoch=2, save_dir=Path(tmp_path) / 'training')
    log_ap_by_iou(trainer)
    with (tmp_path / 'metrics' / 'ap_iou.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]['epoch'] == '3'
    assert abs(float(rows[0]['AP0.75']) - 0.45) < 1e-12
