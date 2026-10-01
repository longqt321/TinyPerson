from pathlib import Path

import pytest
import torch
from PIL import Image
from ultralytics import YOLO

from tinydet.analysis.heavy_bottom import (
    activation_ratio,
    erf_extent,
    letterbox,
    make_probe,
    map_box,
    region_masks,
    resolve_stages,
    size_bin,
)
from tinydet.evaluation.size_metrics import ap_for_bin
from tinydet.utils.comparison import load_comparison


def test_stages_and_architecture_wiring():
    experiments = load_comparison('configs/experiment/heavy_bottom.yaml')
    assert [item['name'] for item in experiments] == [
        'deep_heavy', 'baseline', 'moderate_heavy_bottom', 'strong_heavy_bottom']
    assert experiments[1]['model']['config'] == 'configs/model/yolo26n.yaml'
    for key in ('seed', 'data', 'train', 'hardware', 'output'):
        assert all(item[key] == experiments[0][key] for item in experiments)
    assert experiments[0]['train']['epochs'] == 50
    for item, depths in zip(experiments, ((1, 1, 2, 2), (1, 1, 1, 1),
                                         (2, 2, 1, 1), (3, 3, 1, 1))):
        model = YOLO(item['model']['config']).model
        stages = resolve_stages(model)
        assert stages == {'C2': 2, 'C3': 4, 'C4': 6, 'C5': 8}
        assert tuple(len(model.model[index].m) for index in stages.values()) == depths
        assert sum(p.numel() for p in model.parameters()) > 0
    with pytest.raises(ValueError, match='Expected'):
        resolve_stages(torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1)))


def test_letterbox_bins_and_regions():
    image, scale, left, top = letterbox(Image.new('RGB', (100, 50)), 200)
    assert image.size == (200, 200)
    assert (scale, left, top) == (2, 0, 50)
    assert map_box([10, 5, 20, 15], scale, left, top) == [20, 60, 40, 80]
    assert [size_bin(x, x) for x in (0, 7.99, 8, 15.99, 16, 31.99, 32)] == [
        '0-8', '0-8', '8-16', '8-16', '16-32', '16-32', '>32']
    obj, bg = region_masks([1, 1, 3, 3], 4, 4, 64)
    assert obj.any() and bg.any() and not (obj & bg).any()
    ratio = activation_ratio(torch.ones(3, 4, 4), [1, 1, 3, 3], 64)
    assert ratio is not None and ratio[0] == pytest.approx(1)
    mapped, _ = region_masks([16, 16, 32, 32], 4, 4, 64)
    assert mapped.sum() == 1 and mapped[1, 1]
    obj, bg = region_masks([0, 0, 2, 2], 4, 4, 64)
    assert obj.any() and bg.any() and not (obj & bg).any()


def test_erf_energy():
    energy = torch.ones(9, 9)
    extent = erf_extent(energy, (4, 4))
    assert extent is not None and extent[0][0] <= extent[1][0]
    assert all(0 < item[2] <= 1 for item in extent)
    impulse = torch.zeros(9, 9)
    impulse[4, 4] = 1
    assert erf_extent(impulse, (4, 4))[0] == (1, 1, 1/81)
    assert erf_extent(torch.zeros(9, 9), (4, 4)) is None
    assert erf_extent(torch.full((9, 9), float('nan')), (4, 4)) is None


def test_probe_reuse_deterministic(tmp_path):
    data = Path('datasets/TinyTiny_person/data.yaml')
    one, two = tmp_path / 'one.json', tmp_path / 'two.json'
    first = make_probe(data, one, size=640, count=8, seed=42)
    second = make_probe(data, two, size=640, count=8, seed=42)
    assert first == second and one.read_text().replace('one.json', '') == two.read_text().replace('two.json', '')
    assert len(first) == 8 and all(Path(row['image']).is_file() for row in first)


def test_size_ap_ignores_excluded_gt():
    samples = [{'gt': [{'box': [0, 0, 8, 8], 'bin': '0-8'},
                       {'box': [20, 20, 40, 40], 'bin': '16-32'}],
                'pred': [([20, 20, 40, 40], .99), ([0, 0, 8, 8], .8)]}]
    assert ap_for_bin(samples, '0-8', .5) == pytest.approx(1)
    assert ap_for_bin(samples, '16-32', .5) == pytest.approx(1)
    samples[0]['pred'][0] = ([50, 50, 60, 60], .99)
    assert ap_for_bin(samples, '0-8', .5) < 1


def test_modal_diagnostics_defers_volume_lookup(monkeypatch, capsys):
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path('modal').resolve()))
    app = importlib.import_module('app')
    calls = []
    monkeypatch.setattr(app, '_prepare_heavy_bottom', SimpleNamespace(
        remote=lambda *args: calls.append(args) or ('checkpoint', 'probe', 'output', None)))
    monkeypatch.setattr(app, '_heavy_bottom_diagnose', SimpleNamespace(
        remote=lambda *args: '/mnt/runs/result'))
    monkeypatch.setattr(app, '_finish_heavy_bottom', SimpleNamespace(remote=lambda output: output))
    app.heavy_bottom_diagnose.info.raw_f(run='yolo26n/real-run-id')
    assert calls[0][0] == 'yolo26n/real-run-id'
    assert calls[0][1] == '' and calls[0][3] == ''
    assert '/mnt/runs/result' in capsys.readouterr().out
    with pytest.raises(ValueError, match='placeholder'):
        app.heavy_bottom_diagnose.info.raw_f(run='baseline/RUN_ID')


def test_erf_profile_matches_sorted_pixels():
    from tinydet.analysis.heavy_bottom import energy_profile, profile_extent

    generator = torch.Generator().manual_seed(42)
    energy = torch.rand(31, 29, generator=generator)
    for center in ((14, 15), (14.25, 15.7), (0.5, 0.2)):
        assert profile_extent(energy_profile(energy, center), center, 31, 29) == erf_extent(energy, center)
    assert profile_extent(torch.zeros(32), (14, 15), 31, 29) is None


def test_probe_does_not_resize_images(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Probe creation should read dimensions only')

    monkeypatch.setattr(Image.Image, 'resize', forbidden)
    assert len(make_probe(Path('datasets/TinyTiny_person/data.yaml'), tmp_path / 'probe.json', count=8)) == 8


def test_size_ap_reuses_overlaps(monkeypatch):
    import tinydet.evaluation.size_metrics as metrics

    samples = [{'gt': [{'box': [0, 0, 7, 7], 'bin': '0-8'},
                       {'box': [20, 20, 40, 40], 'bin': '16-32'}],
                'pred': [([20, 20, 40, 40], .99), ([0, 0, 7, 7], .8), ([0, 0, 7, 7], .7)]}]
    expected = {name: [metrics.ap_for_bin(samples, name, float(t))
                       for t in torch.linspace(.5, .95, 10)] for name in metrics.BINS}
    calls = []
    original = metrics.iou
    monkeypatch.setattr(metrics, 'iou', lambda box, boxes: calls.append(1) or original(box, boxes))
    result = metrics.size_ap(samples)
    assert len(calls) == 3
    for name, values in expected.items():
        assert result[name]['AP50'] == values[0]
        if values[0] is not None:
            assert result[name]['AP50-95'] == pytest.approx(sum(values)/10)


def test_modal_preflight_failure_never_starts_gpu(monkeypatch):
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path('modal').resolve()))
    app = importlib.import_module('app')
    def fail(*args, **kwargs):
        raise FileNotFoundError('missing best.pt')
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU must not start')
    monkeypatch.setattr(app, '_prepare_heavy_bottom', SimpleNamespace(remote=fail))
    monkeypatch.setattr(app, '_heavy_bottom_diagnose', SimpleNamespace(remote=forbidden))
    monkeypatch.setattr(app, '_preflight', SimpleNamespace(remote=fail))
    monkeypatch.setattr(app, '_evaluate', SimpleNamespace(remote=forbidden))
    with pytest.raises(FileNotFoundError):
        app.heavy_bottom_diagnose.info.raw_f(run='model/missing')
    with pytest.raises(FileNotFoundError):
        app.evaluate.info.raw_f(run='model/missing', split='test')


def test_grouped_diagnostics_cpu_finalize(tmp_path, monkeypatch):
    import json

    import tinydet.analysis.heavy_bottom as diagnostics

    # Two targets in one image must share the forward graph and NMS pass.
    model = YOLO('configs/model/yolo26n.yaml')
    checkpoint = tmp_path / 'model.pt'
    model.save(checkpoint)
    image = tmp_path / 'image.png'
    Image.new('RGB', (64, 64), (100, 100, 100)).save(image)
    rows = [{'image': str(image), 'gt_id': index, 'box_input': box, 'size_bin': '8-16'}
            for index, box in enumerate(([20, 20, 30, 30], [35, 35, 45, 45]))]
    probe = tmp_path / 'probe.json'
    probe.write_text(json.dumps({'size': 64, 'instances': rows}))
    nms_calls = []
    original_nms = diagnostics.non_max_suppression
    monkeypatch.setattr(diagnostics, 'non_max_suppression',
                        lambda *a, **kw: nms_calls.append(1) or original_nms(*a, **kw))
    def forbidden(*args, **kwargs):
        raise AssertionError('Plotting must not run during collection')
    monkeypatch.setattr(diagnostics, '_panel', forbidden)
    output = diagnostics.collect_diagnostics(checkpoint, probe, tmp_path / 'result', panels=0)
    assert len(nms_calls) == 1
    assert not (output / 'metrics').exists()
    payload = torch.load(output / 'diagnostics.pt', weights_only=True)
    assert payload['metadata']['forward_passes'] == 1
    assert len(payload['items']) == 2
    plots = []
    monkeypatch.setattr(diagnostics, '_panel', lambda *a: plots.append(a))
    diagnostics.finalize_diagnostics(output)
    assert (output / 'metrics/activation_ratio.csv').is_file()
    assert (output / 'metrics/erf.csv').is_file()
    assert len(plots) == 3
