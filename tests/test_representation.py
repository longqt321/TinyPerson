"""Pure CPU checks for the new representation diagnostics."""
import numpy as np
import torch

from tinydet.analysis.representation import (
    average_precision, background_centers, linear_probe, relative_feature_delta,
    shift_tensor, split_image, vector_at,
)


def test_group_split_is_repeatable():
    assert split_image("images/one.jpg", 42) == split_image("images/one.jpg", 42)


def test_ranking_ap():
    assert average_precision([1, 0, 1], [0.9, 0.1, 0.8]) == 1.0


def test_probe_separates_easy_examples():
    x = [[-2.0], [-1.0], [1.0], [2.0]]
    scores = linear_probe(x, [0, 0, 1, 1], x)
    assert scores[0] < scores[1] < scores[2] < scores[3]


def test_vector_at_respects_feature_coordinates():
    tensor = torch.arange(16.0).reshape(1, 1, 4, 4)
    assert vector_at(tensor, (5, 5), 8).tolist() == [10.0]


def test_shift_no_wrap():
    tensor = torch.zeros(1, 1, 4, 4)
    tensor[0, 0, 0, 0] = 1
    out = shift_tensor(tensor, 1, 1, fill=0)
    assert out[0, 0, 1, 1] == 1
    assert out[0, 0, 0, 0] == 0
    assert out.sum() == 1


def test_delta_zero_and_nonzero():
    base = torch.ones(1, 2, 4, 4)
    changed = base.clone()
    assert relative_feature_delta(base, changed, [0, 0, 4, 4], 4) == 0
    changed[..., 0, 0] = 0
    assert relative_feature_delta(base, changed, [0, 0, 4, 4], 4) > 0


def test_background_sample_avoids_target():
    rng = np.random.default_rng(0)
    boxes = [{"box": [30, 30, 34, 34]}]
    centers = background_centers(None, boxes, 100, 64, rng)
    assert all(not (27 <= x <= 37 and 27 <= y <= 37) for x, y in centers)


def test_output_confined_to_trusted_root(tmp_path):
    import pytest

    from tinydet.analysis.representation import validate_output_directory

    root = tmp_path / "outputs"
    root.mkdir()
    valid = root / "representation"
    assert validate_output_directory(root, valid) == valid.resolve()
    with pytest.raises(ValueError):
        validate_output_directory(root, root)
    with pytest.raises(ValueError):
        validate_output_directory(root, root / ".." / "secrets")
    with pytest.raises(ValueError):
        validate_output_directory(root, tmp_path / "outside")


def test_symlink_escape_rejected(tmp_path):
    import pytest

    from tinydet.analysis.representation import validate_output_directory

    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_output_directory(root, root / "link" / "report")


def test_study_requires_explicit_trusted_root(tmp_path):
    import pytest

    from tinydet.analysis.representation import study

    with pytest.raises(ValueError, match="allowed_output_root"):
        study(tmp_path / "best.pt", tmp_path / "data.yaml", tmp_path / "outputs" / "trial", {})
