from tinydet.utils.comparison import load_comparison


def test_comparison_shares_settings_and_changes_only_model():
    experiments = load_comparison("configs/experiment/comparison.yaml")
    assert len(experiments) == 4
    assert experiments[0]["output"]["wandb_project"] == "tiny-person-research"
    assert len({item["model"]["config"] for item in experiments}) == 4
    for key in ("seed", "data", "train", "hardware", "output"):
        assert all(item[key] == experiments[0][key] for item in experiments)


def test_comparison_training_args_parse_with_ultralytics():
    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    train = load_comparison("configs/experiment/comparison.yaml")[0]["train"]
    cfg = get_cfg(DEFAULT_CFG, overrides=train)
    assert cfg.epochs == 50
    assert cfg.patience == 15
    assert cfg.mosaic == 1.0
    assert cfg.close_mosaic == 10
    assert cfg.compile is False
