"""Ultralytics trainer adapter for installing MSR feature modules."""

from copy import deepcopy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from tinydet.modules.msr import install_msr_feature_module, validate_feature_module


class MSRDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, feature_module=None, **kwargs):
        self.feature_module = (
            validate_feature_module(feature_module) if feature_module is not None else None
        )
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import yaml_model_load

        config = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        model = self.set_model_names_for_load(DetectionModel(
            config, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
        ))
        if self.feature_module is not None:
            install_msr_feature_module(model, self.feature_module)
        if weights is not None:
            model.load(weights)
        return model
