"""Explicit registration and hooks for custom QAC and QAC v2 modules parsed from YAML."""


def _install_hooks() -> None:
    """Apply model YAML hooks once after Ultralytics constructs a detector."""
    from ultralytics.nn.tasks import DetectionModel

    if getattr(DetectionModel, "_tinydet_hooks", False):
        return
    original_criterion = DetectionModel.init_criterion

    def init_criterion(self):
        from tinydet.modules.qac import QACDetect, QualityAwareDetectionLoss, QualityAwareE2ELoss
        from tinydet.modules.qac_v2 import (
            QACDetectV2,
            QualityAwareDetectionLossV2,
            QualityAwareE2ELossV2,
        )

        is_qac = isinstance(self.model[-1], QACDetect) or bool(self.yaml.get("qac", False))
        is_qac_v2 = isinstance(self.model[-1], QACDetectV2) or bool(self.yaml.get("qac_v2", False))
        if is_qac:
            gamma = float(self.yaml.get("quality_gamma", 1.0))
            min_conf = float(self.yaml.get("quality_min_conf", 0.25))
            criterion = (
                QualityAwareE2ELoss(self, quality_gamma=gamma, quality_min_conf=min_conf)
                if getattr(self.model[-1], "one2one_cv2", None) is not None
                else QualityAwareDetectionLoss(
                    self, quality_gamma=gamma, quality_min_conf=min_conf
                )
            )
        elif is_qac_v2:
            gamma = float(self.yaml.get("quality_gamma", 1.0))
            min_conf = float(self.yaml.get("quality_min_conf", 0.25))
            nwd_constant = float(self.yaml.get("nwd_constant", 12.8))
            nwd_weight = float(self.yaml.get("nwd_weight", 0.50))
            beta_min = float(self.yaml.get("beta_min", 1.5))
            beta_max = float(self.yaml.get("beta_max", 6.0))
            area_ref = float(self.yaml.get("area_ref", 576.0))
            criterion = (
                QualityAwareE2ELossV2(
                    self,
                    quality_gamma=gamma,
                    quality_min_conf=min_conf,
                    nwd_constant=nwd_constant,
                    nwd_weight=nwd_weight,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    area_ref=area_ref,
                )
                if getattr(self.model[-1], "one2one_cv2", None) is not None
                else QualityAwareDetectionLossV2(
                    self,
                    quality_gamma=gamma,
                    quality_min_conf=min_conf,
                    nwd_constant=nwd_constant,
                    nwd_weight=nwd_weight,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    area_ref=area_ref,
                )
            )
        else:
            criterion = original_criterion(self)

        beta = float(self.yaml.get("tal_beta", 6.0))
        losses = (criterion.one2many, criterion.one2one) if hasattr(criterion, "one2many") else (criterion,)
        for loss in losses:
            if hasattr(loss, "assigner"):
                loss.assigner.beta = beta
        return criterion

    DetectionModel.init_criterion = init_criterion
    DetectionModel._tinydet_hooks = True


def register_ultralytics_modules() -> None:
    """Register TinyDet modules before Ultralytics parses a model YAML."""
    from ultralytics.nn import tasks

    from tinydet.modules.qac import QACDetect
    from tinydet.modules.qac_v2 import QACDetectV2

    tasks.QACDetect = QACDetect
    tasks.QACDetectV2 = QACDetectV2
    _install_hooks()
