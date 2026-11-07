"""Tiny object detection modules: DCAF v2.1, P2Refine."""

from tinydet.modules.dcaf import (
    CONTROL_PARAM_SUFFIXES,
    DCAF,
    BandBranch,
    DilatedContextBranch,
    LearnableGaussianScaleSpace,
    RefinementBranch,
    collect_dcaf_control_params,
    compute_dcaf_analytical_macs,
    configure_dcaf_control_param_group,
    is_dcaf_control_param_name,
)
from tinydet.modules.p2_refine import P2Refine

__all__ = [
    "CONTROL_PARAM_SUFFIXES",
    "DCAF",
    "BandBranch",
    "DilatedContextBranch",
    "LearnableGaussianScaleSpace",
    "P2Refine",
    "RefinementBranch",
    "collect_dcaf_control_params",
    "compute_dcaf_analytical_macs",
    "configure_dcaf_control_param_group",
    "is_dcaf_control_param_name",
]
