"""Tiny object detection modules: DCAF v2.1."""

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

__all__ = [
    "CONTROL_PARAM_SUFFIXES",
    "DCAF",
    "BandBranch",
    "DilatedContextBranch",
    "LearnableGaussianScaleSpace",
    "RefinementBranch",
    "collect_dcaf_control_params",
    "compute_dcaf_analytical_macs",
    "configure_dcaf_control_param_group",
    "is_dcaf_control_param_name",
]
