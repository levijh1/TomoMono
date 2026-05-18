"""Alignment methods for TomoMono — XCA, PMA, VMF, and legacy variants."""

from alignment.cross_correlate import cross_correlate_align, compute_grad_image
from alignment.pma import projection_matching_alignment, PMA
from alignment.vmf import vertical_mass_fluctuation_align
from alignment.legacy import (
    tomopy_align,
    optical_flow_align,
    shift_min_to_middle,
    bilateralFilter,
    find_optimal_rotation,
    rotate_correlate_align,
    unrotate,
)

__all__ = [
    "cross_correlate_align",
    "compute_grad_image",
    "projection_matching_alignment",
    "PMA",  # backwards-compat alias
    "vertical_mass_fluctuation_align",
    "tomopy_align",
    "optical_flow_align",
    "shift_min_to_middle",
    "bilateralFilter",
    "find_optimal_rotation",
    "rotate_correlate_align",
    "unrotate",
]
