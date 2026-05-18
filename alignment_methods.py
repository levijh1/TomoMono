"""
Backwards-compatibility shim.

The contents of this module have moved into the ``alignment``, ``metrics``,
and ``filters`` packages. Re-exports are kept here so existing imports
(``from alignment_methods import X``) continue to work without changes to
notebooks and downstream scripts.

New code should import from the specific subpackage instead, e.g.:

    from alignment import cross_correlate_align, PMA
    from metrics import reprojection_consistency_score, fourier_shell_correlation
    from filters import kovacik_filter
"""

from alignment import (
    cross_correlate_align,
    compute_grad_image,
    projection_matching_alignment,
    PMA,
    vertical_mass_fluctuation_align,
    tomopy_align,
    optical_flow_align,
    shift_min_to_middle,
    bilateralFilter,
    find_optimal_rotation,
    rotate_correlate_align,
    unrotate,
)
from metrics import (
    sinogram_consistency_score,
    reprojection_consistency_score,
    fourier_shell_correlation,
    reconstruction_sharpness_score,
)
from filters import kovacik_filter

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
    "sinogram_consistency_score",
    "reprojection_consistency_score",
    "fourier_shell_correlation",
    "reconstruction_sharpness_score",
    "kovacik_filter",
]
