"""Quality metrics for TomoMono alignment and reconstruction.

Which metric to trust for what:

- ``fourier_shell_correlation`` (FSC) — **the best measure of reconstruction
  quality.** It splits the tilt series into two independent half-sets,
  reconstructs each, and reports the spatial frequency out to which the two
  volumes agree, i.e. a real resolution estimate. Use this to judge how good a
  *reconstruction* is.

- ``reprojection_consistency_score`` (RCS) — **the best measure of alignment
  quality.** It reprojects the reconstruction and compares each synthetic
  projection against the measured one (per-angle NRMSE). A low, even score
  means the projections are mutually consistent, which is exactly what good
  alignment produces.

- ``sinogram_consistency_score`` — **a rough gauge, not a reliable score.** It
  checks the Helgason-Ludwig center-of-mass conditions on the sinogram. Treat
  it as a quick sanity check: it is handy for spotting outlier projections,
  getting a coarse sense of how close the alignment is, and visualizing the
  central-slice sinogram — but do not rely on its absolute value.
"""

from metrics.sinogram_consistency import sinogram_consistency_score
from metrics.reprojection_consistency import reprojection_consistency_score
from metrics.fsc import fourier_shell_correlation

__all__ = [
    "sinogram_consistency_score",
    "reprojection_consistency_score",
    "fourier_shell_correlation",
]
