"""
Centralized array-backend and optional-GPU detection for the TomoMono package.

Other modules should import from here instead of running their own try/except
ladders. The probes execute exactly once at import time, with a single ``print``
summarising the active backends.

Exports
-------
xp : module
    ``cupy`` if a working CUDA GPU is available, else ``numpy``.
cp : module or None
    The CuPy module when usable, else ``None``.
torch : module or None
    The PyTorch module when a usable CUDA/MPS device is present, else ``None``.
    Used as a feature flag (``if torch is not None``) for GPU reconstruction.
svmbir : module or None
    Imported lazily and only if available.
ndimage_shift : callable
    GPU-aware drop-in for ``scipy.ndimage.shift``.
gaussian_filter : callable
    GPU-aware drop-in for ``scipy.ndimage.gaussian_filter``.
fourier_shift : callable
    GPU-aware drop-in for ``scipy.ndimage.fourier_shift``.
to_numpy(arr) -> np.ndarray
    Convert an ``xp`` array (numpy or cupy) to a numpy array without copying
    when already numpy.
"""

import os
import multiprocessing as _mp

# Must run before tomopy/numexpr import. Login nodes expose many cores; tomopy
# asks numexpr for >NUMEXPR_MAX_THREADS and aborts. Bump the ceiling.
os.environ["NUMEXPR_MAX_THREADS"] = str(max(
    _mp.cpu_count(),
    int(os.environ.get("NUMEXPR_MAX_THREADS", "0") or 0),
    64,
))

import numpy as np
import scipy.ndimage as _sp_ndimage


def _probe_torch():
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.zeros(1, device='cuda')  # confirm an allocation actually succeeds
            return _t
        if _t.backends.mps.is_available():
            return _t
    except Exception:
        pass
    return None


def _probe_cupy():
    try:
        import cupy as _cp
        _cp.array([1])  # real allocation — raises if GPU is unavailable or busy
        return _cp
    except Exception:
        return None


def _probe_svmbir():
    try:
        import svmbir as _s
        return _s
    except ImportError:
        return None


torch = _probe_torch()
cp = _probe_cupy()
svmbir = _probe_svmbir()

if cp is not None:
    from cupyx.scipy.ndimage import (
        shift as ndimage_shift,
        gaussian_filter as gaussian_filter,
        fourier_shift as fourier_shift,
    )
    xp = cp
else:
    ndimage_shift = _sp_ndimage.shift
    gaussian_filter = _sp_ndimage.gaussian_filter
    fourier_shift = _sp_ndimage.fourier_shift
    xp = np


def to_numpy(arr):
    """Return ``arr`` as a numpy array. No copy if it already is one."""
    if isinstance(arr, np.ndarray):
        return arr
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


# One-line backend banner so callers don't repeat the print
_parts = []
_parts.append("cupy" if cp is not None else "numpy")
if torch is not None:
    _parts.append("torch-GPU")
if svmbir is not None:
    _parts.append("svmbir")
print(f"[gpu] active backends: {', '.join(_parts)}")
