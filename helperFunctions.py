import numpy as np
import matplotlib.pyplot as plt
import tomopy
from IPython.display import display, clear_output
import ipywidgets as widgets
import time
import sys
from matplotlib.widgets import Slider
import tifffile
try:
    import cupy as cp
    cp.array([1])  # real allocation — raises if GPU is unavailable or busy
    from cupyx.scipy.ndimage import fourier_shift, gaussian_filter as _gpu_gf
    xp = cp
except Exception:
    cp = None
    from scipy.ndimage import fourier_shift
    _gpu_gf = None
    xp = np

from scipy.ndimage import gaussian_filter as _cpu_gf





def FFT2(input):
  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input)))

def FFT(input):
  return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(input)))

def IFFT2(input):
  return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input)))

def IFFT(input):
  return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(input)))

def show(matrix, title):
  plt.imshow(matrix)
  plt.title(title)
  plt.show()

def add_noise(m):
  m = tomopy.prep.alignment.add_noise(m)
  return m

# def gpu_available():
#     """Returns True if CuPy/CUDA GPU acceleration is available on this device."""
#     return cp is not None


# def _gf(arr, sigma, truncate=2):
#     """Gaussian filter dispatched to GPU (cupyx) or CPU (scipy) based on xp."""
#     if xp is not np:
#         return _gpu_gf(arr, sigma=sigma, truncate=truncate)
#     return _cpu_gf(arr, sigma=sigma, truncate=truncate)


# def fadeoutImage(img, fadeMethod='rectangle', fadeToVal=None, transitionLength=None,
#                  ellipseSize=None, numSegments=1, angularOffsetSegments=0,
#                  windowShift=None):
#     """
#     Fade the edges of an image to a constant value using a smooth window.

#     GPU-accelerated via CuPy when available; falls back to NumPy/SciPy otherwise.

#     Parameters
#     ----------
#     img : array_like [N, M]
#     fadeMethod : {'ellipse', 'rectangle'}
#     fadeToVal : float, optional
#         Target fade value. Defaults to the mean of the boundary ring.
#     transitionLength : [ty, tx], optional
#         Transition width in pixels. Defaults to mean(shape) / 8.
#     ellipseSize : [ry, rx], optional
#         Fractional mask size in each axis (0–1). Default [0.8, 0.8].
#     numSegments : int
#         Angular segments for spatially-varying fadeout. Default 1.
#     angularOffsetSegments : float
#         Segment boundary angular offset in radians. Default 0.
#     windowShift : [dy, dx], optional
#         Shift of the fade window centre in pixels. Default [0, 0].

#     Returns
#     -------
#     imgFaded : np.ndarray [N, M], float32
#     transitionMask : np.ndarray [N, M], float32
#     """
#     if fadeMethod not in ('ellipse', 'rectangle'):
#         raise ValueError(f"fadeMethod must be 'ellipse' or 'rectangle', got '{fadeMethod}'")

#     img = np.asarray(img, dtype=np.float32)
#     rows, cols = img.shape

#     if transitionLength is None:
#         tl = int(np.ceil(np.mean([rows, cols]) / 8))
#         transitionLength = [tl, tl]
#     if windowShift is None:
#         windowShift = [0, 0]
#     if ellipseSize is None:
#         ellipseSize = [0.8, 0.8]

#     tly, tlx = float(transitionLength[0]), float(transitionLength[1])
#     wsy, wsx = float(windowShift[0]), float(windowShift[1])

#     arr = xp.asarray(img)

#     # Coordinate grids centred on the image
#     Y_vec = xp.arange(-rows / 2 + 0.5, rows / 2, 1.0, dtype=xp.float32)
#     X_vec = xp.arange(-cols / 2 + 0.5, cols / 2, 1.0, dtype=xp.float32)
#     Xg, Yg = xp.meshgrid(X_vec, Y_vec)
#     Xg_s = Xg - wsx   # shifted grids for mask geometry
#     Yg_s = Yg + wsy

#     # --- Outer mask, inner mask, boundary ring ---
#     if fadeMethod == 'ellipse':
#         ry = (ellipseSize[0] * rows - 2) / 2
#         rx = (ellipseSize[1] * cols - 2) / 2
#         outer = Xg_s ** 2 / rx ** 2 + Yg_s ** 2 / ry ** 2 < 1
#         ryI = max(1.0, ry - tly)
#         rxI = max(1.0, rx - tlx)
#         inner = Xg_s ** 2 / rxI ** 2 + Yg_s ** 2 / ryI ** 2 < 1
#         idxBoundary = outer ^ inner
#         transitionMask = outer.astype(xp.float32)

#     else:  # rectangle
#         ry = int(np.ceil(ellipseSize[0] * rows))
#         rx = int(np.ceil(ellipseSize[1] * cols))
#         outer = xp.zeros((rows, cols), dtype=xp.float32)
#         y0 = max(0, (rows - ry) // 2)
#         x0 = max(0, (cols - rx) // 2)
#         outer[y0:y0 + min(ry, rows - y0), x0:x0 + min(rx, cols - x0)] = 1.0
#         outer = xp.roll(outer, (-int(wsy), int(wsx)), axis=(0, 1))

#         ryI = max(1, int(ry - tly))
#         rxI = max(1, int(rx - tlx))
#         inner = xp.zeros((rows, cols), dtype=xp.float32)
#         y0i = max(0, (rows - ryI) // 2)
#         x0i = max(0, (cols - rxI) // 2)
#         inner[y0i:y0i + min(ryI, rows - y0i), x0i:x0i + min(rxI, cols - x0i)] = 1.0
#         inner = xp.roll(inner, (-int(wsy), int(wsx)), axis=(0, 1))

#         idxBoundary = (outer > 0.5) ^ (inner > 0.5)
#         transitionMask = outer

#     # --- Determine fadeToVals ---
#     if fadeToVal is not None:
#         fadeToVals = xp.float32(fadeToVal)

#     elif numSegments > 1:
#         # Gaussian pre-filter for stable boundary estimation
#         imFilt = _gf(arr, sigma=10.0 / 2.35, truncate=2)

#         theta = xp.arctan2(Yg - wsy, Xg - wsx)
#         theta = xp.mod(theta + xp.float32(angularOffsetSegments), xp.float32(2 * np.pi))
#         segIdx = xp.minimum(
#             (xp.floor(numSegments / (2 * np.pi) * theta) + 1).astype(xp.int32),
#             xp.int32(numSegments))
#         # Mirror across Y-axis for left-right symmetry in the fadeout values
#         mirTheta = xp.mod(xp.float32(np.pi) - theta, xp.float32(2 * np.pi))
#         mirSegIdx = xp.minimum(
#             (xp.floor(numSegments / (2 * np.pi) * mirTheta) + 1).astype(xp.int32),
#             xp.int32(numSegments))

#         fadeToVals = xp.zeros_like(arr)
#         for seg in range(1, numSegments + 1):
#             combined = idxBoundary & ((segIdx == seg) | (mirSegIdx == seg))
#             val = float(imFilt[combined].mean()) if bool(combined.any()) else float(arr.mean())
#             fadeToVals[segIdx == seg] = xp.float32(val)

#         fadeToVals = _gf(fadeToVals, sigma=(tly / 2.35, tlx / 2.35), truncate=2)
#         fadeToVals = xp.asarray(fadeToVals)

#     else:
#         val = float(arr[idxBoundary].mean()) if bool(idxBoundary.any()) else float(arr.mean())
#         fadeToVals = xp.float32(val)

#     # --- Smooth the transition mask with a Gaussian (standard holography approach) ---
#     if tly > 1 and tlx > 1:
#         transitionMask = _gf(transitionMask.astype(xp.float32), sigma=(tly / 2.35, tlx / 2.35), truncate=2)
#         transitionMask = xp.asarray(transitionMask)

#     # --- Apply fadeout ---
#     imgFaded = arr * transitionMask + fadeToVals * (1 - transitionMask)

#     imgFaded = xp.where(xp.isnan(imgFaded) | xp.isinf(imgFaded), xp.float32(1.0), imgFaded)

#     if xp is not np:
#         return imgFaded.get().astype(np.float32), transitionMask.get().astype(np.float32)
#     return imgFaded.astype(np.float32), transitionMask.astype(np.float32)


def _subpixel_shift_batch(images, shift_y, shift_x):
    """
    Vectorized batch version of subpixel_shift for a (N, H, W) array.

    shift_y and shift_x must be 1-D arrays of length N. A single symmetric
    pad is computed from the maximum absolute shift across all images, then
    all N phase ramps are applied in one batched FFT. GPU-accelerated via
    CuPy when available.
    """
    shift_y = np.asarray(shift_y, dtype=np.float64)
    shift_x = np.asarray(shift_x, dtype=np.float64)
    n, rows, cols = images.shape

    min_pad = 10
    max_abs_y = float(np.max(np.abs(shift_y))) if n > 0 else 0.0
    max_abs_x = float(np.max(np.abs(shift_x))) if n > 0 else 0.0
    pad_y = max(min_pad, int(np.ceil(max_abs_y)) * 4 + 1)
    pad_x = max(min_pad, int(np.ceil(max_abs_x)) * 4 + 1)

    padded = np.pad(images, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='edge')
    ph, pw = padded.shape[1], padded.shape[2]

    # Symmetric cosine taper on all four edges — uniform across the batch
    wy = np.ones(ph, dtype=np.float32)
    wy[:pad_y] = np.sin(0.5 * np.pi * np.arange(pad_y, dtype=np.float32) / pad_y) ** 2
    wy[ph - pad_y:] = wy[:pad_y][::-1]
    wx = np.ones(pw, dtype=np.float32)
    wx[:pad_x] = np.sin(0.5 * np.pi * np.arange(pad_x, dtype=np.float32) / pad_x) ** 2
    wx[pw - pad_x:] = wx[:pad_x][::-1]
    window = wy[:, None] * wx[None, :]  # (ph, pw)

    padded = padded.astype(np.float32)
    cs = max(4, min(32, rows // 6, cols // 6))
    bg_means = (0.5 * (
        images[:, :cs, :cs].mean(axis=(1, 2)) +
        images[:, :cs, -cs:].mean(axis=(1, 2))
    )).astype(np.float32)  # (N,)

    padded -= bg_means[:, None, None]
    padded *= window[None, :, :]
    padded += bg_means[:, None, None]

    arr = xp.asarray(padded)
    F = xp.fft.fft2(arr)  # (N, ph, pw)
    fy = xp.fft.fftfreq(ph).reshape(1, -1, 1)
    fx = xp.fft.fftfreq(pw).reshape(1, 1, -1)
    sy = xp.asarray(shift_y).reshape(-1, 1, 1)
    sx = xp.asarray(shift_x).reshape(-1, 1, 1)
    phase = xp.exp(-2j * np.pi * (sy * fy + sx * fx))
    shifted = xp.fft.ifft2(F * phase).real
    if xp is not np:
        shifted = shifted.get()
    return np.asarray(shifted[:, pad_y:pad_y + rows, pad_x:pad_x + cols], dtype=images.dtype)


def subpixel_shift(image, shift_y, shift_x):
    """
    Shift a 2D image by (shift_y, shift_x) pixels using a Fourier phase ramp.

    When image is a 3D array (N, H, W), shift_y and shift_x must be 1-D
    arrays of length N. All shifts are applied in a single batched FFT via
    _subpixel_shift_batch.

    Pads by replicating edge values outward, then applies a cosine taper over the
    padded strip toward the estimated background value (mean of the two upper corner patches).
    The top taper spans 1/4 of the pad distance; the bottom taper is 4x the pad distance.
    GPU-accelerated via CuPy when available.
    """
    if image.ndim == 3:
        return _subpixel_shift_batch(image, shift_y, shift_x)

    rows, cols = image.shape

    min_pad = 10  # <-- NEW: enforce minimum padding

    # Only pad the side each shift component exposes, but enforce minimum padding
    pad_top    = max(min_pad, int(np.ceil( shift_y)) + 1) if shift_y > 0 else min_pad
    pad_bottom = max(min_pad, int(np.ceil(-shift_y))*4 + 1) if shift_y < 0 else min_pad
    pad_left   = max(min_pad, int(np.ceil( shift_x)) + 1) if shift_x > 0 else min_pad
    pad_right  = max(min_pad, int(np.ceil(-shift_x)) + 1) if shift_x < 0 else min_pad

    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    ph, pw = padded.shape

    cs = max(4, min(32, rows // 6, cols // 6))
    bg_mean = np.float32(np.mean([image[:cs, :cs], image[:cs, -cs:]]))

    def _taper_1d(size, pad_start, pad_end):
        t = np.ones(size, dtype=np.float32)
        if pad_start > 0:
            t[:pad_start] = np.sin(0.5 * np.pi * np.arange(pad_start, dtype=np.float32) / pad_start)**2
        if pad_end > 0:
            t[size - pad_end:] = (np.sin(0.5 * np.pi * np.arange(pad_end, dtype=np.float32) / pad_end)**2)[::-1]
        return t

    # y window: top uses inner-edge ascending taper (outer wall = 0), bottom uses outer-edge taper
    wy = np.ones(ph, dtype=np.float32)
    if pad_top > 0:
        taper_len_top = max(1, int(np.ceil(shift_y / 6))) if shift_y > 0 else min_pad
        taper_len_top = min(taper_len_top, pad_top)
        wy[:pad_top - taper_len_top] = 0.0
        wy[pad_top - taper_len_top:pad_top] = np.sin(
            0.5 * np.pi * np.arange(taper_len_top, dtype=np.float32) / taper_len_top)**2

    if pad_bottom > 0:
        wy[ph - pad_bottom:] = (np.sin(
            0.5 * np.pi * np.arange(pad_bottom, dtype=np.float32) / pad_bottom)**2)[::-1]

    window = wy[:, None] * _taper_1d(pw, pad_left, pad_right)[None, :]

    padded = np.asarray(padded, dtype=np.float32)
    padded -= bg_mean
    padded *= window
    padded += bg_mean

    arr = xp.asarray(padded)
    shifted_fft = fourier_shift(xp.fft.fft2(arr), (shift_y, shift_x))
    shifted_padded = xp.fft.ifft2(shifted_fft).real
    if xp is not np:
        shifted_padded = shifted_padded.get()

    # Crop back to original size (this still works correctly with larger padding)
    return np.asarray(
        shifted_padded[pad_top:pad_top + rows, pad_left:pad_left + cols]
    )

class MoviePlotter:
    """Plots a sequence of images as a movie in a Jupyter Notebook with interactive controls using widgets.Play."""
    def __init__(self, x, trust_box=None, color='gray'):
        self.x = x  # (M, N, N) array where M is the total number of images and N is the number of pixels in x and y
        self.trust_box = trust_box  # (top, bottom, left, right) pixel margins, or None
        self.color = color
        self.global_min = np.min(x)
        self.global_max = np.max(x)
        self.play = widgets.Play(
            value=0,
            min=0,
            max=len(x) - 1,
            step=1,
            description="Press play",
            interval=500,
            disabled=False
        )
        self.slider = widgets.IntSlider(min=0, max=len(x) - 1, value=0, step=1, description='Frame')
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        self.slider.observe(self.slider_update, names='value')
        self.output = widgets.Output()
        display(widgets.HBox([self.play, self.slider]))
        display(self.output)
        self.update_plot(0)

    def slider_update(self, change):
        self.update_plot(change['new'])

    def update_plot(self, frame):
        with self.output:
            self.output.clear_output(wait=True)
            plt.imshow(self.x[frame], vmin=self.global_min, vmax=self.global_max, cmap=self.color)
            ax = plt.gca()  # capture before colorbar changes current axes
            plt.colorbar(label='Intensity')
            plt.title(f"Frame {frame}")
            if self.trust_box is not None:
                top, bottom, left, right = self.trust_box
                ny, nx = self.x.shape[1], self.x.shape[2]
                kw = dict(color='red', linewidth=1.5, linestyle='--')
                if top > 0:    ax.axhline(int(np.ceil(top)) - 0.5,        **kw)
                if bottom > 0: ax.axhline(ny - int(np.ceil(bottom)) - 0.5, **kw)
                if left > 0:   ax.axvline(int(np.ceil(left)) - 0.5,        **kw)
                if right > 0:  ax.axvline(nx - int(np.ceil(right)) - 0.5,  **kw)
            plt.show()

### Matplotlib widget
def runwidget(m):
    """Makes a movie of a list of images (3D array) that is good for running in a script"""
    vmin, vmax = np.min(m), np.max(m)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(m[0], vmin=vmin, vmax=vmax, cmap='gray')
    plt.title("Frame 0")

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.1,0.1, 0.8, 0.05], facecolor='teal')

    def update_line(indx):
        ax.clear()

        # zplot = m[indx].T
        # nem = 100
        # q = np.quantile(zplot[nem:-nem, nem:-nem], [0.01,0.99])
        # plt.imshow(m[indx], cmap='bone', vmin=q[0], vmax=q[1])

        ax.imshow(m[indx], vmin=vmin, vmax=vmax, cmap='gray')
        plt.title(f"Frame {indx}")
        plt.draw()

    slider = Slider(ax_slider, "Height (cross-section)", valmin=0, valmax=m.shape[0]-1, valinit=20, valstep = 1)
    slider.on_changed(update_line)

    plt.show()


# Custom logger class
class DualLogger:
    def __init__(self, filepath, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()


###Helper Functions for using tiff files
def convert_to_numpy(file_location):
    """
    Converts a TIFF file to a numpy array, extracting scale information if available.
    
    Parameters:
    - file_location: Path to the TIFF file.
    
    Returns:
    - A tuple containing the numpy array and a dictionary with scale information (or None if not available).
    """
    with tifffile.TiffFile(file_location) as tif:
        tif_data = tif.asarray()
        
        # Extract scale information from metadata, if available
        try:
            xres = tif.pages[0].tags['XResolution'].value
            yres = tif.pages[0].tags['YResolution'].value
            unit = tif.pages[0].tags['ResolutionUnit'].value
            scale_info = {'XResolution': xres, 'YResolution': yres, 'Unit': unit}
        except KeyError:
            scale_info = None  # No scale information found

    # Adjust dimensions to ensure tif_data is 3D
    if tif_data.ndim == 2:
        tif_data = np.expand_dims(tif_data, axis=0)
    elif tif_data.ndim > 3:
        raise ValueError("Unsupported TIFF dimensions: expected 2D or 3D, got higher.")
    # print(tif_data.dtype)
    return tif_data, scale_info


def convert_to_tiff(numpy_data, file_location, scale_info=None):
    """
    Saves a 3D numpy array as a TIFF file, including scale information if provided.
    
    Parameters:
    - numpy_data: 3D numpy array to be saved.
    - file_location: Path and filename for the output TIFF file.
    - scale_info: Optional dictionary containing 'XResolution', 'YResolution', and 'Unit'.
                  If provided, these values are used to set the resolution of the TIFF file.
    """
    # if numpy_data.ndim != 3:
    #     raise ValueError("Input array must be 3D.")

    # Convert scale information to appropriate units for TIFF metadata
    if scale_info:
        xres = scale_info.get('XResolution', 1)  # Default resolution is 1 if not specified
        yres = scale_info.get('YResolution', 1)
        unit = scale_info.get('Unit', 'MICRON')  # Default unit is inches

        # Save the numpy array as a TIFF with resolution information
        tifffile.imsave(file_location, numpy_data, resolution=(xres, yres, unit))
    else:
        # Save without resolution information if not provided
        tifffile.imsave(file_location, numpy_data)


def convert_to_2Dtiff(numpy_data, file_location, scale_info=None):
    """
    Saves a 2D numpy array as a TIFF file, including scale information if provided.
    
    Parameters:
    - numpy_data: 2D numpy array to be saved.
    - file_location: Path and filename for the output TIFF file.
    - scale_info: Optional dictionary containing 'XResolution', 'YResolution', and 'Unit'.
                  If provided, these values are used to set the resolution of the TIFF file.
    """
    if numpy_data.ndim != 2:
        raise ValueError("Input array must be 2D.")

    numpy_data = np.array([numpy_data])

    # Convert scale information to appropriate units for TIFF metadata
    if scale_info:
        xres = scale_info.get('XResolution', 1)  # Default resolution is 1 if not specified
        yres = scale_info.get('YResolution', 1)
        unit = scale_info.get('Unit', 'MICRON')  # Default unit is inches

        # Save the numpy array as a TIFF with resolution information
        tifffile.imsave(file_location, numpy_data, resolution=(xres, yres, unit))
    else:
        # Save without resolution information if not provided
        tifffile.imsave(file_location, numpy_data)

def degree_to_positiveRadians(angles):
    angles_radians = np.deg2rad(angles)
    return (angles_radians + 2*np.pi) % (2*np.pi)