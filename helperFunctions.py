import numpy as np
import matplotlib.pyplot as plt
import tomopy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import time
import sys
from matplotlib.widgets import Slider
import tifffile
try:
    import cupy as cp
    cp.array([1])  # real allocation — raises if GPU is unavailable or busy
    from cupyx.scipy.ndimage import fourier_shift
    xp = cp
except Exception:
    cp = None
    from scipy.ndimage import fourier_shift
    xp = np





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

def subpixel_shift(image, shift_y, shift_x, homemade=False):
    """
    Shift a 2D image by (shift_y, shift_x) pixels using a Fourier phase ramp.

    Pads top/left/right with zeros and bottom with reflection before shifting,
    then crops back. The padding tapers linearly to zero at its outer edge to
    prevent Gibbs ringing.
    """
    rows, cols = image.shape
    pad_y = int(np.ceil(abs(shift_y))) + 2
    pad_x = int(np.ceil(abs(shift_x))) + 2

    # Bottom: reflect; top/left/right: zeros
    padded_image = np.pad(image, ((0, pad_y), (0, 0)), mode='reflect')
    padded_image = np.pad(padded_image, ((pad_y, 0), (pad_x, pad_x)), mode='constant', constant_values=0)

    ph, pw = padded_image.shape

    # Separable taper: 1 over the image region, linearly decays to 0 at outer padding edge
    wy = np.ones(ph)
    wy[:pad_y]      = np.linspace(0, 1, pad_y + 1)[:-1]
    wy[pad_y+rows:] = np.linspace(1, 0, pad_y + 1)[1:]

    wx = np.ones(pw)
    wx[:pad_x]      = np.linspace(0, 1, pad_x + 1)[:-1]
    wx[pad_x+cols:] = np.linspace(1, 0, pad_x + 1)[1:]

    padded_image *= wy[:, None] * wx[None, :]

    arr = xp.asarray(padded_image)
    fft_image = xp.fft.fft2(arr)

    if homemade:
        u = xp.fft.fftfreq(pw)
        v = xp.fft.fftfreq(ph)
        U, V = xp.meshgrid(u, v)
        phase_ramp = xp.exp(-2j * np.pi * (shift_x * U + shift_y * V))
        shifted_fft_image = fft_image * phase_ramp
    else:
        shifted_fft_image = fourier_shift(fft_image, (shift_y, shift_x))

    shifted_padded = xp.fft.ifft2(shifted_fft_image).real
    if xp is not np:
        shifted_padded = shifted_padded.get()

    return shifted_padded[pad_y:pad_y+rows, pad_x:pad_x+cols]

class MoviePlotter:
    """Plots a sequence of images as a movie in a Jupyter Notebook with interactive controls using widgets.Play."""
    def __init__(self, x, trust_box=None):
        self.x = x  # (M, N, N) array where M is the total number of images and N is the number of pixels in x and y
        self.trust_box = trust_box  # (top, bottom, left, right) pixel margins, or None
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
            plt.imshow(self.x[frame], vmin=self.global_min, vmax=self.global_max)
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
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(m[0], vmin=np.min(m), vmax=np.max(m), cmap='gray')
    plt.title("Frame 0")

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.1,0.1, 0.8, 0.05], facecolor='teal')

    def update_line(indx):
        ax.clear()

        # zplot = m[indx].T
        # nem = 100
        # q = np.quantile(zplot[nem:-nem, nem:-nem], [0.01,0.99])
        # plt.imshow(m[indx], cmap='bone', vmin=q[0], vmax=q[1])
        
        ax.imshow(m[indx], vmin=np.min(m), vmax=np.max(m), cmap='gray')
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
