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

def subpixel_shift(image, shift_y, shift_x):
    """
    Shift a 2D image using a phase ramp in the Fourier domain and pad out-of-bounds regions with zeros (if that part of the code is uncommented).

    Parameters:
    image (2D numpy array): The input image to be shifted.
    shift_x (float): The shift in the x direction (horizontal shift).
    shift_y (float): The shift in the y direction (vertical shift).

    Returns:
    shifted_image (2D numpy array): The shifted image with out-of-bounds areas padded with zeros.
    """
    # Fourier transform of the image
    fft_image = np.fft.fft2(image)

    # Get the image dimensions
    rows, cols = image.shape

    # Create frequency coordinate grids
    u = np.fft.fftfreq(cols)  # Frequency coordinates along the x-axis
    v = np.fft.fftfreq(rows)  # Frequency coordinates along the y-axis
    U, V = np.meshgrid(u, v)  # 2D grid of frequency coordinates

    # Calculate the phase ramp for shifting
    phase_ramp = np.exp(-2j * np.pi * (shift_x * U + shift_y * V))

    # Apply the phase ramp to the Fourier-transformed image
    shifted_fft_image = fft_image * phase_ramp

    # Inverse Fourier transform to get the shifted image
    shifted_image = np.fft.ifft2(shifted_fft_image).real


    
    # ##Uncomment this code if you want the shift to not wrap around the image and just be zeros at the edge instead
    # # Create a mask of valid regions
    # mask = np.ones_like(image)

    # # Calculate how much to pad with zeros based on shift
    # pad_x_left = int(np.ceil(shift_x)) if shift_x > 0 else 0
    # pad_x_right = int(np.ceil(-shift_x)) if shift_x < 0 else 0
    # pad_y_top = int(np.ceil(shift_y)) if shift_y > 0 else 0
    # pad_y_bottom = int(np.ceil(-shift_y)) if shift_y < 0 else 0

    # # Apply zero padding based on shifts
    # if pad_y_top != 0:
    #     mask[:pad_y_top, :] = 0  # Top padding
    # if pad_y_bottom != 0:
    #     mask[-pad_y_bottom:, :] = 0  # Bottom padding
    # if pad_x_left != 0:
    #     mask[:, :pad_x_left] = 0  # Left padding
    # if pad_x_right != 0:
    #     mask[:, -pad_x_right:] = 0  # Right padding

    # # Apply the mask to the shifted image
    # shifted_image *= mask
    
    return shifted_image

class MoviePlotter:
    """Plots a sequence of images as a movie in a Jupyter Notebook with interactive controls using widgets.Play."""
    def __init__(self, x):
        self.x = x  # (M, N, N) array where M is the total number of images and N is the number of pixels in x and y
        # Calculate global min and max for color normalization
        self.global_min = np.min(x)
        self.global_max = np.max(x)
        # Play and Slider widgets
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
        # Link the slider and the play widget
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        # Slider changes update the plot
        self.slider.observe(self.slider_update, names='value')
        # Output for the plot
        self.output = widgets.Output()
        display(widgets.HBox([self.play, self.slider]))
        display(self.output)
        # Initial plot
        self.update_plot(0)  # Update initially with the first frame

    def slider_update(self, change):
        self.update_plot(change['new'])

    def update_plot(self, frame):
        with self.output:
            self.output.clear_output(wait=True)  # Clear the previous frame
            plt.imshow(self.x[frame], vmin=self.global_min, vmax=self.global_max)
            plt.colorbar(label='Intensity')
            plt.title(f"Frame {frame}")
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
