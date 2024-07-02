import numpy as np
import matplotlib.pyplot as plt
import tomopy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import time


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

class MoviePlotter:
    """Plots a sequence of images as a movie in a Jupyter Notebook with interactive controls using widgets.Play."""
    def __init__(self, x):
        self.x = x  # (M, N, N) array where M is the total number of images and N is the number of pixels in x and y
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
        """Update the plot based on slider/frame change."""
        self.update_plot(change['new'])
 
    def update_plot(self, frame_index):
        """Update the plot with the current frame."""
        with self.output:
            clear_output(wait=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(self.x[frame_index].astype(float), cmap='gray')  # Adjust colormap as needed
            plt.axis('off')
            # plt.title(f'Run {self.runNumbers[frame_index]}, Angle: {self.angleNumbers[frame_index]}')
            plt.show()

# # Fourier Transform Cross Correlation Method
# def XCA(m):
#     #Align first projection
#     A = np.sum(m[0, :, :])
#     coa = np.array([0, 0])
#     for i in range(m[0].shape[0]):
#       for j in range(m[0].shape[1]):
#         coa = coa + m[0, j, i]*np.array([j,i])
#     coa = coa / A
#
#     yshift0 = int(round(-(m[0].shape[0] / 2 - coa[0] - 1)))
#     xshift0 = int(round(-(m[0].shape[1] / 2 - coa[1])))
#     m[0] = np.roll(m[0, :, :], -xshift0, axis=1)
#     m[0] = np.roll(m[0, :, :], -yshift0, axis=0)
#
#     ###Align all other projections to first projection
#     for k in range(numangles-1):
#       # Homemade cross correlation method
#       CC = IFFT2(FFT2(m[k]) * FFT2(m[k + 1]))
#       maxpoint = np.where(CC == CC.max())
#       xshift = int(-(reconsize / 2 - maxpoint[1] - 1)[0])
#       yshift = int(-(imagesize / 2 - maxpoint[0] - 1)[0])
#
#       # ## Premade Scipy cross correlation method
#       # CC = sp.signal.correlate(m[k], m[k+1], mode='same', method='fft')
#       # maxpoint = np.where(CC == CC.max())
#       # xshift = int(reconsize / 2 - maxpoint[1])
#       # yshift = int(imagesize / 2 - maxpoint[0])
#
#       m[k + 1, :, :] = np.roll(m[k + 1, :, :], -xshift, axis=1)
#       m[k + 1, :, :] = np.roll(m[k + 1, :, :], -yshift, axis=0)
#     return m
#
#
# ##Multiplication and Summation Cross Correlation Method
# for i in range(numangles):
#   prj_integrated[i] = np.sum(prj_shifted[i], axis=0)
#
# for k in range(numangles):
#   sums = []
#   for i in range(-184,184):
#     multiplied_function = []
#     for j in range(184):
#       if j+i < 0 or j+i > (184-1):
#         multiplied_function.append(0)
#       elif k == numangles-1:
#         multiplied_function.append(prj_integrated[k,j] * prj_integrated[0,j+i])
#       else:
#         multiplied_function.append(prj_integrated[k,j] * prj_integrated[k+1,j+i])
#     sums.append(sum(multiplied_function))
#
#   shiftnumber = -(sums.index(max(sums))-184)
#   if k != numangles - 1:
#     prj_integrated[k+1] = np.roll(prj_integrated[k + 1], shiftnumber)
#     prj_aligned[k+1, :, :] = np.roll(prj_aligned[k + 1, :, :], shiftnumber, axis=1)