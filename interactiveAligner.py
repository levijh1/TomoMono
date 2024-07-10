# import numpy as np
# import matplotlib.pyplot as plt
# import tomopy

# def display_sinogram(sinogram):
#     """
#     Display the sinogram using matplotlib.
    
#     Parameters:
#     - sinogram: 2D numpy array representing the sinogram to display.
#     """
#     plt.imshow(sinogram, cmap='gray')
#     plt.title('Sinogram')
#     plt.xlabel('Projection Number')
#     plt.ylabel('Pixel Position')
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tomopy

def generate_sinogram(data, slice_index):
    """
    Generate a sinogram from a 3D numpy array for a given slice index.
    
    Parameters:
    - data: 3D numpy array representing the radiographic dataset.
    - slice_index: Index of the slice to generate the sinogram from.
    
    Returns:
    - 2D numpy array representing the sinogram.
    """
    # Extract the slice across all projections
    # Assuming the second dimension is the slice dimension
    sinogram = data[:, slice_index, :]
    return sinogram

def display_sinogram(sinogram):
    """
    Display the sinogram with adjustable sinusoid overlay using matplotlib.
    
    Parameters:
    - sinogram: 2D numpy array representing the sinogram to display.
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Display the sinogram
    img = ax.imshow(sinogram, cmap='gray')
    ax.set_title('Sinogram with Sinusoid Overlay')
    ax.set_xlabel('Projection Number')
    ax.set_ylabel('Pixel Position')
    
    # Initial sinusoid parameters
    freq_init = 0.1  # initial frequency
    phase_init = 0  # initial phase
    
    # Plot initial sinusoid
    x = np.arange(sinogram.shape[1])
    y = (np.sin(x * freq_init + phase_init) + 1) * sinogram.shape[0] / 2
    sinusoid, = ax.plot(x, y, 'r-')
    
    # Add sliders for frequency and phase adjustment
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axphase = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    sfreq = Slider(axfreq, 'Freq', 0.01, 1.0, valinit=freq_init)
    sphase = Slider(axphase, 'Phase', 0, 2*np.pi, valinit=phase_init)
    
    # Update function for the sinusoid
    def update(val):
        freq = sfreq.val
        phase = sphase.val
        y = (np.sin(x * freq + phase) + 1) * sinogram.shape[0] / 2
        sinusoid.set_ydata(y)
        fig.canvas.draw_idle()
    
    # Register the update function with each slider
    sfreq.on_changed(update)
    sphase.on_changed(update)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    numAngles = 400
    shepp3d = tomopy.shepp3d(size=256)
    ang = tomopy.angles(nang=numAngles, ang1=0, ang2=360)
    data = tomopy.project(shepp3d, ang, pad=False)
    
    # Generate and display the sinogram for the middle slice
    middle_slice_index = data.shape[1] // 2
    sinogram = generate_sinogram(data, middle_slice_index)
    display_sinogram(sinogram)