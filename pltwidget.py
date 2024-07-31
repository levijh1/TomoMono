import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np




def runwidget(m):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(m[0], vmin=np.min(m), vmax=np.max(m), cmap='gray')
    plt.title("Frame 0")

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.1,0.1, 0.8, 0.05], facecolor='teal')

    def update_line(indx):
        ax.clear()
        ax.imshow(m[indx], vmin=np.min(m), vmax=np.max(m), cmap='gray')
        plt.title(f"Frame {indx}")
        plt.draw()

    slider = Slider(ax_slider, "Height (cross-section)", valmin=0, valmax=m.shape[0]-1, valinit=20, valstep = 1)
    slider.on_changed(update_line)

    plt.show()
