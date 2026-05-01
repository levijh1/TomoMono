import tomopy
import numpy as np
import random
import scipy as sp
from helperFunctions import MoviePlotter, subpixel_shift, runwidget
# import torch
from tqdm import tqdm
from scipy.ndimage import rotate
# import svmbir
from alignment_methods import *

try:
    import torch
    print("PyTorch imported successfully.")
    import svmbir
    print("SVMBIR imported successfully.")
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        torch = None
        svmbir = None
except ImportError:
    torch = None
    svmbir = None

try:
    import cupy as cp
    cp.array([1])  # real allocation — raises if GPU is unavailable or busy
    from cupyx.scipy.ndimage import shift as _ndimage_shift
    xp = cp
except Exception:
    cp = None
    _ndimage_shift = sp.ndimage.shift
    xp = np

class tomoData:
    """
    Class for handling tomographic data, including preprocessing, alignment, and reconstruction.
    """

    def __init__(self, data, angles = None):
        """
        Initializes the TomoData object with the provided dataset.
        Make sure that you check that the ang variable has the same range and number of angles as the collected data.
        This assumes that all angles are evenly spaced between 0 and 360 degrees.
        Parameters:
        - data (np.array): The tomographic data as a 3D numpy array. The first dimension is the projection number.
        - angles (list): Make sure the angles are listed in positive radians (0 to 2*pi)
        """
        self.num_angles = data.shape[0]
        self.image_size = data.shape[1:]
        self.data = data
        self.workingProjections = np.copy(data)
        self.rotation_center = 0
        self.center_offset = 0
        self.finalProjections = np.copy(data)
        self.tracked_shifts = np.zeros((self.num_angles, 2))
        self.tracked_rotations = np.zeros(self.num_angles)
        self.finalReprojections = None
        self._shift_envelope = [0.0, 0.0, 0.0, 0.0]      # [top, bottom, left, right]
        self._shift_envelope_idx = [0, 0, 0, 0]            # projection index for each max shift
        if angles is None:
            self.ang = tomopy.angles(nang=self.num_angles, ang1=0, ang2=360)
        else:
            self.ang = angles

    def reset_workingProjections(self, x_size=900, y_size=650, cropBottomCenter=False):
        """
        Resets working and final projections to the original data and crops to the specified center size.
        This is useful for starting fresh with the original projections after modifications back to back.

        Parameters:
        - x_size (int): Width of the crop.
        - y_size (int): Height of the crop.
        """
        self.workingProjections = np.copy(self.data)
        self.finalProjections = np.copy(self.data)
        if x_size != None and y_size != None:
            if cropBottomCenter:
                self.crop_bottom_center(x_size, y_size)
            else:
                self.crop_center(x_size, y_size)

    def get_recon(self):
        """
        Returns the reconstructed 3D model.

        Returns:
        - np.array: The reconstructed volume.
        """
        return self.recon

    def get_workingprojections(self):
        """
        Returns the current state of working projections (The projections that are being modified).

        Returns:
        - np.array: The working projections.
        """
        return self.workingProjections

    def get_finalProjections(self):
        """
        Returns the final projections (the projections that are modified last and saved for reconstruction).

        Returns:
        - np.array: The final projections.
        """
        return self.finalProjections

    def jitter(self, maxShift=5):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        Jitter ranges from -maxShift to +maxShift pixels in both x and y directions. Affects data variable as well.

        Parameters:
        - maxShift (float): Maximum shift in pixels for both x and y directions.
        """
        multiplier = maxShift * 2
        for i in range(1, self.num_angles-1):
            x_shift = multiplier * (random.random() - 0.5)
            y_shift = multiplier * (random.random() - 0.5)
            arr = xp.asarray(self.data[i])
            shifted = _ndimage_shift(arr, (y_shift, x_shift), mode='reflect')
            self.data[i] = shifted.get() if xp is not np else shifted
        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()

    def jitter_y(self, maxShift=5):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        Jitter ranges from -maxShift to +maxShift pixels in both x and y directions. Affects data variable as well.

        Parameters:
        - maxShift (float): Maximum shift in pixels for both x and y directions.
        """
        multiplier = maxShift * 2
        for i in range(1, self.num_angles-1):
            x_shift = 0
            y_shift = multiplier * (random.random() - 0.5)
            arr = xp.asarray(self.data[i])
            shifted = _ndimage_shift(arr, (y_shift, x_shift), mode='reflect')
            self.data[i] = shifted.get() if xp is not np else shifted
        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()

    def add_noise(self):
        self.data = tomopy.prep.alignment.add_noise(self.data)
        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()   

    def crop_center(self, new_x, new_y):
        """
        Crops each 2D array in the 3D array to a specified size centered in the middle.

        Parameters:
        - new_x (int or None): Target width of the crop. If None, width is unchanged.
        - new_y (int or None): Target height of the crop. If None, height is unchanged.
        """
        y, x = self.workingProjections[0].shape
        if new_x is None:
            startx, endx = 0, x
        else:
            startx = x // 2 - new_x // 2
            endx = startx + new_x
            startx, endx = max(0, startx), min(x, endx)
        if new_y is None:
            starty, endy = 0, y
        else:
            starty = y // 2 - new_y // 2
            endy = starty + new_y
            starty, endy = max(0, starty), min(y, endy)
        self.workingProjections = self.workingProjections[:,starty:endy, startx:endx]
        self.finalProjections = self.finalProjections[:,starty:endy, startx:endx]
        self.image_size = self.workingProjections.shape[1:]
        removed_top, removed_bottom = starty, y - endy
        removed_left, removed_right = startx, x - endx
        self._shift_envelope[0] = max(0.0, self._shift_envelope[0] - removed_top)
        self._shift_envelope[1] = max(0.0, self._shift_envelope[1] - removed_bottom)
        self._shift_envelope[2] = max(0.0, self._shift_envelope[2] - removed_left)
        self._shift_envelope[3] = max(0.0, self._shift_envelope[3] - removed_right)

    def crop_bottom_center(self, new_y, new_x):
        """
        Crops each 2D array in the 3D array to a specified size, aligned to the bottom and centered horizontally.

        Parameters:
        - new_y (int or None): Target height of the crop. If None, height is unchanged.
        - new_x (int or None): Target width of the crop. If None, width is unchanged.
        """
        y, x = self.workingProjections[0].shape
        if new_x is None:
            startx, endx = 0, x
        else:
            startx = x // 2 - new_x // 2
            endx = startx + new_x
            startx, endx = max(0, startx), min(x, endx)
        if new_y is None:
            starty, endy = 0, y
        else:
            starty = y - new_y
            endy = y
            starty, endy = max(0, starty), min(y, endy)
        self.workingProjections = self.workingProjections[:, starty:endy, startx:endx]
        self.finalProjections = self.finalProjections[:, starty:endy, startx:endx]
        self.image_size = self.workingProjections.shape[1:]
        removed_top, removed_bottom = starty, y - endy
        removed_left, removed_right = startx, x - endx
        self._shift_envelope[0] = max(0.0, self._shift_envelope[0] - removed_top)
        self._shift_envelope[1] = max(0.0, self._shift_envelope[1] - removed_bottom)
        self._shift_envelope[2] = max(0.0, self._shift_envelope[2] - removed_left)
        self._shift_envelope[3] = max(0.0, self._shift_envelope[3] - removed_right)


    def track_shifts(self):
        """
        Resets tracked shifts and rotations, and sets final projections to the current working projections.
        Allows you to track all changes to workingProjections so that you can apply the final changes later to finalProjections.
        """
        self.finalProjections = self.workingProjections.copy()
        self.tracked_shifts = np.zeros((self.num_angles, 2))
        self.tracked_rotations = np.zeros(self.num_angles)

    def make_updates_shift(self):
        """
        Applies tracked subpixel shifts from workingProjections to the finalProjections and resets tracked shifts.
        This is done so that the finalProjections can be updated just once and not lose any information.
        """
        # Accumulate envelope and winning indices before zeroing tracked_shifts
        live = [
            max(0.0, float(self.tracked_shifts[:, 0].max())),
            max(0.0, float(-self.tracked_shifts[:, 0].min())),
            max(0.0, float(self.tracked_shifts[:, 1].max())),
            max(0.0, float(-self.tracked_shifts[:, 1].min())),
        ]
        live_idx = [
            int(np.argmax(self.tracked_shifts[:, 0])),
            int(np.argmin(self.tracked_shifts[:, 0])),
            int(np.argmax(self.tracked_shifts[:, 1])),
            int(np.argmin(self.tracked_shifts[:, 1])),
        ]
        for i in range(4):
            if live[i] > self._shift_envelope[i]:
                self._shift_envelope[i] = live[i]
                self._shift_envelope_idx[i] = live_idx[i]

        for m in tqdm(range(self.num_angles), desc='Apply shifts to final projections'):
            self.finalProjections[m] = subpixel_shift(self.finalProjections[m], self.tracked_shifts[m, 0], self.tracked_shifts[m, 1])
        self.tracked_shifts = np.zeros((self.num_angles, 2))

    def make_updates_rotate(self):
        """
        Applies tracked rotations to the final projections and resets tracked shifts.
        """
        for m in tqdm(range(self.num_angles), desc='Apply rotations to final projections'):
            self.finalProjections[m] = rotate(self.finalProjections[m], self.tracked_rotations[m], reshape=False, mode='constant')
        self.tracked_shifts = np.zeros((self.num_angles, 2))

    def normalize(self, isPhaseData):
        """
        Normalizes all projections to be positive values between 0 and 1.
        """
        print("\n")
        print("Normalizing projections")
        if isPhaseData:
            self.workingProjections = -self.workingProjections
        arr = xp.asarray(self.workingProjections)
        arr = (arr - xp.min(arr)) / (xp.max(arr) - xp.min(arr))
        self.workingProjections = arr.get() if xp is not np else arr
        # self.finalProjections = np.copy(self.workingProjections)

    def standardize(self, isPhaseData):
        """
        Standardizes projections to have zero mean and unit variance, then inverts the sign.

        Parameters:
        - isPhaseData (bool): If True, inverts the sign of the projections, which is often necessary for phase data since it can be negative.
        """
        arr = xp.asarray(self.workingProjections)
        arr = (arr - xp.mean(arr)) / xp.std(arr)
        if isPhaseData:
            arr *= -1
        self.workingProjections = arr.get() if xp is not np else arr

    def threshold(self, threshold=-0.1):
        """
        Thresholds the projections, setting values below the threshold to 1 and others to 0.

        Parameters:
        - threshold (float): Threshold value.
        """
        self.workingProjections = (self.workingProjections <= threshold).astype(float)

    @property
    def shift_envelope(self):
        """
        Returns (top, bottom, left, right) — the total number of pixels from each
        edge that have been exposed across all projections due to accumulated shifts,
        including shifts already committed via make_updates_shift. Pixels inside
        this box are unaffected by any shift.
        """
        return (
            max(self._shift_envelope[0], max(0.0, float(self.tracked_shifts[:, 0].max()))),
            max(self._shift_envelope[1], max(0.0, float(-self.tracked_shifts[:, 0].min()))),
            max(self._shift_envelope[2], max(0.0, float(self.tracked_shifts[:, 1].max()))),
            max(self._shift_envelope[3], max(0.0, float(-self.tracked_shifts[:, 1].min()))),
        )

    @property
    def shift_envelope_idx(self):
        """
        Returns (idx_top, idx_bottom, idx_left, idx_right) — projection indices responsible
        for the largest shift in each direction, merging stored history with live tracked_shifts.
        """
        live = [
            max(0.0, float(self.tracked_shifts[:, 0].max())),
            max(0.0, float(-self.tracked_shifts[:, 0].min())),
            max(0.0, float(self.tracked_shifts[:, 1].max())),
            max(0.0, float(-self.tracked_shifts[:, 1].min())),
        ]
        live_idx = [
            int(np.argmax(self.tracked_shifts[:, 0])),
            int(np.argmin(self.tracked_shifts[:, 0])),
            int(np.argmax(self.tracked_shifts[:, 1])),
            int(np.argmin(self.tracked_shifts[:, 1])),
        ]
        return tuple(
            live_idx[i] if live[i] > self._shift_envelope[i] else self._shift_envelope_idx[i]
            for i in range(4)
        )

    def makeNotebookProjMovie(self, show_trust_region=False):
        """
        Displays a movie of the final projections in a Jupyter notebook.
        If show_trust_region=True, overlays red dashed lines marking the box of
        pixels unaffected by shifting in any projection.
        Prints the projection indices responsible for the largest shift in each
        direction, if any shifts have been applied.
        """

        if show_trust_region:
            top, bottom, left, right = self.shift_envelope
            if top > 0 or bottom > 0 or left > 0 or right > 0:
                idx_top, idx_bottom, idx_left, idx_right = self.shift_envelope_idx
                print("Largest shifts per direction:")
                print(f"  Top    (down  {top:.2f} px) — projection {idx_top}")
                print(f"  Bottom (up    {bottom:.2f} px) — projection {idx_bottom}")
                print(f"  Left   (right {left:.2f} px) — projection {idx_left}")
                print(f"  Right  (left  {right:.2f} px) — projection {idx_right}")

                trust_box = self.shift_envelope
        else:
            trust_box = None
            
        MoviePlotter(self.finalProjections, trust_box=trust_box, color='twilight')

    def makeScriptProjMovie(self):
        """
        Displays a movie of the final projections in a script environment.
        """
        runwidget(self.finalProjections)

    def makeNotebookReconMovie(self):
        """
        Displays a movie of the reconstructed volume in a Jupyter notebook.
        """
        MoviePlotter(self.recon)

    def makeScriptReconMovie(self):
        """
        Displays a movie of the reconstructed volume in a script environment.
        """
        runwidget(self.recon)

    def displayReconOrthogonalSlices(self):
        import matplotlib.pyplot as plt
        recon = self.recon
        nz, ny, nx = recon.shape
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(recon[cz, :, :], cmap='gray', aspect='equal')
        axes[0].set_title(f'XY  (z={cz})')

        axes[1].imshow(recon[:, cy, :], cmap='gray', aspect='auto')
        axes[1].set_title(f'XZ  (y={cy})')

        axes[2].imshow(recon[:, :, cx], cmap='gray', aspect='auto')
        axes[2].set_title(f'YZ  (x={cx})')

        for ax in axes:
            ax.axis('off')

        plt.suptitle('Orthogonal slices through reconstruction')
        plt.tight_layout()
        plt.show()

    def bilateralFilter(self, *args, **kwargs):
        print("\n")
        return bilateralFilter(self, *args, **kwargs)

    def cross_correlate_align(self, *args, **kwargs):
        print("\n")
        return cross_correlate_align(self, *args, **kwargs)

    def rotate_correlate_align(self, *args, **kwargs):
        print("\n")
        return rotate_correlate_align(self, *args, **kwargs)

    def PMA(self, *args, **kwargs):
        print("\n")
        return PMA(self, *args, **kwargs)

    def vertical_mass_fluctuation_align(tomo, *args, **kwargs):
        print("\n")
        return vertical_mass_fluctuation_align(tomo, *args, **kwargs)

    def tomopy_align(self, *args, **kwargs):
        print("\n")
        return tomopy_align(self, *args, **kwargs)

    def optical_flow_align(self, *args, **kwargs):
        print("\n")
        return optical_flow_align(self, *args, **kwargs)

    def shift_min_to_middle(self, *args, **kwargs):
        print("\n")
        return shift_min_to_middle(self, *args, **kwargs)
    
    def sinogram_consistency_score(self, *args, **kwargs):
        print("\n")
        return sinogram_consistency_score(self, *args, **kwargs)
    
    def reprojection_consistency_score(self, *args, **kwargs):
        print("\n")
        return reprojection_consistency_score(self, *args, **kwargs)

    def center_projections(self):
        """
        Determines and adjusts the center of rotation for 2D projection images by finding the initial center,
        shifting the projections to center them, and calculating any remaining offset (to check if it needs to be done again).
        
        Run a max of 3 times
        """
        print("Centering Projections")
        self.center_offset = 10
        iterator = 0
        while self.center_offset > 1 and iterator < 3:
            iterator += 1
            self.rotation_center = tomopy.find_center_vo(self.workingProjections)
            print("Original center: {}".format(self.rotation_center))
            print("Center of frame: {}".format(self.image_size[1] // 2))
            x_shift = (self.image_size[1] / 2 - (self.rotation_center))
            y_shift = 0
            x_shift_check = 3
            if abs(x_shift_check) > 2:
                for m in range(self.num_angles):
                    self.workingProjections[m] = subpixel_shift(self.workingProjections[m], y_shift, x_shift)
                self.rotation_center = tomopy.find_center_vo(self.workingProjections)
                print("Aligned projections shifted by {} pixels".format(x_shift))
                x_shift_check = (self.image_size[1] // 2 - (self.rotation_center))
            self.center_offset = abs(x_shift_check)
            print(f"Projections are currently centered at pixel {self.rotation_center}. Residual offset: {self.center_offset}")
            self.tracked_shifts[:, 1] += x_shift

    def reconstruct(self, algorithm, snr_db=None):
        """
        Reconstructs the 3D volume from projections using the specified algorithm.

        Parameters:
        - algorithm (str): The reconstruction algorithm to use.
        - snr_db (float or None): Signal-to-noise ratio for SVMBIR, if applicable.
        """
        #Center projections before reconstruction. So reconstruction knows where center is.
        self.rotation_center = tomopy.find_center_vo(self.finalProjections)

        print("\n")
        if algorithm.endswith("CUDA"):
            if torch.cuda.is_available():
                print("Using GPU-accelerated reconstruction, Algorithm: ", algorithm)
                options = {
                    'proj_type': 'cuda',
                    'method': algorithm,
                    'num_iter': 400,
                    'extra_options': {}
                }
                self.recon = tomopy.recon(
                    self.finalProjections,
                    self.ang,
                    center=self.rotation_center,
                    algorithm=tomopy.astra,
                    options=options,
                    ncore=1
                )
            else:
                raise ValueError("GPU is not available, but the selected algorithm was 'gpu'.")
        elif algorithm == 'svmbir':
            print("Using SVMBIR-based reconstruction.")
            print("center_offset assumed to be : {}".format(self.center_offset))
            if snr_db is None:
                self.recon = svmbir.recon(self.finalProjections, self.ang, center_offset=self.center_offset, verbose=1)
            else:
                self.recon = svmbir.recon(self.finalProjections, self.ang, center_offset=self.center_offset, snr_db=snr_db, verbose=1)
        else:
            print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            self.recon = tomopy.recon(
                self.finalProjections,
                self.ang,
                center=self.rotation_center,
                algorithm=algorithm,
                sinogram_order=False
            )
        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.95)
        print("Reconstruction completed.")