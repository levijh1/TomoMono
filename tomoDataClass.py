import tomopy
import numpy as np
import scipy as sp
from helperFunctions import MoviePlotter, subpixel_shift, runwidget
from tqdm import tqdm
from scipy.ndimage import rotate
from alignment_methods import *

from gpu import xp, cp, torch, svmbir, ndimage_shift as _ndimage_shift


def simulate_projections(recon, angles, center=None, emission=True, pad=False, ncore=None, use_astra=None):
    """
    Simulate projections through a 3D volume at the given angles.

    Parameters
    ----------
    recon : ndarray (nz, ny, nx)
    angles : ndarray, radians
    center : float or None — rotation center; None uses nx/2 (tomopy default)
    emission : bool
    pad : bool
    ncore : int or None
    use_astra : bool or None — if True, require ASTRA; if False, use tomopy only; if None (default), try ASTRA first, fall back to tomopy
    """
    # Auto-disable ASTRA when there's no working GPU — ASTRA prints alarming
    # stderr messages before raising, even though the fallback path works.
    if use_astra is None and torch is None:
        use_astra = False
    # Try ASTRA if not explicitly disabled
    if use_astra is not False:
        vol_id = proj_id = alg_id = None
        try:
            import astra
            nz, ny, nx = recon.shape
            vol_geom = astra.create_vol_geom(ny, nx, nz)
            proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, nz, nx, angles)
            vol_id = astra.data3d.create('-vol', vol_geom, data=recon.astype(np.float32))
            proj_id = astra.data3d.create('-sino', proj_geom)
            cfg = astra.astra_dict('FP3D_CUDA')
            cfg['ProjectionDataId'] = proj_id
            cfg['VolumeDataId'] = vol_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            proj_data = astra.data3d.get(proj_id)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(vol_id)
            # ASTRA parallel3d output: (nz, n_angles, nx) → tomopy order: (n_angles, nz, nx)
            return np.transpose(proj_data, (1, 0, 2))
        except Exception as e:
            # Free any ASTRA GPU objects that were created before the failure
            try:
                if alg_id is not None:
                    astra.algorithm.delete(alg_id)
                if proj_id is not None:
                    astra.data3d.delete(proj_id)
                if vol_id is not None:
                    astra.data3d.delete(vol_id)
            except Exception:
                pass
            if use_astra is True:
                raise
            print(f"ASTRA forward projection failed ({e}), falling back to tomopy.project")
    kwargs = {'emission': emission, 'pad': pad, 'ncore': ncore}
    if center is not None:
        kwargs['center'] = center
    return tomopy.project(recon, angles, **kwargs)


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
        self.recon = None
        self._shift_envelope = [0.0, 0.0, 0.0, 0.0]      # [top, bottom, left, right]
        self._shift_envelope_idx = [0, 0, 0, 0]            # projection index for each max shift
        if angles is None:
            self.ang = tomopy.angles(nang=self.num_angles, ang1=0, ang2=360)
        else:
            self.ang = angles

    def reset_workingProjections(self, x_size=None, y_size=None, cropBottomCenter=False):
        """
        Resets working and final projections to the original data and crops to the specified center size.
        This is useful for starting fresh with the original projections after modifications back to back.

        Parameters:
        - x_size (int): Width of the crop.
        - y_size (int): Height of the crop.
        """
        self.workingProjections = np.copy(self.data)
        self.finalProjections = np.copy(self.data)
        if x_size is not None and y_size is not None:
            anchor = 'bottom' if cropBottomCenter else 'center'
            self.crop(y_size, x_size, anchor=anchor)

    def get_recon(self):
        """Returns the reconstructed 3D volume (``self.recon``)."""
        return self.recon

    def get_working_projections(self):
        """Returns the current working projections (the ones being modified in place)."""
        return self.workingProjections

    def get_final_projections(self):
        """Returns the final projections (the ones used for reconstruction)."""
        return self.finalProjections

    # Backwards-compatibility shims for the old camelCase / inconsistent names.
    def get_workingprojections(self):
        """Deprecated alias — use ``get_working_projections()``."""
        return self.get_working_projections()

    def get_finalProjections(self):
        """Deprecated alias — use ``get_final_projections()``."""
        return self.get_final_projections()

    def jitter(self, maxShift=5):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        Jitter ranges from -maxShift to +maxShift pixels in both x and y directions. Affects data variable as well.

        Parameters:
        - maxShift (float): Maximum shift in pixels for both x and y directions.
        """
        for i in range(1, self.num_angles-1):
            x_shift = np.random.uniform(-maxShift, maxShift)
            y_shift = np.random.uniform(-maxShift, maxShift)
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
        for i in range(1, self.num_angles-1):
            x_shift = 0
            y_shift = np.random.uniform(-maxShift, maxShift)
            arr = xp.asarray(self.data[i])
            shifted = _ndimage_shift(arr, (y_shift, x_shift), mode='reflect')
            self.data[i] = shifted.get() if xp is not np else shifted
        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()

    def add_noise(self):
        self.data = tomopy.prep.alignment.add_noise(self.data)
        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()   

    def crop(self, new_y, new_x, anchor='center'):
        """
        Crop each projection to the requested (new_y, new_x) size.

        The horizontal crop is always centered on the rotation axis (so the
        center of rotation stays at the midpoint of the cropped frame).
        The vertical crop is controlled by ``anchor``:

          anchor='center' (default): centered vertically
          anchor='bottom':           bottom-aligned

        Parameters:
        - new_y (int or None): target height; None leaves height unchanged.
        - new_x (int or None): target width;  None leaves width unchanged.
        - anchor ({'center','bottom'}): vertical alignment of the crop.
        """
        if anchor not in ('center', 'bottom'):
            raise ValueError(f"anchor must be 'center' or 'bottom', got {anchor!r}")

        y, x = self.workingProjections[0].shape

        # Horizontal crop is always centered
        if new_x is None:
            startx, endx = 0, x
        else:
            startx = max(0, x // 2 - new_x // 2)
            endx = min(x, startx + new_x)

        # Vertical crop depends on anchor
        if new_y is None:
            starty, endy = 0, y
        elif anchor == 'bottom':
            starty = max(0, y - new_y)
            endy = y
        else:  # 'center'
            starty = max(0, y // 2 - new_y // 2)
            endy = min(y, starty + new_y)

        self.workingProjections = self.workingProjections[:, starty:endy, startx:endx]
        self.finalProjections = self.finalProjections[:, starty:endy, startx:endx]
        self.image_size = self.workingProjections.shape[1:]

        removed_top, removed_bottom = starty, y - endy
        removed_left, removed_right = startx, x - endx
        self._shift_envelope[0] = max(0.0, self._shift_envelope[0] - removed_top)
        self._shift_envelope[1] = max(0.0, self._shift_envelope[1] - removed_bottom)
        self._shift_envelope[2] = max(0.0, self._shift_envelope[2] - removed_left)
        self._shift_envelope[3] = max(0.0, self._shift_envelope[3] - removed_right)

    def crop_center(self, new_x, new_y):
        """Deprecated shim — use ``tomo.crop(new_y, new_x, anchor='center')``."""
        return self.crop(new_y, new_x, anchor='center')

    def crop_bottom_center(self, new_y, new_x):
        """Deprecated shim — use ``tomo.crop(new_y, new_x, anchor='bottom')``."""
        return self.crop(new_y, new_x, anchor='bottom')


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
        Applies tracked rotations to the final projections and resets tracked rotations.
        """
        for m in tqdm(range(self.num_angles), desc='Apply rotations to final projections'):
            self.finalProjections[m] = rotate(self.finalProjections[m], self.tracked_rotations[m], reshape=False, mode='constant')
        self.tracked_rotations = np.zeros(self.num_angles)

    def normalize(self, isPhaseData):
        """
        Normalizes all projections to be positive values between 0 and 1.
        """
        print("\n")
        print("Normalizing projections")
        arr = xp.asarray(self.workingProjections)
        if isPhaseData:
            arr *= -1
        arr -= xp.min(arr)
        arr /= xp.max(arr)
        self.workingProjections = arr.get() if xp is not np else arr
        # self.finalProjections = np.copy(self.workingProjections)

    def standardize(self, isPhaseData):
        """
        Standardizes projections to have zero mean and unit variance, then inverts the sign.

        Parameters:
        - isPhaseData (bool): If True, inverts the sign of the projections, which is often necessary for phase data since it can be negative.
        """
        arr = xp.asarray(self.workingProjections)
        arr -= xp.mean(arr)
        arr /= xp.std(arr)
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
    
    def displayWorkingSinogram(self, row_index=None):
        if row_index is None:
            row_index = self.workingProjections.shape[1] // 2
        plt.imshow(self.workingProjections[:,row_index,:], cmap='gray')
        plt.title(f'Sinogram (Row {row_index})')
        plt.ylabel('Angle')
        plt.show()

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
            if abs(x_shift) > 0.01:
                for m in range(self.num_angles):
                    self.workingProjections[m] = subpixel_shift(self.workingProjections[m], y_shift, x_shift)
                self.rotation_center = tomopy.find_center_vo(self.workingProjections)
                print("Aligned projections shifted by {} pixels".format(x_shift))
                x_shift_check = (self.image_size[1] // 2 - (self.rotation_center))
            else:
                x_shift_check = x_shift
            self.center_offset = abs(x_shift_check)
            print(f"Projections are currently centered at pixel {self.rotation_center}. Residual offset: {self.center_offset}")
            self.tracked_shifts[:, 1] += x_shift

    def reconstruct(self, algorithm, snr_db=None, num_iter=400, extra_options=None):
        """
        Reconstructs the 3D volume from projections using the specified algorithm.

        Parameters:
        - algorithm (str): The reconstruction algorithm to use.
        - snr_db (float or None): Signal-to-noise ratio for SVMBIR, if applicable.
        - num_iter (int): Number of iterations for iterative CUDA algorithms (default 400).
        - extra_options (dict or None): Extra ASTRA options (e.g. {'MinConstraint': 0}).
        """
        #Center projections before reconstruction. So reconstruction knows where center is.
        self.rotation_center = tomopy.find_center_vo(self.finalProjections)

        print("\n")
        if algorithm.endswith("CUDA"):
            if torch is not None and torch.cuda.is_available():
                print("Using GPU-accelerated reconstruction, Algorithm: ", algorithm)
                options = {
                    'proj_type': 'cuda',
                    'method': algorithm,
                    'num_iter': num_iter,
                    'extra_options': extra_options or {}
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
        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.99)
        self._recon_pre_kovacik = None  # reset so kovacik_filter uses the new recon
        print("Reconstruction completed.")

    def simulateProjections(self, recon=None, angles=None, center=None, emission=True, pad=False, ncore=None, use_astra=None):
        """
        Simulate projections through the reconstruction at the angles stored in this object.

        Parameters
        ----------
        recon : ndarray or None — volume to project; defaults to self.recon
        angles : ndarray or None — projection angles in radians; defaults to self.ang
        center : float or None — rotation center; defaults to self.rotation_center (None → nx/2)
        emission : bool
        pad : bool
        ncore : int or None
        use_astra : bool or None — if True, require ASTRA; if False, use tomopy only; if None (default), try ASTRA first, fall back to tomopy
        """
        if recon is None:
            if not hasattr(self, 'recon') or self.recon is None:
                raise AttributeError("No reconstruction available. Call reconstruct() first or provide recon argument.")
            recon = self.recon
        if angles is None:
            angles = self.ang
        if center is None:
            rc = getattr(self, 'rotation_center', 0)
            center = rc if rc else None
        return simulate_projections(recon, angles, center=center, emission=emission, pad=pad, ncore=ncore, use_astra=use_astra)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-attach delegates: each free function below is exposed as a method on
# ``tomoData`` that takes the instance as its first argument. The generated
# method prints a leading newline (preserves prior visual spacing in logs),
# then forwards to the free function and returns its result. The wrapped
# function's docstring is preserved so ``help(tomo.PMA)`` still works.
# ─────────────────────────────────────────────────────────────────────────────

def _attach_delegate(cls, fn):
    """Attach a method on ``cls`` named ``fn.__name__`` that calls ``fn(self, ...)``."""
    name = fn.__name__

    def delegate(self, *args, **kwargs):
        print("\n")
        return fn(self, *args, **kwargs)

    delegate.__name__ = name
    delegate.__qualname__ = f"{cls.__name__}.{name}"
    delegate.__doc__ = fn.__doc__
    delegate.__wrapped__ = fn
    setattr(cls, name, delegate)


for _fn in (
    bilateralFilter,
    cross_correlate_align,
    rotate_correlate_align,
    projection_matching_alignment,
    vertical_mass_fluctuation_align,
    tomopy_align,
    optical_flow_align,
    shift_min_to_middle,
    sinogram_consistency_score,
    reprojection_consistency_score,
    fourier_shell_correlation,
    kovacik_filter,
):
    _attach_delegate(tomoData, _fn)
del _fn

# Backwards-compatibility alias for the previous CAPS naming.
tomoData.PMA = tomoData.projection_matching_alignment