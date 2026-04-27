# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:17:48 2026

@author: tbuckway
"""

from skimage.registration import phase_cross_correlation
from skimage.transform import pyramid_gaussian
from scipy.ndimage import gaussian_filter1d
import numpy as np
import cupy as cp 
from sklearn.cluster import KMeans
import tomopy
import functools
import time

if cp.is_available(): 
    from cupyx.scipy.ndimage import fourier_shift, gaussian_filter
    xp = cp 
else:
    from scipy.ndimage import fourier_shift, gaussian_filter
    xp = np
    
def timer(func):
    """Print the execution time of the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()    # Record start time
        value = func(*args, **kwargs)       # Execute the function
        end_time = time.perf_counter()      # Record end time
        run_time = end_time - start_time    # Calculate duration
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper
 
def cross_correlation_alignment(
        images, 
        num_images_for_median=None,
        use_grad=False,
        downsample=1,
        upsample_factor=1,
        additional_padding=10,
        ):
    images = np.asarray(images,dtype=np.float32)
    n,ny,nx= images.shape
    rel_shifts = np.zeros((n,2),dtype=np.float64)   
    abs_shifts = np.zeros((n,2),dtype=np.float64)   
    K = num_images_for_median
    if K is not None and K <= 1:
        K = None
    for i in range(1,n):
        if K is None:
            ref_image = images[i-1]
        else:
            start = max(0,i-K) 
            ref_image = np.median(images[start:i],axis=0)
        mov_image = images[i]
        if use_grad:
            ref_image = compute_grad_image(ref_image)
            mov_image = compute_grad_image(images[i])
        if downsample != 1:
           ref_image = tuple(pyramid_gaussian(
               image=ref_image,
               max_layer=1,
               downscale=downsample,
               sigma=None,
               preserve_range=False,
               ))[1] 
           mov_image = tuple(pyramid_gaussian(
               image=mov_image,
               max_layer=1,
               downscale=downsample,
               sigma=None,
               preserve_range=False,
               ))[1]
        shift_rc, _, _ = phase_cross_correlation(
            reference_image=ref_image,
            moving_image=mov_image,
            upsample_factor=upsample_factor,
            )
        rel_shifts[i] = shift_rc*downsample
        abs_shifts[i] = np.sum(rel_shifts[:i+1],axis=0)
        
    ymax , xmax = int(np.ceil(np.max(np.abs(abs_shifts[:,0])))+additional_padding), int(np.ceil(np.max(np.abs(abs_shifts[:,1])))+additional_padding)
    pad_width = ((0,0),(ymax,ymax),(xmax,xmax))
    padded_images = np.pad(images,pad_width=pad_width,mode='edge')
    aligned_images = np.empty_like(padded_images,dtype=np.float32)
    aligned_images[0] = padded_images[0] 
    for i in range(1,n):
        image = xp.asarray(padded_images[i])        
        fft_image = xp.fft.fft2(image)
        shifted_fft = fourier_shift(fft_image, xp.asarray(abs_shifts[i]))
        shifted_image = xp.real(xp.fft.ifft2(shifted_fft))      
        aligned_images[i] = shifted_image.get() if cp.is_available() else shifted_image 
    return aligned_images, rel_shifts, abs_shifts

def compute_grad_image(image):
    img = xp.asarray(image)
    dx, dy = xp.gradient(img, axis=(-2,-1))
    grad_img = xp.sqrt(dx**2+dy**2)
    return grad_img.get() if cp.is_available() else grad_img

def vertical_mass_fluctuations(
        images,
        mode='phase',
        sigma=1.0,
        upsample_factor=100,
        smooth_sigma=None
        ):
    n, ny, nx = images.shape
    M = np.zeros((n,ny),dtype=np.float64)
    for i in range(n):
        image = xp.asarray(images[i])        
        if mode == 'attenuation':
            p = xp.log(xp.abs(image)+1e-12)
        elif mode == 'phase':
            p = -1*(xp.arg(image)) if image is xp.iscomplexobj(image) else image
        else:
            raise ValueError(f"mode {mode} is not valid entry. Enter \'attenuation\' or \'phase\'.")
        m = xp.sum(p,axis=1)
        m = m - gaussian_filter(m, sigma=sigma)
        M[i] = m.get() if cp.is_available() else m     
    km = KMeans(n_clusters=1, n_init=10, random_state=0)
    km.fit(M)
    ref = km.cluster_centers_[0]
    # ref_idx = int(np.argmin(np.sum((M - centroid) ** 2, axis=1)))
    # ref = M[ref_idx] 
    shifts_y = np.zeros(n,dtype=np.float64)
    aligned_images = np.empty_like(images)
    for i in range(n):
        mov = M[i]
        d_y, _, _ = phase_cross_correlation(
            reference_image=ref,
            moving_image=mov,
            upsample_factor=upsample_factor
            )
        shifts_y[i] = d_y[0]
    if smooth_sigma is not None and smooth_sigma>0:
        shifts_y = gaussian_filter1d(shifts_y, sigma=smooth_sigma)
    for i in range(n):
        image = xp.asarray(images[i])
        fft_image = xp.fft.fft2(image)
        shifted_fft = fourier_shift(fft_image, (shifts_y[i],0.0))
        shifted_image = xp.real(xp.fft.ifft2(shifted_fft))      
        aligned_images[i] = shifted_image.get() if cp.is_available() else shifted_image 
    return aligned_images, shifts_y


class multi_res_PMA:
    def __init__(self,
        projections,
        angles,
        # n_iterations,
        scale=2, # downsample scales (e.g., 4x, 2x)
        levels=3,
        skip_last_level=False,
        pma_method='optical_flow',
        algorithm='gridrec',
        pma_options={},
        recon_params={},
        project_params={},
        ):
        
        # Initial setup
        self.original_projections = projections
        self.angles = angles
        self.n, self.ny, self.nx = projections.shape
               
        # Cumulative shifts in full-res units
        self.shifts = np.zeros((self.n, 2), dtype=np.float64)
        self.aligned = projections.copy()

        if recon_params['center'] is None:
            self.full_center = float(tomopy.find_center(projections, angles, ind=projections.shape[1]//2))
        else:
            self.full_center = recon_params['center']
        
        # Multi-resolution loop
        D_level = 1
        self.volume_results = {}
        self.proj_results = {}
        for level in reversed(range(levels)):
            if skip_last_level and level == 0: 
                print("Skipping highest resolution")
                break 
            print(f"\n--- Starting down sample {scale**level}x ---")
            
            # 1. Prepare downsampled data based on current cumulative shifts
            current_projs = self._apply_cumulative_shifts(self.original_projections, self.shifts)
            proj_8bit = normalize_by_bit_depth(current_projs, '8')
            self.proj_results[f'proj_scale_{scale**level}x'] = proj_8bit
            
            if level > 0:
                scaled_projs = list(pyramid_gaussian(current_projs,downscale=scale,channel_axis=0))[level]
            else:
                scaled_projs = current_projs
            
            recon_params_low = recon_params.copy() 
            project_params_low = project_params.copy() 
            
            recon_params_low['center'] = self.full_center / (scale**level)
            project_params_low['center'] = self.full_center / (scale**level)
            
            # We initialize a temporary PMA instance for the sub-problem
            pma = projection_matching_alignment(
                scaled_projs, 
                self.angles,
                gamma=0.01,
                D_level=D_level,
                pma_method=pma_method,
                algorithm=algorithm,
                pma_options=pma_options.copy(),
                recon_params=recon_params_low,
                project_params=project_params_low
            )
                       
            vol_16bit = normalize_by_bit_depth(pma.volume, '16')
            self.volume_results[f'scale_{scale**level}x'] = vol_16bit 
            # 4. Upscale the found shifts and add to total
            self.shifts += pma.shifts * scale**level
            D_level += 1 
        # Final Aligned Result
        self.aligned = self._apply_cumulative_shifts(self.original_projections, self.shifts)
        print("\nMulti-resolution alignment complete.")
    def _apply_cumulative_shifts(self, data, shifts):
        """Helper to shift images using Fourier shift."""
        shifted = np.zeros_like(data)
        for i in range(self.n):
            img = xp.asarray(data[i])
            fft_image = xp.fft.fft2(img)
            shifted_fft = fourier_shift(fft_image, shifts[i])
            shifted_image = xp.real(xp.fft.ifft2(shifted_fft))
            shifted[i] = shifted_image.get() if cp.is_available() else shifted_image
        return shifted

@timer
class projection_matching_alignment:
    def __init__(self,
        projections,
        angles,
        # n_iterations,
        pma_method='optical_flow',
        algorithm='gridrec',
        gamma=0.01,
        D_level=1,
        pma_options={},
        recon_params={},
        project_params={},
        ):
        self.center = recon_params.get('center')
        self.gamma=gamma
        self.D = D_level
        self.prev_energy = None
        
        if self.center is None:
            print("Center not provided. Calculating using tomopy.find_center...")
            # angles = np.asarray(angles).astype(np.float32, copy=False)
            # projections_f32 = np.asarray(projections).astype(np.float32, copy=False)            
            self.center = float(tomopy.find_center(projections, angles, ind=projections.shape[1]//2))
            print(f"Automatic center found at: {self.center}")
        
        recon_params['center'] = self.center
        project_params['center'] = self.center 
        
        pma_method_options = {
            'optical_flow': self._optical_flow_method,
            'cross_correlation': self._cross_correlation
            }
        
        if algorithm == 'tomopy.astra':
            self.algorithm = tomopy.astra
        else:
            self.algorithm = algorithm
            
        self.aligned = projections.copy()
        n,ny,nx = self.aligned.shape
        self.shifts = np.zeros((n,2),dtype=np.float64())
        
        angles = np.asarray(angles, dtype=np.float32).ravel()
        
        recon_params = {k: v for k, v in recon_params.items() if v is not None}
        
        self.converged = False
        it = 0
        self.counter = 0 
        self.prev_steps = (1.0,1.0)
        print(f"Running PMA until converges with dx and dy < {max(self.gamma / self.D, 0.002)}")
        while not self.converged:    
        # for it in range(n_iterations):
            print(f"PMA iteration {it+1}")
            self.volume = tomopy.recon(
                self.aligned,
                angles,
                algorithm=self.algorithm,
                **recon_params
                )
            self.reproj = tomopy.sim.project.project(
                self.volume,
                angles, 
                **project_params
                )
            pma_method_options[pma_method](**pma_options)
            it += 1
            if self.counter == 5:
                break
            if it == 100:
                print(f"PMA breaks after iteration {it}")
                break
            
        print(f"PMA converged in {it} iterations")    
    def _optical_flow_method(self,weights=None,sigma=3.0,smooth_sigma=1.0, **kwargs):
        if weights is None:
            weights = np.ones_like(self.aligned)
        n,ny,nx = self.aligned.shape 
        dx = np.zeros(n,dtype=np.float64())
        dy = np.zeros(n,dtype=np.float64())
        
        energy = 0.0
        
        for i in range(n):
            p = self.aligned[i]
            p_hat = self.reproj[i]
            W = weights[i] 
            dp_haty,dp_hatx = compute_grads(p_hat) 
            r_hp = highpass_filter(p-p_hat,sigma)
            energy += np.sum(r_hp**2)
            denom = np.sum(W**2*r_hp**2)+1e-8
            dx[i] = np.sum(W**2*highpass_filter(dp_hatx,sigma)*r_hp)/denom
            dy[i] = np.sum(W**2*highpass_filter(dp_haty,sigma)*r_hp)/denom
        
        dxmax = np.max(np.abs(dx))
        dymax = np.max(np.abs(dy))

        if max(dxmax,dymax) < 0.05:
            alpha=0.2 
        else:
            alpha = 1.0
        dx *= alpha
        dy *= alpha
        if self.prev_energy is not None:
            dE = abs(self.prev_energy -energy)/(self.prev_energy+1e-8)
            if dE < 1e-4:
                print("Stopping: energy plateau reached")
                self.converged = True 
        self.prev_energy = energy 
        
        step_key = (round(dxmax,3),round(dymax,3))
        if self.prev_steps == step_key:
            self.counter += 1 
        else:
            self.prev_steps = step_key
            self.counter = 0 
        
        # dx = gaussian_filter1d(dx, smooth_sigma)
        # dy = gaussian_filter1d(dy, smooth_sigma)
            
        self.shifts[:,0] += dy
        self.shifts[:,1] += dx
        if alpha == 0.2:
            print(f"Regularized shifts of ({dxmax:.03f},{dymax:.03f})x{alpha}")
        print(f"Max shifts are ({np.abs(dx.max()):.03f},{np.abs(dy.max()):.03f})")

        if max(dxmax, dymax) < max(self.gamma / self.D, 0.002):
                print("Stopping: noise floor reached")
                self.converged = True
        self._shift_images(dy, dx)
        
    def _cross_correlation(self,upsample_factor=100,smooth_sigma=1.0, **kwargs):
        n,ny,nx = self.aligned.shape 
        dx = np.zeros(n,dtype=np.float64())
        dy = np.zeros(n,dtype=np.float64())
        for i in range(n):
            shift, _, _ = phase_cross_correlation(
                self.reproj[i], self.aligned[i],upsample_factor=upsample_factor,
                )
            dy[i], dx[i] = shift 
        
        alpha=1.0
        dx *= alpha
        dy *= alpha
        
        dxmax = np.round(np.max(np.abs(dx)),3)
        dymax = np.round(np.max(np.abs(dy)),3)
        if self.prev_steps == (dxmax,dymax):
            self.counter += 1 
        else:
            self.prev_step = (dxmax,dymax)
            self.counter = 0 

        # dx = gaussian_filter1d(dx, smooth_sigma)
        # dy = gaussian_filter1d(dy, smooth_sigma)
        
        self.shifts[:,0] += dy
        self.shifts[:,1] += dx 
        print(f"Max shifts are ({np.abs(dx.max()):.03f},{np.abs(dy.max()):.03f})")
        if np.max(np.abs(dx)) < self.gamma/self.D and np.max(np.abs(dy)) < self.gamma/self.D:
            self.converged = True 
        
        self._shift_images(dy, dx)    
    def _shift_images(self,dy,dx):
        n, ny, nx = self.aligned.shape
        for i in range(n):
            img = xp.asarray(self.aligned[i])
            fft_image = xp.fft.fft2(img)
            shifted_fft = fourier_shift(fft_image, (dy[i],dx[i]))
            shifted_image = xp.real(xp.fft.ifft2(shifted_fft))      
            self.aligned[i] = shifted_image.get() if cp.is_available() else shifted_image 
    
    

def highpass_filter(image, sigma=3.0):
    """Linear high-pass filter F_hp = I - Gaussian(I)"""
    img = xp.asarray(image)
    hp_img = img - gaussian_filter(img, sigma)
    return hp_img.get() if cp.is_available() else hp_img

def compute_grads(image):
    ny, nx = image.shape
    img = xp.asarray(image)
    
    F = xp.fft.fft2(img)
    
    ux = xp.fft.fftfreq(nx).reshape(1,-1)
    uy = xp.fft.fftfreq(ny).reshape(-1,1)
    
    dx = xp.real(xp.fft.ifft2(2j*xp.pi*ux*F))
    dy = xp.real(xp.fft.ifft2(2j*xp.pi*uy*F))
    
    # dy,dx = xp.gradient(img)
    return (dy.get(), dx.get()) if cp.is_available() else (dy,dx)

# def crop_images(images, abs_shifts):
#     _, H, W = images.shape

#     # Row and column shifts
#     row_shifts = abs_shifts[:, 0]
#     col_shifts = abs_shifts[:, 1]

#     # Convert to conservative integer bounds
#     row_min = int(np.floor(row_shifts.min()))
#     row_max = int(np.ceil(row_shifts.max()))
#     col_min = int(np.floor(col_shifts.min()))
#     col_max = int(np.ceil(col_shifts.max()))

#     # Compute valid crop region
#     r_start = max(0, row_max)
#     r_end   = min(H, H + row_min)

#     c_start = max(0, col_max)
#     c_end   = min(W, W + col_min)

#     # Safety check
#     if r_start >= r_end or c_start >= c_end:
#         raise ValueError("Invalid crop region computed from shifts")

#     return images[:, r_start:r_end, c_start:c_end]

def crop_images(images, orig_shape, total_shifts):
    _ , ny, nx = images.shape
    org_y, org_x = orig_shape
    pad_cen_x, pad_cen_y = ((nx-1)/2,(ny-1)/2)
    dx_cen, dy_cen = np.median(total_shifts,0)
    crop_cen_x = pad_cen_x + dx_cen 
    crop_cen_y = pad_cen_y + dy_cen
    rmin = int(np.floor(crop_cen_y - (org_y-1)/2)) 
    rmax = rmin + org_y 
    cmin = int(np.floor(crop_cen_x - (org_x-1)/2)) 
    cmax = cmin + org_x 
    cropped_images = images[:,rmin:rmax,cmin:cmax]
    return cropped_images 

def normalize_by_bit_depth(arr, bit_depth):
    #taken from pear as well from same module

    arr = arr.astype(np.float32, copy=False)

    amin = arr.min()
    amax = arr.max()
    scale = amax - amin

    if scale == 0:
        raise ValueError("Array has zero dynamic range")

    # normalize in-place
    arr = (arr - amin) / scale


    if bit_depth == "8":
        return (arr * 255).astype(np.uint8, copy=False)
    elif bit_depth == "16":
        return (arr * 65535).astype(np.uint16, copy=False)
    elif bit_depth == "32":
        return arr.astype(np.float32, copy=False)
    elif bit_depth == "raw":
        return arr
    else:
        return arr

if __name__ == '__main__': 
    import prpty.load as ld 
    import tifffile
    # username= 'taylo'
    # # images = ld.extract_from_hdf5('data', h5file=r"C:\Users\taylo\Box\BYU_CXI_Research_Team\ProjectFolders\IFE-STAR\IFE-Ptycho-Tomo\APS_2ID_GUP_October2025\data\tomo_data_run_final.hdf5")
    # dataset = ld.Load_pty(rf"C:\Users\{username}\Box\BYU_CXI_Research_Team\ProjectFolders\IFE-STAR\IFE-Ptycho-Tomo\APS_2ID_GUP_October2025\data\tomo_data_run_final.hdf5")
    # images = dataset.dataset['data'][:8,:,400:800]
    # angles = dataset.dataset['angles'][:8]
    # # dx, dy = np.gradient(images, axis=(-1,-2))
    # # grad_image = np.sqrt(dx**2+dy**2) 
    # start_time = time.time()
    # cc_images, rel_shifts, abs_shifts= cross_correlation_alignment(images,
    #                                         num_images_for_median=1,
    #                                         use_grad=True,
    #                                         downsample=4,
    #                                         upsample_factor=100
    #                                         )
    # print(f'XCA took {time.time()- start_time} seconds')
    # start_time = time.time()
    # vmf_images, shifts = vertical_mass_fluctuations(cc_images,
    #                                                 mode='phase',
    #                                                 sigma=2.0,
    #                                                 upsample_factor=100,
    #                                                 smooth_sigma=2.0
    #                                                 )
    # print(f'VMF took {time.time()-start_time} seconds')
    
    # pma = projection_matching_alignment(vmf_images, angles, 1)

    # cc_images_refined, shifts= cross_correlation_alignment(cc_images,upsample_factor=100,num_images_for_median=None)
    # cc_images, shifts = cross_correlation_alignment_chunked(
    #     images=images,
    #     subset_size=200,
    #     upsample_factor=100,
    #     normalization='phase')
    
    # images = normalize_by_bit_depth(images, '16')
    
    # cc_images = normalize_by_bit_depth(cc_images, "16")
    # vmf_images = normalize_by_bit_depth(vmf_images, "16")
    
    # tifffile.imwrite(rf"C:\Users\{username}\OneDrive - Brigham Young University\Desktop\temp_imagecc.tiff",cc_images)
    # tifffile.imwrite(rf"C:\Users\{username}\OneDrive - Brigham Young University\Desktop\temp_image.tiff",images)
    # tifffile.imwrite(rf"C:\Users\{username}\OneDrive - Brigham Young University\Desktop\temp_vmf_images.tiff",vmf_images)

    data = tifffile.imread(r"C:\Users\taylo\Downloads\pma_align.tiff")
    data_reduced = normalize_by_bit_depth(data, '8')