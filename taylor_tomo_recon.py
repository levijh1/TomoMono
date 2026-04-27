# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:04:53 2025

@author: tbuckway
"""

import tomopy
import tomo_align 
import logging
import tifffile 
import numpy as np
import time
import h5py
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


def tomo_data(file,redo_align=False):
    try:
        with h5py.File(file) as hf:
            if redo_align:
                projs = hf['data'][...]
                use_aligned_data = False
            else:
                projs = hf['aligned'][...]
                use_aligned_data = True
            angles = hf['angles'][...]
    except KeyError:
        with h5py.File(file) as hf:
            projs = hf['data'][...]
            angles = hf['angles'][...]
        use_aligned_data = False
    return projs, angles, use_aligned_data

def align_projs(projs, angles):
    _ , ny, nx = projs.shape
    available_cores = int(os.environ.get('SLURM_NTASKS', os.cpu_count() or 1))
    print(f"Running with {available_cores} cores.")
    
    with open("tomopy_alg_options.json", "r") as f: 
        algo_library = json.load(f) 
    
    selected_algo = 'fbp'
    algo_specific_params = algo_library[selected_algo]["kwargs"]
    if selected_algo == 'tomopy.astra':
        algo_name = algo_specific_params['options']['method']
        algo_specific_params['options']['num_iter'] = 3
    else:
        algo_name = selected_algo
    
    base_recon_params = {
            'center': None,
            'sinogram_order': False,
            'ncore': available_cores,
            'nchunk': None,
        } 
    
    full_recon_params = base_recon_params | algo_specific_params
    
    xca_params = {'num_images_for_median': 1, 'use_grad': True, 'downsample': 4, 'upsample_factor': 100, 'additional_padding': 10}
    vmf_params = {'mode': 'phase', 'sigma': 2.0, 'upsample_factor': 100, 'smooth_sigma': 2.0}
    pma_options = {'weights': None, 'sigma': 3.0, 'smooth_sigma': 1.0}
    project_params = {'center': None, 'emission': True, 'pad': False, 'ncore': available_cores, 'nchunk': None}
    
    start_time = time.time()
    XCA_projs, xca_rel_shifts, xca_abs_shifts= tomo_align.cross_correlation_alignment(projs, **xca_params)
    print(f'XCA took {time.time()- start_time} seconds')
    start_time = time.time()
    vmf_projs, vmf_shifts = tomo_align.vertical_mass_fluctuations(XCA_projs, **vmf_params)
    print(f'VMF took {time.time()-start_time} seconds')
    
    vmf_to_orig_shape = tomo_align.crop_images(vmf_projs,(ny,nx), xca_abs_shifts+vmf_shifts[:,np.newaxis])
    reduced_vmf = vmf_to_orig_shape[:,:,:] # choose a roi for pma to work with 
    padded_pre_pma = np.pad(reduced_vmf,((0,0),(10,10),(10,10)),mode='edge')
    
    start_time = time.time()
    # pma = tomo_align.projection_matching_alignment(
    #     vmf_projs_cropped,
    #     angles,
    #     gamma=0.01,
    #     D_level=1,
    #     algorithm='gridrec',
    #     pma_method='optical_flow',
    #     pma_options=pma_options,
    #     recon_params=full_recon_params,
    #     project_params=project_params,
    #     )
    pma = tomo_align.multi_res_PMA(
        padded_pre_pma,
        angles,
        scale=2,
        levels=7,
        skip_last_level=True,
        pma_method='optical_flow',
        algorithm=selected_algo,
        pma_options=pma_options,
        recon_params=full_recon_params,
        project_params=project_params,
        )

    print(f'PMA took {time.time()-start_time} seconds')  
    # print("skipped PMA alignment")
    
    alignment_config = {
            "selected_algorithm": algo_name,
            "xca_params": xca_params,
            "vmf_params": vmf_params,
            "pma_options": pma_options,
            "recon_params": full_recon_params,
            "project_params": project_params,
        }
    
    parent_path = Path('../groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo')
    align_results = parent_path / f'algo-{algo_name}'
    file_name = f'algo-{algo_name}.json'
    if not Path.exists(align_results):
        Path.mkdir(align_results ,parents=True,exist_ok=True)

    with open(align_results / file_name,'w') as f:
        json.dump(alignment_config, f, indent=4)
    
    pma_projs = pma.aligned
    
    XCA_results = tomo_align.normalize_by_bit_depth(XCA_projs, "8")
    VMF_results = tomo_align.normalize_by_bit_depth(vmf_projs, "8")
    PMA_results = tomo_align.normalize_by_bit_depth(pma_projs, "8")
    
    print('Writing tifffiles')
    tifffile.imwrite(align_results / 'XCA_align.tiff', XCA_results)
    tifffile.imwrite(align_results / 'vmf_align.tiff', VMF_results)
    tifffile.imwrite(align_results / 'pma_align.tiff', PMA_results)
    
    pma_vol_results = align_results / 'pma_volumes'
    if not Path.exists(pma_vol_results):
        Path.mkdir(pma_vol_results,parents=True,exist_ok=True)
    
    for key, val in pma.volume_results.items():
        tifffile.imwrite(pma_vol_results / f'{key}.tiff',val)    
    
    pma_proj_results = align_results / 'pma_proj'
    if not Path.exists(pma_proj_results):
        Path.mkdir(pma_proj_results,parents=True,exist_ok=True)
    
    for key, val in pma.proj_results.items():
        tifffile.imwrite(pma_proj_results / f'{key}.tiff',val)    
    
    pma_shifts = pma.shifts
    shifts_dict = {
        'xca_abs_shifts': xca_abs_shifts,
        'xca_rel_shifts': xca_rel_shifts,
        'vmf_shifts': vmf_shifts[:,np.newaxis],
        'pma_shifts': pma_shifts,
        }
    print('Saving shift plot')
    plot_shifts(shifts_dict,align_results / 'shifts_plot.png')
    
    # aligned_projs = tomo_align.crop_images(pma_projs, pma_shifts)
    aligned_projs = pma_projs 

    return aligned_projs, shifts_dict, pma.full_center

def plot_shifts(shifts_dict,fname):
    num_plots = len(shifts_dict.keys())
    fig, ax = plt.subplots(1,num_plots-1)
    i = 0
    for key, value in shifts_dict.items():
        if key == 'xca_rel_shifts': 
            continue
        if key == 'vmf_shifts':
            ax[i].scatter(value[:,0],np.zeros_like(value[:,0]),label=key)
        else: 
            ax[i].scatter(value[:,1],value[:,0],label=key)
        ax[i].set_title(key)
        i+=1
    plt.tight_layout()
    plt.savefig(fname=fname)
    

    

def main():
    """According to Dierolf et al. the following steps are need to perform 
    ptychotomograph. 
    1) ptychographic reconstruction of individual and amplitude projections;
    2) Phase-unwrapping and phase-ramp removal;
    3) Registration alignment of projections to correct for experimental 
        inaccuracies in positioning;
    4) Quantitative tomographic reconstruction (of both phase and amplitude part) 
        by standard or a more advanced tomography algorithm
    5) Post-processing of phase and attenuation volumes into different basis 
        sets like electron density map
    This list given and referenced in https://www.nature.com/articles/s41566-017-0072-5.pdf
    """
    parent_path = Path('../groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata')
    hfile = parent_path / 'tomo_data_run_final.hdf5'
    projs, angles, use_aligned_data = tomo_data(file=hfile,redo_align=True)
    angles = angles * np.pi / 180
    if not use_aligned_data:
        print("Aligning tomography images")
        aligned_projs, shifts_dict, center = align_projs(projs,angles)
    else:
        aligned_projs = projs 
        print("Using prevously aligned images")
    
    #------Save aligned projections-------
    if not use_aligned_data:
        print(f"Saving aligned images in {hfile.name}")
        with h5py.File(hfile,'r+') as hf:
            if 'aligned' in hf:
                del hf['aligned']
            if 'shifts' in hf:
                del hf['shifts'] 
            hf.create_dataset('aligned', data=aligned_projs,compression='gzip')
            shifts = hf.create_group('shifts')
            for key, value in shifts_dict.items():
                shifts.create_dataset(name=key,data=value,compression='gzip')
    
    print("Tomography alignment complete!")
    # save_tomo_path = parent_path / 'tomo' 
    # if not Path.exists(save_tomo_path):
    #     Path.mkdir(save_tomo_path,parents=True,exist_ok=True)
        
    #-----Save alignement parameters-----
    
    
    
        
    # with open("tomopy_alg_options.json", "r") as f: 
    #     algo_library = json.load(f) 

    
    # n, ny, nx = projs.shape
    
    # # rot_center = tomopy.find_center(aligned_projs,angles, ind=ny//2)
    
    # sel_algo = 'fbp'
    # algo_opts = algo_library[sel_algo]['kwargs']
    # algo_opts = {k: v for k, v in algo_opts.items() if v is not None}
    # if sel_algo == 'tomopy.astra':
    #     algorithm = tomopy.astra
    #     algo_name = algo_opts['options']['method']
    # else:
    #     algorithm = sel_algo
    #     algo_name = sel_algo
    # available_cores = int(os.environ.get('SLURM_NTASKS', os.cpu_count() or 1))
    # print(f"Performing recon on aligned data using {algo_name}")
    # start_time = time.time()
    # recon = tomopy.recon(
    #     aligned_projs,
    #     angles,
    #     center=center, 
    #     algorithm=algorithm,
    #     sinogram_order=False,
    #     ncore=available_cores,
    #     **algo_opts
    #     )
    # print(f'Recon took {time.time()-start_time:.01f} seconds')
    
    # project_params = {'center': center, 'emission': True, 'pad': False, 'ncore': available_cores}

    # recon_projs = tomopy.sim.project.project(
    #     recon,
    #     angles,
    #     **project_params,
    #     )

    # recon_path = parent_path / 'tomo' / 'recons' 
    # if not Path.exists(recon_path):
    #     Path.mkdir(recon_path ,parents=True,exist_ok=True)
    # print(f'Saving recon as recon_{algo_name}.tiff')
    # tifffile.imwrite(recon_path / f'recon_{algo_name}.tiff',recon)
    # tifffile.imwrite(recon_path / f'recon_projs_{algo_name}.tiff', recon_projs)
    # print("Finished tomography!")
    
    


if __name__ == '__main__':
    main()