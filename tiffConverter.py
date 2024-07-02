import numpy as np
import tifffile

def convert_to_numpy(file_location):
    # Load the TIF file using tifffile
    tif_data = tifffile.imread(file_location)        
    
    # Ensure the data is a 3D numpy array
    if tif_data.ndim == 2:  # If the TIF is only a single image, add a dimension
        tif_data = np.expand_dims(tif_data, axis=0)
    elif tif_data.ndim > 3:  # If the TIF has more than 3 dimensions, raise an error
        raise ValueError("The TIF file has more than 3 dimensions, which is not supported.")
    
    return tif_data

def convert_to_tiff(numpy_data, file_location):
    # Ensure the numpy array is 3D
    if numpy_data.ndim != 3:
        raise ValueError("The numpy array must be 3D.")
    
    # Save the 3D numpy array as a TIF file
    tifffile.imsave(file_location, numpy_data)