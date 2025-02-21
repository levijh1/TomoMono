import numpy as np
import tifffile

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