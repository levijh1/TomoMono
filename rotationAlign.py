from tiffConverter import convert_to_numpy
import tomopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tomoDataClass
from scipy.ndimage import correlate
from scipy.ndimage import rotate
from scipy.signal import correlate2d


fileLocation = "/Users/levih/Desktop/TomoMono/data/fullTomoReconstructions2.tif"
misalignedProjections, scale_info = convert_to_numpy(fileLocation)
print(misalignedProjections.shape)


badProjections = tomoDataClass.tomoData(misalignedProjections)
# badProjections.cross_correlate_align()
# badProjections.makeScriptProjMovie()



img1 = misalignedProjections[0]
img2 = misalignedProjections[1]

rotated_img2 = rotate(img2, 30, reshape=False, mode='nearest')
correlate(img1, rotated_img2)
plt.imshow(rotated_img2)
plt.show()

# print("Finding optimal angle")
# max_similarity = -100000
# optimal_angle = 0
# angle_range = [-1,1]
# angle_step = 1

# for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
#     print(angle)
#     # Rotate img2 by the current angle
#     rotated_img2 = rotate(img2, angle, reshape=False, mode='nearest')
    
#     # Compute the similarity (cross-correlation) between img1 and the rotated img2
#     similarity = np.sum(correlate2d(img1, rotated_img2, mode='same'))
    
#     # Update the optimal angle if the current similarity is the highest found so far
#     if similarity > max_similarity:
#         max_similarity = similarity
#         optimal_angle = angle

# print(f"Optimal rotation angle: {optimal_angle} degrees, Maximum similarity: {max_similarity}")