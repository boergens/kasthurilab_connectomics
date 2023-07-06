'''
@File    :   overlap.py
@Time    :   2023/07/05 11:00:56
@Author  :   Surya Chandra Kalia
@Version :   1.0
@Contact :   suryackalia@gmail.com
@Org     :   Kasthuri Lab, University of Chicago
'''

import lzma
import pickle
import numpy as np

# Given two image masks, create the overlap mask in common coordinates
def overlap_img_pair(img_filepath_i, min_coord_i, img_filepath_j, min_coord_j, min_coord_ij, max_coord_ij, output_filepath):
  with lzma.open(img_filepath_i, "rb") as f:
    img_i = pickle.load(f)
    
  with lzma.open(img_filepath_j, "rb") as f:
    img_j = pickle.load(f)
  
  min_coord_i_crop = min_coord_ij - min_coord_i
  max_coord_i_crop = max_coord_ij - min_coord_i
  min_coord_j_crop = min_coord_ij - min_coord_j
  max_coord_j_crop = max_coord_ij - min_coord_j
  
  img_i_crop = img_i[min_coord_i_crop.z:max_coord_i_crop.z+1, min_coord_i_crop.y:max_coord_i_crop.y+1, min_coord_i_crop.x:max_coord_i_crop.x+1].astype(bool)
  img_j_crop = img_j[min_coord_j_crop.z:max_coord_j_crop.z+1, min_coord_j_crop.y:max_coord_j_crop.y+1, min_coord_j_crop.x:max_coord_j_crop.x+1].astype(bool)
  
  overlap_img = np.logical_and(img_i_crop, img_j_crop)
  
  # Save overlapped image mask. It is reduced to uint8 since we just have a bitmask at this stage
  with lzma.open(output_filepath, "wb") as f:
    pickle.dump(overlap_img, f)
      
  return overlap_img