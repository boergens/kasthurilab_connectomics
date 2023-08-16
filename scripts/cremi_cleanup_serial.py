import os
import sys
import numpy as np
import pickle
import lzma
import matplotlib.pyplot as plt
import csv
import h5py
from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure
sys.path.insert(0, os.path.abspath('/home/suryakalia/documents/summer/exploration/kasthurilab_connectomics/'))
# Need to add above path since VSCode Jupyter Notebook doesn't respect system's $PYTHONPATH variable
# This will be eliminated once my module is converted to a conda package and installed to the conda env

from sk_connectomics.util.visualize import *

f = h5py.File("/home/suryakalia/documents/summer/datasets/cremi/sample_A_20160501.hdf", 'r')
# neuron_ids = np.array(f["volumes/labels/neuron_ids"])
clefts = np.array(f["volumes/labels/clefts"])

def process_segment(img, segment_id, segment_volume, min_fraction_to_split, max_label):
  binary_mask = np.where(img == segment_id , 1, 0).astype(np.uint8)
  # 26-neighbor connectivity in 3D
  labeled_mask, num_features = label(binary_mask, structure=np.ones((3,3,3)))
  if (not num_features == 1):
    # Multiple components found. Check for splits/patches
    print("Incorrect segmentation detected for segment id =", segment_id, "volume = ", segment_volume, "num_features =", num_features)
    (unique, counts) = np.unique(labeled_mask, return_counts=True)
    segment_volume_map = dict(zip(unique, counts))
    sorted_segment_volume_map = dict(sorted(segment_volume_map.items(), key=lambda item: item[1], reverse = True))
    largest = True
    for id, vol in sorted_segment_volume_map.items():
      if id == 0:
        # Ignore background segment
        continue
      if largest:
        # Set original segment_id to the largest component
        labeled_mask = np.where(labeled_mask == id, segment_id, labeled_mask)
        largest = False
      else:
        # Check if segment constitutes a significant fraction of the parent segment
        if vol/segment_volume >= min_fraction_to_split :
          # Got a valid segment. Provide a new label
          max_label += 1
          print("Found a split in segmentation. New split segment_id, vol = ", max_label, ",", vol, "volume ratio = ", vol/segment_volume)
          labeled_mask = np.where(labeled_mask == id, max_label, labeled_mask)
        else:
          # Got a small splinter. Eliminate segment
          print("Eliminated splinter of size ratio: ", vol/segment_volume)
          labeled_mask = np.where(labeled_mask == id, 0, labeled_mask)
    
    return labeled_mask, max_label

  else:
    # Correct segmentation. All part of one connected component
    mask = segment_id * binary_mask
    return mask, max_label

def cleanup_segmentation_image(img, vol_threshold, min_fraction_to_split):
  clean_img = np.zeros(img.shape)
  # Eliminate any background pixels
  img = np.where(img==(2**64 - 1), 0, img)
  # Track current max label for creating new labels if needed
  max_label = np.max(img)
  (unique, counts) = np.unique(img, return_counts=True)
  for i in tqdm(range(len(unique))):
    # Ignore background segments. Can be splintered
    if (unique[i] == 0):
      continue
    # Process segment only if above threshold
    if (counts[i] >= vol_threshold):
      mask, max_label = process_segment(img, unique[i], counts[i], min_fraction_to_split, max_label)
      clean_img += mask
      
      
  
  return clean_img

def main(): 
  # clean_neuron_ids = cleanup_segmentation_image(neuron_ids, 100, 0.1)
  clean_clefts = cleanup_segmentation_image(clefts, 100, 0.1)
  
  with lzma.open("/home/suryakalia/documents/summer/datasets/cremi_clean/" + "clean_clefts.xz", "wb") as f:
    pickle.dump(clean_clefts, f)

if __name__ == '__main__':
  main()