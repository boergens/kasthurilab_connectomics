import os
import sys
import numpy as np
import pickle
import lzma
import matplotlib.pyplot as plt
import csv
import h5py
from tqdm import tqdm
import multiprocessing
from scipy.ndimage import label, generate_binary_structure
sys.path.insert(0, os.path.abspath('/home/suryakalia/documents/summer/exploration/kasthurilab_connectomics/'))
# Need to add above path since VSCode Jupyter Notebook doesn't respect system's $PYTHONPATH variable
# This will be eliminated once my module is converted to a conda package and installed to the conda env

from sk_connectomics.util.visualize import *


f = h5py.File("/home/suryakalia/documents/summer/datasets/cremi/sample_A_20160501.hdf", 'r')
neuron_ids = np.array(f["volumes/labels/neuron_ids"])
clefts = np.array(f["volumes/labels/clefts"])

def process_segment(img, segment_id, segment_volume, min_fraction_to_split, max_label, num_procs):
  # print("Entered process segment for segment = ", segment_id)
  binary_mask = np.where(img == segment_id , 1, 0).astype(np.uint8)
  # print("Binary mask created for segment = ", segment_id)
  # 26-neighbor connectivity in 3D
  labeled_mask, num_features = label(binary_mask, structure=np.ones((3,3,3)))
  # print("Labeled Mask type:", labeled_mask.dtype)
  # print("Generated labels for segment = ", segment_id)
  if (not num_features == 1):
    # Multiple components found. Check for splits/patches
    # print("Incorrect segmentation detected for segment id =", segment_id, "volume = ", segment_volume, "num_features =", num_features)
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
        # print("Labeled Mask type after first where:", labeled_mask.dtype)
        # print("Replaced existing id", id, " with segmentation id =", segment_id)
        
        largest = False
      else:
        # Check if segment constitutes a significant fraction of the parent segment
        if vol/segment_volume >= min_fraction_to_split :
          # Got a valid segment. Provide a new label
          max_label += num_procs
          # print("Found a split in segmentation. New split segment_id, vol = ", max_label, ",", vol, "volume ratio = ", vol/segment_volume)
          labeled_mask = np.where(labeled_mask == id, max_label, labeled_mask)
          # print("Labeled Mask type after second where:", labeled_mask.dtype)
          
        else:
          # Got a small splinter. Eliminate segment
          # print("Eliminated splinter of size ratio: ", vol/segment_volume)
          labeled_mask = np.where(labeled_mask == id, 0, labeled_mask)
          # print("Labeled Mask type after third where:", labeled_mask.dtype)
          
    # print("Labeled Mask type before returning:", labeled_mask.dtype)
    return labeled_mask.astype(np.uint32), max_label

  else:
    # print("Correct segmentation detected. Returning mask for segment_id = ", segment_id)
    # Correct segmentation. All part of one connected component
    mask = segment_id * binary_mask
    # print("Binary mask type: ", binary_mask.dtype)
    # print("Mask type after multiplication:", labeled_mask.dtype)
    
    return mask.astype(np.uint32), max_label
  

def init_pool(img, vol_threshold, min_fraction_to_split, max_label, num_procs):
  global p_img
  global p_vol_threshold
  global p_min_fraction_to_split
  global p_num_procs
  # global p_queue
  global p_id
  global p_max_label
  
  p_img = img
  p_vol_threshold = vol_threshold
  p_min_fraction_to_split = min_fraction_to_split
  p_num_procs = num_procs
  # p_queue = queue
  p_id = multiprocessing.current_process()._identity[0]
  p_max_label = int(max_label + (p_id % p_num_procs))
  print("Initialized process id = ", p_id, "max label = ", p_max_label)
  
  

def parallel_process_segment_wrapper(tuple_list):
  global p_max_label
  # print("Received task list = ", tuple_list, " by pid = ", p_id)
  # print("Received task list of length = ", len(tuple_list))
  # print("p_num_proc = ", p_num_procs)
  # print("p_max_label = ", p_max_label)
  
  clean_img = np.zeros(p_img.shape).astype(np.uint32)
  for tuple in tqdm(tuple_list):
    segmentation_id, vol = tuple
    if (segmentation_id == 0):
      continue
    # print("processing instance", tuple)
    # print("process id", multiprocessing.current_process()._identity[0])
    # print("img shape = ", p_img.shape)
    # print(p_img[0][0])
    # p_img[0][0] =  multiprocessing.current_process()._identity[0]
    mask, p_max_label = process_segment(p_img, segmentation_id, vol, p_min_fraction_to_split, p_max_label, p_num_procs)
    # print("finished processing instance", tuple, " mask min = ", np.min(mask), " max = ", np.max(mask))

    print("PID =", p_id, "Finished processing segmentation_id =", segmentation_id)
    clean_img += mask
  # print("Returning image by pid = ", p_id)
  # p_queue.put(clean_img)
  return clean_img
  # p_queue.put(len(tuple_list))
  # return len(tuple_list)
  

def parallel_cleanup_segmentation_image(img, vol_threshold, min_fraction_to_split, num_procs, num_chunks):
  clean_img = np.zeros(img.shape).astype(np.uint32)
  # Eliminate any background pixels
  img = np.where(img==(2**64 - 1), 0, img)
  # Track current max label for creating new labels if needed
  max_label = np.max(img)
  (unique, counts) = np.unique(img, return_counts=True)
  
  queue = multiprocessing.Queue()
  
  pool = multiprocessing.Pool(num_procs, initializer=init_pool, initargs=(img, vol_threshold, min_fraction_to_split, max_label, num_procs))
  
  print("Original task list len = ", len(unique))
  
  task_list = list(zip(unique,counts))
  
  print("Task list = ", task_list)
  print("Len Task list = ", len(task_list))
  
  chunk_size = int(len(task_list)/num_chunks) + 1
  splitted_task_list = [task_list[x:x+chunk_size] for x in range(0, len(task_list), chunk_size)]

  num_chunks = len(splitted_task_list)
  
  print("Splitted task list = ", splitted_task_list)
  print("Len Splitted task list = ", len(splitted_task_list))
  
  results = pool.map(parallel_process_segment_wrapper, splitted_task_list)
  print("Results size = ", len(results))
  print("Results = ", results)
  # pool.join()
  # pool.close()
  print("Finished with multiprocessing pool")
  
  for result in results:
    clean_img += result
  # for i in range(num_chunks):
  #   print("Popped image from queue")
  #   print(queue.get())
  # for i in tqdm(range(len(unique))):
  #   # Ignore background segments. Can be splintered
  #   if (unique[i] == 0):
  #     continue
  #   # Process segment only if above threshold
  #   if (counts[i] >= vol_threshold):
  #     mask, max_label = process_segment(img, unique[i], counts[i], min_fraction_to_split, max_label)
  #     clean_img += mask
      
  print("End of processing")
  
  return clean_img



clean_neuron_ids = parallel_cleanup_segmentation_image(neuron_ids, 100, 0.1, 10, 10)
with lzma.open("/home/suryakalia/documents/summer/datasets/cremi_clean/" + "clean_neuron_ids.xz", "wb") as f:
    pickle.dump(clean_neuron_ids, f)
    
    
print(clean_neuron_ids)
print(np.max(clean_neuron_ids))
print(np.min(clean_neuron_ids))

plt.imsave("clean_neuron_ids_slice_0.png", clean_neuron_ids[0,:,:], cmap='gray', vmin=0, vmax=np.max(clean_neuron_ids))
plt.imsave("clean_neuron_ids_slice_62.png", clean_neuron_ids[62,:,:], cmap='gray', vmin=0, vmax=np.max(clean_neuron_ids))
plt.imsave("clean_neuron_ids_slice_124.png", clean_neuron_ids[124,:,:], cmap='gray', vmin=0, vmax=np.max(clean_neuron_ids))


