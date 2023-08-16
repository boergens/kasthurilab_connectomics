'''
@File    :   visualize.py
@Time    :   2023/08/03 11:13:15
@Author  :   Surya Chandra Kalia
@Version :   1.0
@Contact :   suryackalia@gmail.com
@Org     :   Kasthuri Lab, University of Chicago
'''

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Given a list of new segment ids, fetch their original segment ids and create a mask of union of the listed segments
def get_segment_mask(new_id_list, data_dir, h5_file_name):
  with open(data_dir + "/neuron_id_mapping.json", "r") as json_file:
    neuron_id_old_new_map = json.load(json_file)
  
  neuron_id_new_old_map = {int(v):int(k) for k,v in neuron_id_old_new_map.items()}

  old_id_list = []
  for id in new_id_list:
    # Check if it is a valid ID
    if id in neuron_id_new_old_map:
      old_id_list.append(neuron_id_new_old_map[id])
    else:
      # Invalid ID. Skip for now
      print("ERROR: Invalid segmentation id provided: ", id, " skipping visualization")
  
  print("Mapped to following old ids: ", old_id_list)
  
  hdf_handle = h5py.File(data_dir + "/" + h5_file_name, 'r')
  neuron_ids = np.array(hdf_handle["volumes/labels/neuron_ids"])
  
  mask = neuron_ids == old_id_list[0]
  
  for i in range (1, len(old_id_list)):
    mask = mask | (neuron_ids == old_id_list[i])
  
  return np.where(mask, 1, 0).astype(np.uint8)

# Given a segment mask, visualize the best layer with largest cross sectional area
def get_best_slice(mask):
  neuron_layer_sum = np.sum(mask, axis=(1,2))
  print("Layer Sums: ", neuron_layer_sum)
  print("Layer with largest cross sectional area: ", neuron_layer_sum.argmax())
  plt.imshow(mask[neuron_layer_sum.argmax() , :, :])