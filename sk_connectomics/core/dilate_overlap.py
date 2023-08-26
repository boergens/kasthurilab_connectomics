'''
@File    :   dilate_overlap.py
@Time    :   2023/06/28 11:19:09
@Author  :   Surya Chandra Kalia
@Version :   1.0
@Contact :   suryackalia@gmail.com
@Org     :   Kasthuri Lab, University of Chicago
'''

import h5py
import numpy as np
from sk_connectomics.util.coordinate import *
from sk_connectomics.util.overlap import *
import csv  
import os
import lzma
import pickle
from scipy import ndimage
import glob
import csv
from tqdm import tqdm
import tifffile
from multiprocessing import Pool
import pandas as pd
from PIL import Image
import json
from sklearn.metrics import precision_recall_fscore_support


# Given a segmentation map (3D), we will dilate the neuron boundaries and identify overlap between neuron pairs.
# Step-by-step control flow:
# 1. Read the dense segmentation map into memory. Each neuron is identified with a unique ID.
# 2. For each neuron ID, identify a bounding box capturing the min and max x,y,z coordinates.
# 3. Include padding and crop out each neuron segment as separate images. Keep a track of origin offset for each ID in a consolidated txt file as well.
#  This will be useful for reconstructing the segments back to global coordinates.
# 4. Dilate each segment by provided voxel amount.
# 5. Eliminate glea cells (having very large voxel counts) and background segments (having very small voxel counts)
# 6. Run n*n loop over all neuron pairs to identify overlapping regions. Most can be trivially eliminated by seeing bounding box overlap. 
#  The others will have to be idetified by croppnig out the bounding box intersection, performing a logical AND op, and storing the region of overlap

class DilateOverlap:
  def __init__(self, cremi_file_path, output_dir, dilation_voxel_count, voxel_volume_threshold, num_cores, pred_path = None):
    self.cremi_file_path = cremi_file_path
    self.output_dir = output_dir
    self.hdf_handle = h5py.File(cremi_file_path, 'r')
    self.neuron_ids = np.array(self.hdf_handle["volumes/labels/neuron_ids"])
    self.hdf_handle = h5py.File(cremi_file_path, 'r')
    self.clefts = np.array(self.hdf_handle["volumes/labels/clefts"])
    self.pred = None
    if pred_path is not None:
      with lzma.open(pred_path, "rb") as f:
        self.pred = pickle.load(f)
        
    # Clean out background voxels of cleft
    self.clefts[self.clefts == 18446744073709551615] = 0
    
    # # DEBUG: Use thin slice of input
    # self.neuron_ids = self.neuron_ids[:5, :, :]
    self.segment_black_list = []
    self.segment_volume_map = {}
    self.dilation_voxel_count = dilation_voxel_count
    self.voxel_volume_threshold = voxel_volume_threshold
    self.num_cores = num_cores
  
  def create_bounding_boxes(self, img):
    # Scan through whole image and keep track of min/max x, y, z values of each segmentID
    minCoordMap = {}
    maxCoordMap = {}
    dims = img.shape
    for z in tqdm(range(dims[0])):
      for y in range (dims[1]):
        for x in range (dims[2]):
          id = img[z][y][x]
          if (id not in minCoordMap):
            # New ID encountered
            minCoordMap[id] = Coordinate(x, y, z)
            maxCoordMap[id] = Coordinate(x, y, z)
          else:
            # Update the min/max values for the ID
            minCoordMap[id] = minCoord(minCoordMap[id], Coordinate(x, y, z))
            maxCoordMap[id] = maxCoord(maxCoordMap[id], Coordinate(x, y, z))

    return minCoordMap, maxCoordMap
  
  def create_neuron_bounding_boxes(self):
    self.min_coord_map_neuron, self.max_coord_map_neuron = self.create_bounding_boxes(self.neuron_ids)
    
  def create_cleft_bounding_boxes(self):
    self.min_coord_map_cleft, self.max_coord_map_cleft = self.create_bounding_boxes(self.clefts)
  
  def create_pred_bounding_boxes(self):
    self.min_coord_map_pred, self.max_coord_map_pred = self.create_bounding_boxes(self.pred)
  
  # Add list of segment IDs which should be treated as background voxels
  def blacklist_append(self, segment_list):
    self.segment_black_list.extend(segment_list)
  
  # Clear segment ID blacklist 
  def blacklist_clear(self):
    self.segment_black_list = []
  
  # Eliminate all segments smaller than voxel_threshold and previously blacklisted segment IDs
  def trim_invalid_segments(self):
    print("Trimming invalid segments")
    # Create a segmentId to volume map
    (unique, counts) = np.unique(self.neuron_ids, return_counts=True)
    for i in range (len(unique)):
      self.segment_volume_map[unique[i]] = counts[i]
    
    # To simplify the process, set blacklisted segments to have -1 voxel count. Will be eliminated by the voxel threshold
    for segmentID in self.segment_black_list:
      self.segment_volume_map[segmentID] = -1
      
    # Eliminate segments from input image with volume lesser than voxel_volume_threshold. Zeroing out all these segments in the raw input will be computationally expensive as we will have 
    # to incur a lookup cost for each voxel when iterating, and also modifying the data. Instead, we will jult eliminate the corresponding bounding boxes
    for segmentID in list(self.segment_volume_map.keys()) : 
      if (self.segment_volume_map[segmentID] < self.voxel_volume_threshold):
        del self.segment_volume_map[segmentID] 
        del self.min_coord_map_neuron[segmentID]
        del self.max_coord_map_neuron[segmentID]
      
  # Given a list of trimmed bounding boxes, crop out the 3D segment masks and store as separate images, along with metadata 
  def crop_out_bounding_boxes(self, input_mode):
    print("Cropping out bounding boxes")
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    if not os.path.exists(self.output_dir + "/" + input_mode + "_crops"):
      os.makedirs(self.output_dir + "/" + input_mode + "_crops")
      
    # Clear out directory to avoid mixing up with old crops
    files = glob.glob(self.output_dir + "/" + input_mode + "_crops/*")
    for f in files:
        os.remove(f)
    
    # Pick img based on input mode
    if (input_mode == "neuron"):
      img = self.neuron_ids
    elif (input_mode == "cleft"):
      img = self.clefts
    else:
      img = self.pred
    
    # Reverse sort by segment volume. n*n loop for img overlap will be more efficient if larger images are loaded in memory less often
    if (input_mode == "neuron"):
      # Use segment volume map that was created earlier after trimming
      sorted_segment_volume_map = dict(sorted(self.segment_volume_map.items(), key = lambda x:x[1], reverse = True))
      # If neuron segmentation isn't dense, we need to eliminate the background voxels here as well.
      del(sorted_segment_volume_map[0])
    else:
      # Create a segmentId to volume map
      (unique, counts) = np.unique(img, return_counts=True)
      print(unique, counts)
      segment_volume_map = {}
      for i in range (len(unique)):
        segment_volume_map[unique[i]] = counts[i]
      sorted_segment_volume_map = dict(sorted(segment_volume_map.items(), key = lambda x:x[1], reverse = True))
      # Remove background segment from clefts/pred
      del(sorted_segment_volume_map[0])
      
    print("Sorted segment volume map: ", sorted_segment_volume_map )
    
    crop_num = 1
    header = ['SEGMENT_ID', 'VOLUME', 'MIN_X', 'MIN_Y', 'MIN_Z', 'MAX_X', 'MAX_Y', 'MAX_Z']
    with open(self.output_dir + "/metadata_" + input_mode + "_crop.csv", 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      # write the header
      writer.writerow(header)
      for segmentID, volume in tqdm(sorted_segment_volume_map.items()):
        
        if (input_mode == "neuron"):
          minCoord = self.min_coord_map_neuron[segmentID]
          maxCoord = self.max_coord_map_neuron[segmentID]
          # Ensure that we crop out a larger bounding box to account for new size post dilation.
          minCoord = dilateMin(minCoord, self.dilation_voxel_count, 1, img.shape)
          maxCoord = dilateMax(maxCoord, self.dilation_voxel_count, 1, img.shape)
        elif (input_mode == "cleft"):
          # No dilation required for cleft and pred
          minCoord = self.min_coord_map_cleft[segmentID]
          maxCoord = self.max_coord_map_cleft[segmentID]
        else:
          minCoord = self.min_coord_map_pred[segmentID]
          maxCoord = self.max_coord_map_pred[segmentID]
          
          
        # Crop out bounding box and mask out all voxels not belonging to the given segment
        img_crop = img[minCoord.z:maxCoord.z+1, minCoord.y:maxCoord.y+1, minCoord.x:maxCoord.x+1]
        img_crop = np.where(img_crop == segmentID , 1, 0).astype(np.uint8)
        
        if (input_mode == "neuron"):
          # Dilate image_crop. For anisotropic dilation we need 1 3-D dilaiton and n-1 2-D dilations in X-Y plane
          
          img_crop = ndimage.binary_dilation(img_crop).astype(img_crop.dtype)
          
          if (self.dilation_voxel_count > 1):
            for z in range(img_crop.shape[0]):
              img_crop[z, :, :] = ndimage.binary_dilation(img_crop[z, :, :], iterations=self.dilation_voxel_count-1).astype(img_crop.dtype)
        
        # Save cropped image mask. It is reduced to uint8 since we just have a bitmask at this stage
        with lzma.open(self.output_dir + "/" + input_mode + "_crops/" + str(segmentID) + ".xz", "wb") as f:
          pickle.dump(img_crop, f)
        
        # Save metadata of img_crop for each segmentID
        data = [segmentID, volume, minCoord.x, minCoord.y, minCoord.z, maxCoord.x, maxCoord.y, maxCoord.z]
        writer.writerow(data)
        crop_num+=1

    print("Successfully created ", crop_num, " segment corps")
    
  def find_overlaping_neuron_segments(self):
    if not os.path.exists(self.output_dir+"/neuron_overlaps"):
      os.makedirs(self.output_dir+"/neuron_overlaps")
      
    # Clear out directory to avoid mixing up with old overlaps
    files = glob.glob(self.output_dir+"/neuron_overlaps/*")
    for f in files:
      os.remove(f)
    
    # Read list of crops using metadata file
    with open(self.output_dir + "/metadata_neuron_crop.csv", newline='') as f:
      reader = csv.reader(f)
      crop_list = list(reader)[1:]
    
    print("Number of crops read = ", len(crop_list))
    
    
    print("Calculating segment overlaps")
    overlap_count = 0
    header = ['SEGMENT_ID_1', 'SEGMENT_ID_2', 'MIN_X', 'MIN_Y', 'MIN_Z', 'MAX_X', 'MAX_Y', 'MAX_Z']
    with open(self.output_dir + "/metadata_neuron_overlap.csv", 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      # Iterate over all possible overlap combinations  
      for i in tqdm(range(len(crop_list)-1)):
        min_coord_i = Coordinate(crop_list[i][2], crop_list[i][3], crop_list[i][4])
        max_coord_i = Coordinate(crop_list[i][5], crop_list[i][6], crop_list[i][7])
        for j in range(i+1, len(crop_list)):
          min_coord_j = Coordinate(crop_list[j][2], crop_list[j][3], crop_list[j][4])
          max_coord_j = Coordinate(crop_list[j][5], crop_list[j][6], crop_list[j][7])
          overlap_found, min_coord_ij, max_coord_ij = intersection_crop(min_coord_i, max_coord_i, min_coord_j, max_coord_j)
          if (overlap_found):
            # Save metadata of overlap IDs, volume and coordinated
            data = [crop_list[i][0], crop_list[j][0], min_coord_ij.x, min_coord_ij.y, min_coord_ij.z, max_coord_ij.x, max_coord_ij.y, max_coord_ij.z]
            writer.writerow(data)
            overlap_count += 1
            
    print("Num neuron overlap instances = ", overlap_count)
    
  
  def find_overlaping_neuron_synapse_segments(self, input_mode):
    if not os.path.exists(self.output_dir+"/" + input_mode + "_overlaps"):
      os.makedirs(self.output_dir+"/" + input_mode + "_overlaps")
      
    # Clear out directory to avoid mixing up with old overlaps
    files = glob.glob(self.output_dir+"/" + input_mode + "_overlaps/*")
    for f in files:
      os.remove(f)
    
    # Read list of crops using metadata file
    with open(self.output_dir + "/metadata_neuron_overlap.csv", newline='') as f:
      reader = csv.reader(f)
      neuron_metadata = list(reader)[1:]
    
    with open(self.output_dir + "/metadata_" + input_mode + "_crop.csv", newline='') as f:
      reader = csv.reader(f)
      synapse_metadata = list(reader)[1:]
    
    print("Number of neuron crops read = ", len(neuron_metadata))
    print("Number of synapse crops read = ", len(synapse_metadata))
    
    print("Calculating neuron and " + input_mode + " synapse overlaps")
    overlap_count = 0
    header = ['SEGMENT_ID_1', 'SEGMENT_ID_2', 'MIN_X', 'MIN_Y', 'MIN_Z', 'MAX_X', 'MAX_Y', 'MAX_Z']
    with open(self.output_dir + "/metadata_" + input_mode + "_overlap.csv", 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      # Iterate over all possible overlap combinations
      # for i in tqdm(range(len(crop_list)-1)):
      for i in tqdm(range(len(neuron_metadata))):
        # print("index",index)
        # print("row",row)
        min_coord_i = Coordinate(neuron_metadata[i][2], neuron_metadata[i][3], neuron_metadata[i][4])
        max_coord_i = Coordinate(neuron_metadata[i][5], neuron_metadata[i][6], neuron_metadata[i][7])
        for j  in range(len(synapse_metadata)):
          min_coord_j = Coordinate(synapse_metadata[j][2], synapse_metadata[j][3], synapse_metadata[j][4])
          max_coord_j = Coordinate(synapse_metadata[j][5], synapse_metadata[j][6], synapse_metadata[j][7])
          overlap_found, min_coord_ij, max_coord_ij = intersection_crop(min_coord_i, max_coord_i, min_coord_j, max_coord_j)
          if (overlap_found):
            # Save metadata of overlap IDs, volume and coordinated
            data = [neuron_metadata[i][0] + "_" + neuron_metadata[i][1], synapse_metadata[j][0], min_coord_ij.x, min_coord_ij.y, min_coord_ij.z, max_coord_ij.x, max_coord_ij.y, max_coord_ij.z]
            writer.writerow(data)
            overlap_count += 1
            
    print("Num neuron " + input_mode + " synapse overlap instances = ", overlap_count)

  # Iterate over the metadata_overlap.csv list to overlap the segments. Parallelized iteration to speed up overlap process 
  def overlap_all_segment_pairs(self, input_mode):
    
    if (input_mode == "neuron"):
      # Read list of crops for retrieving their coordinates
      with open(self.output_dir + "/metadata_neuron_crop.csv", newline='') as f:
        reader = csv.reader(f)
        neuron_crop_list = list(reader)[1:]
      
      neuron_crop_coord_map = {}
      for row in neuron_crop_list:
        neuron_crop_coord_map[row[0]] = [Coordinate(row[2], row[3], row[4]), Coordinate(row[5], row[6], row[7])]
      
    else:
      # Read list of neuron overlaps and synapse srops for retrieving their coordinates
      with open(self.output_dir + "/metadata_neuron_overlap.csv", newline='') as f:
        reader = csv.reader(f)
        neuron_overlap_list = list(reader)[1:]
      
      with open(self.output_dir + "/metadata_" + input_mode + "_crop.csv", newline='') as f:
        reader = csv.reader(f)
        synapse_crop_list = list(reader)[1:]
        
      neuron_overlap_coord_map = {}
      for row in neuron_overlap_list:
        neuron_overlap_coord_map[row[0]+"_"+row[1]] = [Coordinate(row[2], row[3], row[4]), Coordinate(row[5], row[6], row[7])]
      
      synapse_crop_coord_map = {}
      for row in synapse_crop_list:
        synapse_crop_coord_map[row[0]] = [Coordinate(row[2], row[3], row[4]), Coordinate(row[5], row[6], row[7])]
    
    
    
    
    # Read list of overlaps using metadata file
    with open(self.output_dir + "/metadata_" + input_mode + "_overlap.csv", newline='') as f:
      reader = csv.reader(f)
      overlap_list = list(reader)[1:]
    
    print(len(overlap_list))
    print(overlap_list[0])
    # print(len(neuron_crop_list))
    # print(neuron_crop_list[0])

  
    # Create args list for multiprocessing pool
    args_list = []
    for overlap_candidate in overlap_list:
      seg_id_i = overlap_candidate[0]
      seg_id_j = overlap_candidate[1]
      min_coord_ij = Coordinate(overlap_candidate[2],overlap_candidate[3],overlap_candidate[4])
      max_coord_ij = Coordinate(overlap_candidate[5],overlap_candidate[6],overlap_candidate[7])
      if (input_mode == "neuron"):
        min_coord_i = neuron_crop_coord_map[seg_id_i][0]
        min_coord_j = neuron_crop_coord_map[seg_id_j][0]
        img_filepath_i = self.output_dir + "/neuron_crops/" + str(seg_id_i) + ".xz"
        img_filepath_j = self.output_dir + "/" + input_mode + "_crops/" + str(seg_id_j) + ".xz"
      else:
        min_coord_i = neuron_overlap_coord_map[seg_id_i][0]
        min_coord_j = synapse_crop_coord_map[seg_id_j][0]
        img_filepath_i = self.output_dir + "/neuron_overlaps/" + str(seg_id_i) + ".xz"
        img_filepath_j = self.output_dir + "/" + input_mode + "_crops/" + str(seg_id_j) + ".xz"
      
      output_filepath = self.output_dir + "/" + input_mode + "_overlaps/" + str(seg_id_i) + "_" + str(seg_id_j) + ".xz"
      args_list.append([img_filepath_i, min_coord_i, img_filepath_j, min_coord_j, min_coord_ij, max_coord_ij, output_filepath])
      
    print(len(args_list))
    print(args_list[0])
    # pool = Pool(os.cpu_count())
    # DEBUG:
    pool = Pool(self.num_cores)
    print("Created multiprocessing pool with cpu count = ", self.num_cores)
    pool.starmap(overlap_img_pair, args_list)
    print("Finished Pool processing")
    
    
  def construct_full_overlap_mask(self, input_mode):
    # Read list of overlaps using metadata file
    with open(self.output_dir + "/metadata_" + input_mode + "_overlap.csv", newline='') as f:
        reader = csv.reader(f)
        overlap_list = list(reader)
        
    print("Merging all overlapping segments into a common image")
    overlap_img_combined = np.zeros(self.neuron_ids.shape).astype(bool)
    
    header = ['SEGMENT_ID_1', 'SEGMENT_ID_2', 'MIN_X', 'MIN_Y', 'MIN_Z', 'MAX_X', 'MAX_Y', 'MAX_Z', 'VOLUME']
    with open(self.output_dir + "/metadata_" + input_mode + "_overlap_volume.csv", 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      # Iterate over all overlaps and merge them into one image
      for k in tqdm(range(1, len(overlap_list))):
        segment_id_i = overlap_list[k][0]
        segment_id_j = overlap_list[k][1]
        min_x = int(overlap_list[k][2])
        min_y = int(overlap_list[k][3])
        min_z = int(overlap_list[k][4])
        max_x = int(overlap_list[k][5])
        max_y = int(overlap_list[k][6])
        max_z = int(overlap_list[k][7])
        
        overlap_file_path = self.output_dir + "/" + input_mode + "_overlaps/" + str(segment_id_i) + "_" + str(segment_id_j) + ".xz"
        with lzma.open(overlap_file_path, "rb") as f:
          overlap_img = pickle.load(f)
        
        # Calculate volume of overlap by counting the non-zero voxels and storing that sum in metadata
        overlap_volume_sum = np.sum(overlap_img)
        new_metadata_row = overlap_list[k]
        new_metadata_row.append(overlap_volume_sum)
        writer.writerow(new_metadata_row)
        
        overlap_img_combined[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = np.logical_or(overlap_img_combined[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1], overlap_img)

      
    # Convert overlap image file to uint8
    overlap_img_combined = overlap_img_combined.astype(np.uint8)
    
    # Save combained overlap image mask as both numpy array and TIFF stack
    with lzma.open(self.output_dir + "/overlap_" + input_mode + "_img_combined" + ".xz", "wb") as f:
      pickle.dump(overlap_img_combined, f)
      
      tifffile.imsave(self.output_dir + "/overlap_" + input_mode + "_img_combined" + ".tiff", overlap_img_combined)
  
  def connectivity_matrix(self, input_mode):
    # 1. Read neuron IDs and map them to a contiguous range
    # 2. Create adjacency matrix representing connection counts
    # 3. Read synapse overlaps
    # 4. Iterating over the non-zero volume synapse overlaps, update the adjacency matrix
    
    # Read list of crops for retrieving their coordinates
    with open(self.output_dir + "/metadata_neuron_crop.csv", newline='') as f:
      reader = csv.reader(f)
      neuron_crop_list = list(reader)[1:]
    
    neuron_id_map = {}
    count = 0
    for row in neuron_crop_list :
      neuron_id_map[int(row[0])] = count
      count += 1
    
    adj_matrix = np.zeros((count, count)).astype(np.uint8)
    
    # Read cleft/pred overlap metadata
    with open(self.output_dir + "/metadata_" + input_mode + "_overlap_volume.csv", newline='') as f:
      reader = csv.reader(f)
      overlap_list = list(reader)[1:]

    # Iterate through synapse connections and populate adjacency matrix
    for row in overlap_list:
      overlap_volume = int(row[8])
      if (overlap_volume == 0):
        continue
      neuron_pair = row[0].split('_')
      neuron_id_1 = neuron_id_map[int(neuron_pair[0])]
      neuron_id_2 = neuron_id_map[int(neuron_pair[1])]
      adj_matrix[neuron_id_1, neuron_id_2] += 1
      adj_matrix[neuron_id_2, neuron_id_1] += 1
    
    print("Matrix min, max = ", adj_matrix.min(), " ", adj_matrix.max())
    
    adj_matrix_sum = adj_matrix.sum(axis=1)
    
    print("Max connections for single process = ", adj_matrix_sum.max(), " At new ID = ", np.argmax(adj_matrix_sum), " Old ID = ", [key for key, val in neuron_id_map.items() if val == np.argmax(adj_matrix_sum)] )
    
    # Save the neuron ID mappings
    with open(self.output_dir + '/neuron_id_mapping.json', 'w') as f:
      json.dump(neuron_id_map, f)
    
    # Save adjacency matrix
    adj_img = Image.fromarray(adj_matrix)
    adj_img.save(self.output_dir + "/adjacency_matrix_" + input_mode + ".png")
    
    # Save scaled adjacency matrix to better visualize matrix since values are going to be very small
    scaled_adj_matrix = adj_matrix.astype(np.double)
    scaled_adj_matrix = ((scaled_adj_matrix - scaled_adj_matrix.min()) * (1/(scaled_adj_matrix.max() - scaled_adj_matrix.min()) * 255)).astype('uint8')
    scaled_adj_img = Image.fromarray(scaled_adj_matrix)
    scaled_adj_img.save(self.output_dir + "/scaled_adjacency_matrix_" + input_mode + ".png")
    
    # Inspect only non-zero rows/cols
    mask = np.nonzero(np.sum(scaled_adj_matrix, axis = 1))[0]
    non_zero_scaled_adj_matrix = scaled_adj_matrix[np.ix_(mask, mask)]
    non_zero_scaled_adj_img = Image.fromarray(non_zero_scaled_adj_matrix)
    non_zero_scaled_adj_img.save(self.output_dir + "/non_zero_scaled_adjacency_matrix_" + input_mode + ".png")
    
  
  def calc_f1_score(self):
    # 1. Binarize connectivity matrices
    # 2. Extract upper triangular matrix from connectivity matrix to include each segment pair only once
    # 3. Calculate precision, recall, f1 score
    
    cleft_adj_matrix = np.array(Image.open(self.output_dir + "/adjacency_matrix_cleft.png"))
    pred_adj_matrix = np.array(Image.open(self.output_dir + "/adjacency_matrix_pred.png"))
    
    cleft_adj_matrix = np.where(cleft_adj_matrix>0 , 1, 0)
    pred_adj_matrix = np.where(pred_adj_matrix>0 , 1, 0)
    
    assert cleft_adj_matrix.shape[0] == cleft_adj_matrix.shape[1]
    assert pred_adj_matrix.shape[0] == pred_adj_matrix.shape[1]
    assert cleft_adj_matrix.shape[0] == pred_adj_matrix.shape[0]
    
    n = cleft_adj_matrix.shape[0]
    print("Connectivity Matrix Size = ", n)
    
    cleft_pairs = cleft_adj_matrix[np.triu_indices(n)]
    pred_pairs = pred_adj_matrix[np.triu_indices(n)]
    
    precision, recall, f1, support = precision_recall_fscore_support(cleft_pairs, pred_pairs, average="binary")
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 score = ", f1)
    print("Support = ", support)
    
    
  def run(self):
    self.create_neuron_bounding_boxes()
    self.trim_invalid_segments()
    self.crop_out_bounding_boxes("neuron")
    self.find_overlaping_neuron_segments()
    self.overlap_all_segment_pairs("neuron")
    self.construct_full_overlap_mask("neuron")
    
    self.create_cleft_bounding_boxes()
    self.crop_out_bounding_boxes("cleft")
    self.find_overlaping_neuron_synapse_segments("cleft")
    self.overlap_all_segment_pairs("cleft")
    self.construct_full_overlap_mask("cleft")

    self.connectivity_matrix("cleft")

    if self.pred is not None:
      self.create_pred_bounding_boxes()
      self.crop_out_bounding_boxes("pred")
      self.find_overlaping_neuron_synapse_segments("pred")
      self.overlap_all_segment_pairs("pred")
      self.construct_full_overlap_mask("pred")

      self.connectivity_matrix("pred")
      self.calc_f1_score()

# References: 
# Scipy image dilaiton: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html
# Compression methods comparison: https://stackoverflow.com/questions/57983431/whats-the-most-space-efficient-way-to-compress-serialized-python-data
