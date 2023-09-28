'''
@File    :   crop_segments.py
@Time    :   2023/06/29 16:48:30
@Author  :   Surya Chandra Kalia
@Version :   1.0
@Contact :   suryackalia@gmail.com
@Org     :   Kasthuri Lab, University of Chicago
'''

import os
import sys
import numpy as np
import pickle
import lzma
sys.path.insert(0, os.path.abspath('/home/suryakalia/documents/summer/exploration/kasthurilab_connectomics/'))
# Need to add above path since VSCode Jupyter Notebook doesn't respect system's $PYTHONPATH variable
# This will be eliminated once my module is converted to a conda package and installed to the conda env

from sk_connectomics.core.dilate_overlap import DilateOverlap


dilator = DilateOverlap(cremi_file_path="/home/suryakalia/documents/summer/datasets/cremi_clean/clean_sample_A.h5",
                        output_dir="/home/suryakalia/documents/summer/tests/cremi_clean/",
                        dilation_voxel_count=1,
                        voxel_volume_threshold=50,
                        num_cores=1,
                        pred_path="/home/suryakalia/documents/summer/datasets/cremi_clean/result_A_components.xz")


# dilator.blacklist_append([20474]) # list of cells that should be ignored, such as glia cells

dilator.run()

# dilator.create_neuron_bounding_boxes()
# dilator.trim_invalid_segments()
# dilator.crop_out_bounding_boxes("neuron")
# dilator.find_overlaping_neuron_segments()
# dilator.overlap_all_segment_pairs("neuron")
# dilator.construct_full_overlap_mask("neuron")

# dilator.create_cleft_bounding_boxes()
# dilator.crop_out_bounding_boxes("cleft")
# dilator.find_overlaping_neuron_synapse_segments("cleft")
# dilator.overlap_all_segment_pairs("cleft")
# dilator.construct_full_overlap_mask("cleft")

# dilator.connectivity_matrix("cleft")


# dilator.create_pred_bounding_boxes()
# dilator.crop_out_bounding_boxes("pred")
# dilator.find_overlaping_neuron_synapse_segments("pred")
# dilator.overlap_all_segment_pairs("pred")
# dilator.construct_full_overlap_mask("pred")

# dilator.connectivity_matrix("pred")

# dilator.calc_f1_score()
