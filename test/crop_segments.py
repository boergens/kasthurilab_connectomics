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

dilator = DilateOverlap(cremi_file_path="/home/suryakalia/documents/summer/datasets/cremi/sample_A_20160501.hdf",
                        output_dir="/home/suryakalia/documents/summer/tests/cremi_A_crops",
                        dilation_voxel_count=1,
                        voxel_volume_threshold=50)

dilator.blacklist_append([20474])

# dilator.run()
dilator.overlap_all_segments()
dilator.construct_full_overlap_mask()