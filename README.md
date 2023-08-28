# Connectomics research library, Kasthuri Lab, UChicago

### Author: Surya Chandra Kalia
### Advisors: Kevin Boergens, Narayan (Bobby) Kasthuri
### Time: Summer 2023

## Introduction:

This Python library was developed to aid connectomics research. Our focus here is on the evaluation of synapse detection models. The output of a synapse detection model would usually be a mask or heatmap over a 3D voxel space. 

<Optional Image of mask/heatmaps>

Given a ground truth annotation, we conventionally used the IOU (intersection over union) metric for evaluating object detection and segmentation models. This essentially measures the overlap of the prediction and ground truth as a fraction of the union of the two volumes. Here is an example of this metric used in 2D images:

<IOU image>


While this is a simple and convenient evaluation metric, it can be a bit too strict and rigid for the context of connectomics. This is because our end goal is to reconstruct the connectome by identifying the correct points of synapse contacts between the 3D segmented neurons. For this reconstruction task, we are not concerned with the accurate replication of the synapse shape. Instead, we just need to ensure that the synapse predictions correctly identify the neuron segments that are connected by that synapse.

<Optional shape denoting the purpose>

Using this library, we can construct the connectivity matrix across neuron segments given the synapse ground truth/prediction. Given the connectivity matrix, we calculate the F1 score to quantitatively comment on the accuracy of the predictions based on how well they help reconstruct the connectome.

## Setup

### Installation

All conda dependencies are specified in `environment.yml` which can be imported directly using:

```
conda env create -f environment.yml
```

This will create a conda env named `gpu_torch`. Activate the environment using the following command:
```
conda activate gpu_torch
```

Source : https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

> [!NOTE]
> In case the pip requirements don't get installed during the conda env creation, there is a requirements.txt file available for manual installation. This can be done by activating the conda env and running the following command:  pip3 install -r requirements.txt

### Dataset download

Download the cropped version of the CREMI dataset available here: https://cremi.org/data/

The path to the downloaded file should be set as the `cremi_file_path` argument when running the code. The following is an example of the filepath to the hdf file:

```
cremi_file_path = /home/suryakalia/documents/summer/datasets/cremi/sample_A_20160501.hdf
```

### [OPTIONAL] Predictions

In order to be able to calculate the F1 scores, we would need to have a synapse prediction file containing the synapse annotations as segmentations (i.e. each synapse region labelled with a unique ID)

This would be in the form of an int/uint numpy array with dimenstions identical to that of CREMI dataset's `clefts` volume. The expected axes format for the numpy array is (z, y, x) and this is to be compressed as aan lzma file (.xz)

As an example, one sample is provided in the repo with the filename `result_A_components.xz`. This would be passed in as the optional `pred_path` argument when initializing the `DilateOverlap` class

If needed, you can inspect lzma files using pickle as follows:

```
import lzma
import pickle
import numpy as np

with lzma.open("result_A_components.xz", "rb") as f:
  pred_img = pickle.load(f)

print(pred_img.shape)
```

## Execution


