# Connectomics research library, Kasthuri Lab, UChicago

### Author: Surya Chandra Kalia
### Advisors: Kevin Boergens, Narayan (Bobby) Kasthuri
### Time: Summer 2023

## Introduction:

Given a dense annotated dataset of brain EM images, this library would create the contactome by dilating the neural segments as well as provide a metric for comparing accuracy of synapse detection models from the perspective of connectome creation.

Current synapse detection models provide a 3D point cloud prediciton as to where a synapse might be located. While sccuacy of such models can be measured using an IOU metric similar those used by object deteciton models, our end objective here is usually to create a connectome, and not just accurate identificaiton of voxel labels.

In this regard, it is essential to measure how well a synapse prediction helps us to identify the connectivity across neurons. Our library provides a framework for this accuracy measurement by dilating neural segments and identifying the regions of intersection. 
The ovelap between these intersections as well as the ground truth synapse regions helps create an estimate of connectivity map across the neural segments. We repeat this process with the predicted synapse regions to provide a metric for accuracy comparison across synapse detection models.



## Setup:
