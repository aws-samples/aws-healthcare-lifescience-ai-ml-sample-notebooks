---
layout: post
title:  
Use Ground Truth Labeled Data to Train an Object Detection Model in Chest Xrays
date:   2022-07-18 00:00:00 +0000
categories: Workshops
---

In this notebook, we demonstrate how to build a machine learning model to detect the trachea of a patient in an x-ray image using Amazon SageMaker. We will be using 1099 NIH Chest X-ray images sampled from this repository. While the images are originally from that source, we leveraged SageMaker Ground Truth to create bounding boxes around the trachea of the patient. We will thus be using both the raw images and also the manifest file where labellers labeled the trachea of the patient. 

This process could potentially be used as a template for detecting other objects as well within X-ray images. However, we focus only on detecting the trachea of the patient, if it is present. This notebook contains instructions to use the GroundTruth manifest file to understand the labeled data, train, build and deploy the model as an end point in SageMaker.

Resources:

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/X_ray_Object_Detection_Ground_Truth/x_ray_ground_truth_object_detection.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
