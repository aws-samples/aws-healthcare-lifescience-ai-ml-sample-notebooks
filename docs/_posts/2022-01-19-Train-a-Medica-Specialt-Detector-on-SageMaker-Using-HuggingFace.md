---
layout: post
title:  "Train a Medical Specialty Detector on SageMaker Using HuggingFace."
date:   2022-01-19 00:00:00 +0000
categories: Workshops
---

In this workshop, we will show how you can train an NLP classifier using trainsformers from HuggingFace. HuggingFace allows for easily using prebuilt transformers, which you can train for your own use cases.

In this workshop, we will use the SageMaker HuggingFace supplied container to train an algorithm that will distinguish between physician notes that are either part of the General Medicine (encoded as 0), or Radiology (encoded as 1) medical specialties. The data is a subsample from MTSamples which was downloaded from here.

*Resources* :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }]https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Classify_Medical_Specialty_NLP_Huggingface_Transformers/1_sagemaker_medical_specialty_using_transfomers.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
