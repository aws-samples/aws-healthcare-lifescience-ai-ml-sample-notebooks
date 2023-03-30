---
layout: post
title:  Medicare Payment Prediction with Amazon SageMaker Autopilot
date:   2022-01-18 00:00:00 +0000
categories: Workshops
---

Amazon SageMaker Autopilot is an automated machine learning (commonly referred to as AutoML) solution for tabular datasets. You can use SageMaker Autopilot in different ways: on autopilot (hence the name) or with human guidance, without code through SageMaker Studio, or using the AWS SDKs. This notebook, as a first glimpse, will use the AWS SDKs to simply create and deploy a machine learning model. Feature Engineering and hyperparameter tuning can be a laborious process, especially in "messy" dataset, like the one discussed in this notebook. Autopilot will perform the feature engineering and hyperparameter tuning for us.

Predicting payments from insurance providers for healthcare services provided is essential for providers, insurers, and patients. This notebook analzes a 2010 dataset of Medicare payments based on a sample of benificiaries.

Resources:

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Healthcare_Payments_Prediction_SageMaker_AutoPilot/Healthcare_Payments_Prediction_SageMaker_AutoPilot.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
