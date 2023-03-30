---
layout: post
title:  Automate Retraining of Obesity Models using SageMaker Pipelines
date:   2021-12-20 00:00:00 +0000
categories: Workshops
---

This workshop shows how you can build and deploy SageMaker Pipelines for multistep processes. In this example, we will build a pipeline that:

 1. Deduplicates the underlying data
 1. Trains a built-in SageMaker algorithm (XGBoost)

A common workflow is that models need to be retrained when new data arrives. This notebook also shows how you can set up a Lambda function that will retrigger the retraining pipeline when new data comes in.

Resources :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Sagemaker_Pipelines_Automated_Retraining/sagemaker_pipelines_automated_retraining.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
