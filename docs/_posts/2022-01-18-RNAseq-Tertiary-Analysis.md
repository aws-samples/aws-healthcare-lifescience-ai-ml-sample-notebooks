---
layout: post
title:  "Classifying HER2 status from RNASeq Data"
date:   2022-01-18 00:00:00 +0000
categories: Workshops 
author: 
    name: Brian Loyal
---

According to the American Cancer Society (ACS), breast cancer is the second most common cancer in women in the United States, behind skin cancers. It represents about 30% of all new female cancers each year. In 2022, ACS estimates that about 287,850 new cases of invasive breast cancer will be diagnosed in women and about 43,250 women will die from the disease.

HER2 is a protein that helps breast cancer cells grow quickly. Breast cancer cells with high levels of HER2 (referred to as "HER2-positive") are treated with specialized drugs. These targeted therapies have significantly improved the survival rate of HER2+ breast cancer patients. For this reason, it is important to quickly classify breast cancers as either HER2 positive or negative.

RNA Sequencing (RNAseq) is a recent laboratory technique that uses high-throughput DNA sequencing to capture a "snapshot" of the genes expressed by cells at a moment in time. This information can then be used to diagnose genetic diseases, including cancer.

This lab has three sections. In each section you will use RNAseq data from the Cancer Genome Atlas to train a machine learning (ML) model that classifies samples as either HER2 positive or negative. However, each section accomplishes this in a different way.

1. In section one, you will import the open source XGBoost library into a SageMaker Studio Notebook and use it to train and validate the model.

1. In section two, you will use SageMaker Processing and Model Training jobs to optimize your training costs and track experiments.

1. In section three, you will use SageMaker MLOps Projects to create a reproducible pipeline for training and deploying machine learning models.

*Resources* :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/tree/main/workshops/RNAseq_Tertiary_Analysis){:target="_blank"}{:rel="noopener noreferrer"}
