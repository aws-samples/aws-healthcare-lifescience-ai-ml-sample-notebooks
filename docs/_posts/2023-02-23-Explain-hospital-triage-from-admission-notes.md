---
layout: post
title:  Explain Hospital Triage from Admission Notes
date:   2023-02-23 00:00:00 +0000
categories: Workshops
---

The intent of this notebook is to provide a practical guide for data scientists, and machine learning engineers to collaborate with clinicians, and to support real implementations of clinical indicator predictions. As such, explainability of the algorithms is required.

Advances in NLP algorithms, as in the studies above, have made predicting clinical indicators more accurate, yet in order to effectively use machine learning models in a production setting, clinicians also need more insight into how these models work. They need to know that these algorithms make clinical sense before going to production. Clinicians and data scientists, need a way to evaluate realiablility, and explainability of models over time, as more data continues to be evaluated, and machine learning models are retrained.

This notebook will take one of these clinical triage indicators, in-hospital mortality, and show how AWS services and infrastructure, along with pre-trained HuggingFace BERT models, can be used to train a binary classifier on text data, estimate a threshold value for triage, and then use Amazon Sagemaker Clarify to explain what admission note text is supporting the recommendations the algorithm is making.

Resources :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Explain-hospital-triage-from-admission-notes/explain-hospital-triage-prediction-with-amazon-sagemaker-clarify.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
