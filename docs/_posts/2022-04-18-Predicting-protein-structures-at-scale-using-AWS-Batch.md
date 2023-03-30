---
layout: post
title:  "Predicting protein structures at scale using AWS Batch"
date:   2022-10-19 00:00:00 +0000
categories: Blogs 
author: 
    name: Brian Loyal
---

Proteins are large biomolecules that play an important role in the body. Knowing the physical structure of proteins is key to understanding their function. However, it can be difficult and expensive to determine the structure of many proteins experimentally. One alternative is to predict these structures using machine learning algorithms. Several high-profile research teams have released such algorithms, including AlphaFold2, RoseTTAFold, and others. Their work was important enough for Science magazine to name it the 2021 Breakthrough of the Year.

Both AlphaFold2 and RoseTTAFold use a multitrack transformer architecture trained on known protein templates to predict the structure of unknown peptide sequences. These predictions are heavily GPU-dependent and take anywhere from minutes to days to complete. The input features for these predictions include multiple sequence alignment (MSA) data. MSA algorithms are CPU-dependent and can themselves require several hours of processing time.

Running both the MSA and structure prediction steps in the same computing environment can be cost-inefficient because the expensive GPU resources required for the prediction sit unused while the MSA step runs. Instead, using a high-performance computing (HPC) service like AWS Batch allows us to run each step as a containerized job with the best fit of CPU, memory, and GPU resources.

In this post, we demonstrate how to provision and use AWS Batch and other services to run AI-driven protein folding algorithms like RoseTTAFold.

*Resources* :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-solutions-library-samples/aws-batch-arch-for-protein-folding){:target="_blank"}{:rel="noopener noreferrer"}
* [AWS for Industries Blog](https://aws.amazon.com/blogs/industries/predicting-protein-structures-at-scale-using-aws-batch){:target="_blank"}{:rel="noopener noreferrer"}
