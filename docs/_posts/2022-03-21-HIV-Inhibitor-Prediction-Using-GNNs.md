---
layout: post
title:  HIV Inhibitor Prediction Using Graph Neural Networks (GNN) on Amazon SageMaker
date:   2022-03-21 00:00:00 +0000
categories: Workshops
---

Human immunodeficiency virus type 1 (HIV-1) is the most common cause of Acquired Immunodeficiency Syndrome (AIDS). One ongoing area of research is finding compounds that inhibit HIV-1 viral replication.

Convolution neural networks, commonly used in computer vision, are also useful for analyzing graphs. Convolutions allow for inductive learning, whereby features are learned for different graph topologies. These convolutions transform the underlying information in the graph nodes and edges. While a single convolutional layer is generally not sufficient for most tasks, deep graph convolutional neural networks can perform graph prediction (i.e., predict the class of a network), link prediction (predict missing edges in a network), and other tasks.

Deep learning models can also incorporate different edge types as well as external information about edges and nodes. This makes deep learning an attractive approach for analyzing and making predictions about complex graphs. Biological networks are frequently very heterogeneous and include diverse data types such as metabolic, biophysical, proteomic and functional assays, and information about gene regulatory networks. For example, this blog post shows how a knowledge graph with diverse node and edge types can predict drug repurposing.

While scientists can create their own convolutional layers, deep learning researchers have already built many convolutions and architectures that have proven useful in many applications. For example, GraphSage can predict protein-protein interactions. Another commonly used approach is Graph Attention Networks (GAT).

This example notebook trains multiple graph neural network models using Deep Graph Library and deploys them using Amazon SageMaker, a comprehensive and fully-managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models and then directly deploy them into a production-ready hosted environment.

Resources :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Molecular-property-prediction/hiv-inhibitor-prediction-dgl/molecule-hiv-inhibitor-prediction-sagemaker.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
