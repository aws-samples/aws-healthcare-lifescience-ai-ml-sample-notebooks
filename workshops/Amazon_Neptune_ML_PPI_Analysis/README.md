# Amazon Neptune ML for Protein-Protein Interaction

## Summary

Proof of Concept (POC) to predict protein-protein interactions with graph neural networks.

## Project Overview

1. Overview

A recent report from the Indian Institute of Technology (IIT) described the use of graph neural network (GNN) analysis to identify potential PPIs. They used a protein language model (pLM) to translate the amino acid sequence of 121,000 proteins into vector embeddings. They then associated paired embedding vectors to nodes in a graph and used Graph-BERT to classify them as as either positive (potential PPI) or negative. The resulting model was 99% accurate at predicting known PPIs without any manual feature curation.

For this POC, we will use a similar approach as the IIT paper, but treat the PPI prediction goal as a link prediction problem. This will reduce the size of the graph database and permit use of standard graph neural net (GNN) algorithms and libraries (e.g. DGL).

We will use a public dataset of PPIs for a model organism to validate our approach. First, we will convert each amino acid sequence in the dataset into a vector embedding using a pre-trained pLM. Next, we will use the embeddings and known PPIs to populate a graph database. In this case, each node in the graph will represent the sequence embedding for a single protein and protein pairs with known interactions will be connected by an edge. Finally, we will train a graph neural net (GNN) model to predict unknown graph edges, representing potential PPIs.

We will use a five-step workload with Amazon Neptune to train and deploy the PPI prediction mode. First we will calculate sequence embeddings for the proteins in our PPI training data set using a pretrained pLM such as ESM-2 hosted in Amazon SageMaker.  Next, we will load these embeddings and known PPIs into an Amazon Neptune graph database. Then, following the standard Neptune ML workflow we will export the graph data to Amazon S3, use SageMaker to train a GNN model for link prediction, and deploy the model as a real-time inference endpoint. Finally, we will use this endpoint to predict unknown PPIs via Neptune queries.

Academic researchers have publicly reported use of ESM-2 pLM embeddings for a variety of tasks, including protein structure prediction, binding pocket identification, and mutation pathogenicity. Amazon Neptune provides serverless graph data storage, minimizing infrastructure maintenance costs, and supports high-performance graph analytics. The resulting protein graph can be further expanded to include additional protein properties in support of other analyses.

## Setup

### CloudFormation

To deploy Neptune and all supporting infrastructure into an existing VPC, first authenticate into an AWS account using your SSO, then use the provided deploy.sh script to deploy the required CloudFormation template.

```bash
./deploy.sh \
  -b "my-deployment-bucket" \
  -n "my-neptune-ml-stack" \
  -r "us-east-1" \
  -v "vpc-12345678" \ 
  -w "subnet-12345678" \
  -x "subnet-12345678" \
  -y "subnet-12345678" \
  -z "subnet-12345678" \
  -d "sg-12345678" \
  -n "ml.g5.2xlarge"
```
