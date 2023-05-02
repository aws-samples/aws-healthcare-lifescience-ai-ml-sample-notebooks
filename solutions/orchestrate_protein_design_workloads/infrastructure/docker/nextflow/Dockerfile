

# Original Copyright 2021 DeepMind Technologies Limited
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

FROM public.ecr.aws/lts/ubuntu:18.04_stable as base_image

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      build-essential \
      cmake \
      wget \
      git \
      unzip \
      hmmer \
      tar \
      awscli \
      python3.8 \
      python3-pip \
      openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \      
    && apt-get clean \
    && rm /usr/bin/python3 \
    && ln -s /usr/bin/python3.8 /usr/bin/python3

#COPY env_files /env_files


# Install AWS CLI
RUN wget -O "awscliv2.zip" "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" \
  && unzip awscliv2.zip \
  && ./aws/install \
  && rm awscliv2.zip
  
#need to specify default region for boto3 to behave
ENV AWS_DEFAULT_REGION=us-east-1 

# Install Miniconda package manager and dependencies
ENV PATH="/opt/conda/bin:$PATH"


RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh \ 
  && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
  && rm /tmp/Miniconda3-latest-Linux-x86_64.sh \
  && conda update -n base -c conda-forge conda \
  && conda config --set ssl_verify no \
  && conda init bash \
  && conda clean --all --yes
  
  
#RUN  pip install -r env_files/requirements.txt 

RUN cd /usr/bin && wget -qO- https://get.nextflow.io | bash #install nextflow

#install batchfold for convenience also. 
RUN cd /root && git clone --depth 1 https://github.com/aws-solutions-library-samples/aws-batch-arch-for-protein-folding.git && cd aws-batch-arch-for-protein-folding  && pip install . && cd notebooks/ && pip install -U -q -r notebook-requirements.txt

#RUN cd/root && git clone --depth 1 https://github.com/aws-solutions-library-samples/aws-batch-arch-for-protein-folding.git && cd aws-batch-arch-for-protein-folding  && pip install . && cd infrastructure/docker/nextflow && pip install -U -q -r requirements.txt



WORKDIR /root

ENTRYPOINT ["bash", "-c"] #enable passing of arbitrary commands to nextflow
