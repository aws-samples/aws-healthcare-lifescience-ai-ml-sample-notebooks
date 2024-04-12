#!/usr/bin/bash

# This script is used to create a conda environment for Jupyter with the
# same dependencies as the one used for Neptune ML training jobs.
set -e

conda env create -f environment.gpu.yml
conda activate neptune_ml_p36
pip install neptuneml-toolkit scikit-learn
python -m ipykernel install --user --name=neptune_ml_p36
