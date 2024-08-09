#!/bin/bash
# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate rel_env2

# Run a simple test, e.g., check Python version
python --version

# Optionally, check for specific packages
conda list

