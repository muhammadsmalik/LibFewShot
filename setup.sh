#!/bin/bash

# This script sets the PJRT_DEVICE environment variable,
# clones a git repository, changes directory,
# installs python requirements, creates a data directory,
# and downloads a dataset using gsutil.

# Export environment variable
export PJRT_DEVICE=TPU

# Clone the repository
git clone -b custom_yaml https://github.com/muhammadsmalik/LibFewShot

# Change directory
cd LibFewShot

# Install Python requirements
pip install -r requirements.txt

# Create a data directory
mkdir data

# Download the dataset using gsutil
gsutil -m cp -r gs://seed_dataset/consolidated_seeds_dataset data
