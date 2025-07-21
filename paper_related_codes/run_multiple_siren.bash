#!/bin/bash

# This script aims at assessing the robustness of the SIREN for fixed parameters
# We read the material acceleration for 100 subdomains of the JHTDB. Then, the
# pressure field is reconstructed using the SIREN, while keeping its parameters
# unchanged. Once the model is trained, we run an inference script that will
# calculate the relative Mean Absolute Error and plot an image comparing the
# original vs reconstructed pressure fields.
#
# Move this file to the project root folder, and execute it from there.
# Also, make sure to move/keep the checkpoints in the root dir for inference.

# Total number of cases
TOTAL_CASES=100

for ((i=1; i<=TOTAL_CASES; i++)); do
    echo "Running subdomain $i"

    # Format the number with leading zeros (e.g., 001)
    ID=$(printf "%03d" "$i")

    # Modify params.yaml in place
    sed -i "s/^experiment_name: .*/experiment_name: \"subdomain$ID\"/" params.yaml
    sed -i "s|^input_filename: .*|input_filename: \"runs/robustness_100_cases/inputs/subdomain$ID.mat\"|" params.yaml
    sed -i "s/^checkpoint_file_name_on_save: .*/checkpoint_file_name_on_save: \"checkpoint$ID.tar\"/" params.yaml

    # Run the Python script for training
    PYTHONPATH=${PWD} uv run python src/main.py -c params.yaml

    # Run the Python script for inference
    PYTHONPATH=${PWD} uv run python infer.py -c params.yaml

    echo "Finished subdomain $i"
done