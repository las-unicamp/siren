#!/bin/bash

# We read the material acceleration for 100 subdomains of the JHTDB. Then, the
# pressure field is reconstructed using the either the GFI or OS-MODI technique.
# Once we have the result, we run an inference script that will calculate the 
# relative Mean Absolute Error and plot an image comparing the original vs 
# reconstructed pressure fields.
#
# Move this file to the project root folder, and execute it from there.

# Total number of cases
TOTAL_CASES=100

for ((i=1; i<=TOTAL_CASES; i++)); do
    echo "Running subdomain $i"

    # Format the number with leading zeros (e.g., 001)
    ID=$(printf "%03d" "$i")

    # Modify params.yaml in place
    sed -i "s/^experiment_name: .*/experiment_name: subdomain$ID/" params_other_methods.yaml
    sed -i "s|^input_filename: .*|input_filename: runs/robustness_100_cases/inputs/subdomain$ID.mat|" params_other_methods.yaml
    sed -i "s|^output_filename: .*|output_filename: osmodi_subdomain$ID.mat|" params_other_methods.yaml

    # Run the Python script to reconstruct the pressure field
    # PYTHONPATH=${PWD} uv run python other_methods/os_modi/main.py -c params_other_methods.yaml
    PYTHONPATH=${PWD} uv run python other_methods/gfi/main.py -c params_other_methods.yaml

    # Run the Python script for inference
    PYTHONPATH=${PWD} uv run python infer_other_methods.py -c params_other_methods.yaml

    echo "Finished subdomain $i"
done