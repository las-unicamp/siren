# **Pressure Field Reconstruction with SIREN: A Mesh-Free Approach for Image Velocimetry in Complex Noisy Environments**

This repository contains the code for the Sinusoidal Representation Network (SIREN) designed to reconstruct the pressure field from velocimetry data.


### **Overview**

The proposed SIREN approach is a mesh-free method that directly reconstructs the pressure field, bypassing the need for an intrinsic grid connectivity and, hence, avoiding the challenges associated with ill-conditioned cells and unstructured meshes. This provides a distinct advantage over traditional meshbased methods. Moreover, changes in the architecture of the SIREN can be used to filter out inherent noise from velocimetry data. 


### **Installation**

> :warning: **macOS and Windows platforms**: CUDA-enabled builds only work on Linux.
> For macOS and Windows, the project must be configured to use specific accelerators (See UV-Pytorch integration in the [UV guides](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index))

**Using UV:**
We recommend using UV as Python package and project manager. It replaces many tools making the workflow easier.

See [here](https://docs.astral.sh/uv/getting-started/installation/) how to install UV in your machine.

To install and run this code locally, follow these instructions:
1. Clone the repository:
```bash
git clone https://github.com/las-unicamp/siren.git
```
1. Install required dependencies and run script:
```bash
uv run my_script.py
```

<!-- 2. Install required dependencies:
```bash
uv sync
```
1. Activate virtual environment
```bash
source .venv/bin/activate
``` -->

### **Usage**

Once installed, the model can be trained on your own data or applied to the existing datasets.
Instructions for training, testing, and running the model is provided in this section (TO BE ADDED)

### **License**

This project is licensed under the MIT License.


### **Contributors**

Renato F. Miotto
Fernando Zigunov
William R. Wolf
