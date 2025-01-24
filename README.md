# **Pressure Field Reconstruction with SIREN: A Mesh-Free Approach for Image Velocimetry in Complex Noisy Environments**

This repository contains the code for the Sinusoidal Representation Network (SIREN) designed to reconstruct the pressure field from velocimetry data.


### **Overview**

The proposed SIREN approach is a mesh-free method that directly reconstructs the pressure field, bypassing the need for an intrinsic grid connectivity and, hence, avoiding the challenges associated with ill-conditioned cells and unstructured meshes. This provides a distinct advantage over traditional meshbased methods. Moreover, changes in the architecture of the SIREN can be used to filter out inherent noise from velocimetry data. 

<img alt="result example" src="https://github.com/las-unicamp/siren/blob/main/.github/example.png"/>

The image above illustrates an example of the reconstructed pressure field from the JHTDB "isotropic1024coarse" database using various state-of-the-art algorithms. The dots indicate measurement points where velocity was sampled to compute the material derivative (i.e., the pressure gradient). Both the OS-MODI and GFI methods rely on the creation of a mesh grid, which poses challenges due to ill-conditioned triangles. In contrast, the SIREN approach is mesh-free, offering superior performance in scenarios with unstructured grids. This advantage makes the proposed method particularly well-suited for processing Lagrangian Particle Tracking (LPT) data.

### **Citing**
When using this code, please cite the following reference:
- Renato Miotto, Fernando Zigunov, William Wolf: _Pressure Field Reconstruction with SIREN: A Mesh-Free Approach for Image Velocimetry in Complex Noisy Environments_

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
2. Install required dependencies and run script:
```bash
uv run src/main.py
```

### **Usage**

The code expects a MATLAB (.mat) file containing the coordinates (either physical or pixel coordinates) where measurements were taken and the derivatives at these locations. 
This MATLAB file must have the following column headers and data shape:

- "coordinates": ArrayFloat32Nx2 | ArrayFloat32Nx3
- "gradient_x": ArrayFloat32NxN
- "gradient_y": ArrayFloat32NxN
- "gradient_z": ArrayFloat32NxN (optional)
- "mask": ArrayBoolNxN

Here, N is the number of points.

A few parameters must be passed during Python execution. The available parameters can be found in `hyperparameters.py`.
The Visual Studio IDE is already configured to handle passing the parameters in a convenient way (see `.vscode/launch.json`).

After passing the proper arguments and executing the main script, the training will begin.
A Checkpoint will be saved in the root directory, which can be used to run inferences or regenerate the result.
In addition, the code is configured to write log files and intermediary predictions inside `logs/experiment_name` folder.

### **License**

This project is licensed under the MIT License.


### **Contributors**

Renato F. Miotto
Fernando Zigunov
William R. Wolf
