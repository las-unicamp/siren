# **Pressure Field Reconstruction with SIREN: A Mesh-Free Approach for Image Velocimetry in Complex Noisy Environments**

This repository contains the code for the Sinusoidal Representation Network (SIREN) designed to reconstruct the pressure field from velocimetry data.


### **Overview**

The proposed SIREN approach is a mesh-free method that directly reconstructs the pressure field, bypassing the need for an intrinsic grid connectivity and, hence, avoiding the challenges associated with ill-conditioned cells and unstructured meshes. This provides a distinct advantage over traditional meshbased methods. Moreover, changes in the architecture of the SIREN can be used to filter out inherent noise from velocimetry data. 

<img alt="result example" src="https://github.com/las-unicamp/siren/blob/main/.github/example.png"/>

The image above illustrates an example of the reconstructed pressure field from the JHTDB "isotropic1024coarse" database using various state-of-the-art algorithms. The dots indicate measurement points where velocity was sampled to compute the material derivative (i.e., the pressure gradient). Both the OS-MODI and GFI methods rely on the creation of a mesh grid, which poses challenges due to ill-conditioned triangles. In contrast, the SIREN approach is mesh-free, offering superior performance in scenarios with unstructured grids. This advantage makes the proposed method particularly well-suited for processing Lagrangian Particle Tracking (LPT) data.

### **Citing**
When using this code, please cite the following reference:
[Miotto, R.F., Wolf, W.R. and Zigunov, F. _Pressure field reconstruction with SIREN_. Exp Fluids 66, 151 (2025).](https://doi.org/10.1007/s00348-025-04074-1)

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
cd siren
```
2. Install dependencies using UV:
```bash
uv sync
```

### **Usage**

### **Running the Script**

The script requires several parameters, which can be passed through the command line or a configuration file.

#### **VSCode Users**
If you're using VSCode, you can configure the `.vscode/launch.json` file to streamline script execution.

#### **Command Line Execution**
Run the script with the required parameters:
```bash
python main.py --logging_root "log" \
               --experiment_name "my_experiment" \
               --fit "gradients" \
               --input_filename "input.mat" \
               --num_hidden_layers 1 \
               --num_nodes_per_layer 64 \
               --first_layer_omega 10 \
               --hidden_layer_omega 30 \
               --learning_rate 1e-4 \
               --num_epochs 2000 \
               --batch_size None \
               --num_workers 0 \
               --delta None \
               --epochs_until_checkpoint 1000 \
               --load_checkpoint None \
               --checkpoint_file_name_on_save "my_checkpoint.pth.tar"
```

Alternatively, use a configuration file (running the command from the root directory):
```bash
PYTHONPATH=${PWD} uv run python src/main.py -c config.yaml
```

### **Required Parameters**

| Parameter                 | Type    | Description                                                                  |
| --------------------------| ------- | ---------------------------------------------------------------------------- |
| `experiment_name`         | `str`   | Name of the subdirectory where summaries and checkpoints will be saved.      |
| `fit`                     | `str`   | Whether training will fit the `gradients` or the `laplacian`.                |
| `input_filename`          | `str`   | Name of the .mat file containing the input data.                             |
| `num_hidden_layers`       | `int`   | Number of hidden layers.                                                     |
| `num_nodes_per_layer`     | `int`   | Number of nodes per layer.                                                   |
| `first_layer_omega`       | `float` | Frequency scaling of the first layer.                                        |
| `hidden_layer_omega`      | `float` | Frequency scaling of the hidden layers.                                      |
| `learning_rate`           | `float` | Learning rate.                                                               |
| `num_epochs`              | `int`   | Number of epochs to train for.                                               |
| `epochs_until_checkpoint` | `int`   | Number of epochs until checkpoint is saved.                                  |

The code also supports finite difference computation of the derivatives. This feature comes in hand when the GPU memory is
insufficient. To use it, you must specify a parameter `delta` which is a small coordinate perturbation to compute the derivative.

A full list of available parameters to be passed during Python execution can be found in `src/hyperparameters.py`.

### **File Requirements**

The current implementation supports MATLAB file formats (`.mat`). This file is passed to the `input_filename` parameter,
which must have the following column headers and data shape:

- "coordinates": ArrayFloat32Nx2 | ArrayFloat32Nx3
- "gradient_x": ArrayFloat32NxN
- "gradient_y": ArrayFloat32NxN
- "gradient_z": ArrayFloat32NxN (optional)
- "laplacian": ArrayFloat32NxN (optional)
- "mask": ArrayBoolNxN

Here, N is the number of point measurements (valid pixels in PIV or particles in PTV).

The coordinates represent the locations where each measurement of the derivative (either the gradients or the Laplacian) 
of the velocity field was taken.

After passing the proper arguments and executing the main script, the training will begin.
A checkpoint will be saved in the root directory, which can be used to run inferences or regenerate the result.
In addition, the code is configured to write log files and intermediary predictions inside `logs/experiment_name` folder.

> **NOTE:** The current implementation supports MATLAB file formats with the mentioned headers. 
However, the user can implement their own readers to accept files with different data structure.

### **License**

This project is licensed under the MIT License.


### **Contributors**

Renato F. Miotto
Fernando Zigunov
William R. Wolf
