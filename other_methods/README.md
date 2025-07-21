## How to run:

Set the parameters in params_other_methods.yaml.
Then, execute the following line in the root directory:


"""bash
# for OS-MODI
PYTHONPATH=${PWD} uv run python other_methods/os_modi/main.py -c params_other_methods.yaml

# for GFI
PYTHONPATH=${PWD} uv run python other_methods/gfi/main.py -c params_other_methods.yaml
"""

To see the result, run:

"""bash
PYTHONPATH=${PWD} uv run python infer_other_methods.py -c params_other_methods.yaml
"""
