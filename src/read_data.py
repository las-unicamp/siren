import scipy

from src.dtos import TrainingData


def read_data_from_matfile(filename: str) -> TrainingData:
    """
    Reads a MATLAB .mat file and returns its contents as a dictionary.

    Args:
        file (str): Path to the .mat file.

    Returns:
        TrainingData: A dictionary containing the parsed and validated data.

    Raises:
        FileNotFoundError: If the provided file is None or cannot be found.
        KeyError: If required keys are missing from the .mat file.
    """
    if filename is None:
        raise FileNotFoundError("The provided file path is None.")

    raw_data = scipy.io.loadmat(filename)

    # Validate required keys
    required_keys = {"coordinates", "mask"}
    if not required_keys.issubset(raw_data.keys()):
        missing_keys = required_keys - raw_data.keys()
        raise KeyError(f"The .mat file is missing required keys: {missing_keys}")

    # Validate conditional keys
    has_laplacian = "laplacian" in raw_data
    has_gradients = "gradient_x" in raw_data and "gradient_y" in raw_data
    if not (has_laplacian or has_gradients):
        raise ValueError(
            'The .mat file must contain either "laplacian" or both "gradient_x"'
            'and "gradient_y".'
        )

    # Create the TrainingData dictionary
    training_data: TrainingData = {
        "coordinates": raw_data["coordinates"],
        "mask": raw_data["mask"],
    }

    if has_laplacian:
        training_data["laplacian"] = raw_data["laplacian"]
    if has_gradients:
        training_data["gradient_x"] = raw_data["gradient_x"]
        training_data["gradient_y"] = raw_data["gradient_y"]

    return training_data
