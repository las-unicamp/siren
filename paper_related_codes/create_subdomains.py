"""
This script aims at splitting the 1024x1024 plane of the JHTDB 2D plane into
100 subdomains of shape 100x100.

This is done to check the robustness of the SIREN hyperparameter. Particularly,
we fix the hyperparameters and train the 100 subdomains to check for the
consistency of the results, measuring the MAE accross all samples. Ideally,
the MAE should be almost constant.
"""

from scipy.io import loadmat, savemat

whole_2d_file_coords = "SIREN_jhtdb_entire_2D_domain_coordinates.mat"
whole_2d_file_mask = "SIREN_jhtdb_entire_2D_domain_mask.mat"
whole_2d_file_source = "source_jhtdb_entire_2D_domain_noise_0.00.mat"


coord_file = loadmat(whole_2d_file_coords)
mask_file = loadmat(whole_2d_file_mask)
source_file = loadmat(whole_2d_file_source)


coord_x = coord_file["coord_x"].reshape((1024, 1024))
coord_y = coord_file["coord_y"].reshape((1024, 1024))
mask = mask_file["mask"].reshape((1024, 1024))
gt = source_file["ground_truth"].reshape((1024, 1024))
gradient_x = source_file["gradient_x"].reshape((1024, 1024))
gradient_y = source_file["gradient_y"].reshape((1024, 1024))
delta = source_file["delta"]
coordinates = source_file["coordinates"].reshape((1024, 1024, 2))

n_splits = 10
block_size = 100

counter = 1
for i in range(n_splits):
    for j in range(n_splits):
        x_start = i * block_size
        x_end = (i + 1) * block_size
        y_start = j * block_size
        y_end = (j + 1) * block_size

        savemat(
            f"subdomain{counter:03d}.mat",
            {
                "coordinates": coordinates[x_start:x_end, y_start:y_end, :].reshape(
                    -1, 2
                ),
                "gradient_x": gradient_x[x_start:x_end, y_start:y_end],
                "gradient_y": gradient_y[x_start:x_end, y_start:y_end],
                "gt": gt[x_start:x_end, y_start:y_end],
                "mask": mask[x_start:x_end, y_start:y_end],
                "coord_x": coord_x[x_start:x_end, y_start:y_end],
                "coord_y": coord_y[x_start:x_end, y_start:y_end],
                "delta": delta,
                "shape": (block_size, block_size),
            },
        )

        counter += 1
