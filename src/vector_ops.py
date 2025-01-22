from typing import Protocol, overload

import torch

from src.my_types import TensorFloatNx1, TensorFloatNx2, TensorFloatNx3


class DerivativesStrategy(Protocol):
    @overload
    def compute_gradient(
        self, target: TensorFloatNx1, coords: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatNx2 | TensorFloatNx3:
        """Implements the gradient computation using Pytorch autograd"""

    @overload
    def compute_gradient(
        self,
        target: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
        model: torch.nn.Module,
        delta: float,
    ) -> TensorFloatNx2 | TensorFloatNx3:
        """Implements the gradient computation using finite-difference approach"""

    def compute_gradient(self, *args, **kwargs) -> TensorFloatNx2 | TensorFloatNx3:
        """Fallback method to handle both overloads."""

    @overload
    def compute_laplacian(
        self, target: TensorFloatNx1, coords: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatNx1:
        """Laplacian computation."""

    @overload
    def compute_laplacian(
        self,
        target: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
        model: torch.nn.Module,
        delta: float,
    ) -> TensorFloatNx1:
        """Finite-difference Laplacian computation."""

    def compute_laplacian(self, *args, **kwargs) -> TensorFloatNx1:
        """Fallback method to handle both overloads."""


class AutogradDerivativesStrategy:
    def compute_gradient(
        self, target: TensorFloatNx1, coords: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatNx2 | TensorFloatNx3:
        """Compute the gradient of target with respect to input coords.

        Parameters
        ----------
        target : torch.Tensor
            tensor of shape `(n_coords, ?)` representing the targets.

        coords : torch.Tensor
            tensor fo shape `(n_coords, 2)` or `(n_coords, 3)` representing the
            coordinates.

        Returns
        -------
        grad : torch.Tensor
            tensor of shape `(n_coords, 2)` or `(n_coords, 3)` representing the
            gradient.
        """
        (grad,) = torch.autograd.grad(
            target, coords, grad_outputs=torch.ones_like(target), create_graph=True
        )
        return grad

    def compute_laplacian(
        self, target: TensorFloatNx1, coords: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatNx1:
        """Compute Laplacian operator.

        Parameters
        ----------
        target : torch.Tensor
            tensor of shape `(n_coords, 1)` representing the targets.

        coords : torch.Tensor
            tensor of shape `(n_coords, 2)` or `(n_coords, 3)` representing the
            coordinates.

        Returns
        -------
        torch.Tensor
            tensor of shape `(n_coords, 1)` representing the Laplacian.
        """

        # Compute the gradient of the target with respect to coords
        gradients = self.compute_gradient(target, coords)  # Shape: (n_coords, num_dims)

        laplacian = torch.zeros_like(target)

        # Loop over each dimension to compute the second derivatives
        for dim in range(coords.shape[1]):
            grad_dim = gradients[:, dim]  # Extract gradient for the current dimension

            # Compute second derivative for the current dimension
            grad2_dim = torch.autograd.grad(
                grad_dim,
                coords,
                grad_outputs=torch.ones_like(grad_dim),
                create_graph=True,
            )[0][:, dim]  # Extract the derivative for the same dimension

            laplacian += grad2_dim.unsqueeze(-1)  # Accumulate into the Laplacian

        return laplacian


class FiniteDifferenceDerivativesStrategy:
    def compute_gradient(
        self,
        target: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
        model: torch.nn.Module,
        delta: float,
    ) -> TensorFloatNx2 | TensorFloatNx3:
        """
        Compute the gradient using finite differences in a batched manner.

        Parameters
        ----------
        target : TensorFloatN
            Tensor of shape `(n_coords, ?)` representing the targets.
        coords : TensorFloatNx2 | TensorFloatNx3
            Tensor of shape `(n_coords, 2)` or `(n_coords, 3)` representing the
            coordinates.
        model : torch.nn.Module
            Model to evaluate the perturbed coordinates.
        delta : float
            Finite difference step size.

        Returns
        -------
        TensorFloatNx2 | TensorFloatNx3
            Gradient of shape `(n_coords, 2)` or `(n_coords, 3)`.
        """
        num_dimensions = coords.shape[1]
        n_coords = coords.shape[0]

        # Create perturbed coordinates for all dimensions in a batched manner
        perturbed_coords = coords.unsqueeze(1).repeat(1, num_dimensions, 1)
        for dim in range(num_dimensions):
            perturbed_coords[:, dim, dim] += delta

        # Flatten for a single model call
        flat_perturbed_coords = perturbed_coords.view(-1, num_dimensions)

        # Perform a single batched model call
        flat_perturbed_outputs = model(flat_perturbed_coords)

        # Reshape the outputs to match the number of dimensions
        perturbed_outputs = flat_perturbed_outputs.view(n_coords, num_dimensions, -1)

        # Compute finite differences
        finite_differences = (perturbed_outputs - target.unsqueeze(1)) / delta

        # Concatenate the gradients for all dimensions
        return finite_differences.squeeze(-1)

    def compute_laplacian(
        self,
        target: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
        model: torch.nn.Module,
        delta: float,
    ) -> TensorFloatNx1:
        """
        Compute the Laplacian using second-order finite differences in a batched manner.

        Parameters
        ----------
        target : TensorFloatNx1
            Tensor of shape `(n_coords, 1)` representing the target values.
        coords : TensorFloatNx2 | TensorFloatNx3
            Tensor of shape `(n_coords, 2)` or `(n_coords, 3)` representing the
            coordinates.
        model : torch.nn.Module
            Model to evaluate the perturbed coordinates.
        delta : float
            Finite difference step size.

        Returns
        -------
        TensorFloatNx1
            Tensor of shape `(n_coords, 1)` containing the Laplacian values.
        """
        num_dimensions = coords.shape[1]
        n_coords = coords.shape[0]

        # Create forward and backward perturbed coordinates for all dimensions
        perturbed_coords = coords.unsqueeze(1).repeat(1, num_dimensions * 2, 1)

        # Adjust coordinates for forward and backward perturbations
        for dim in range(num_dimensions):
            perturbed_coords[:, 2 * dim, dim] += delta  # Forward perturbation
            perturbed_coords[:, 2 * dim + 1, dim] -= delta  # Backward perturbation

        # Flatten for a single model call
        flat_perturbed_coords = perturbed_coords.view(-1, num_dimensions)

        # Perform a single batched model call
        flat_perturbed_outputs = model(flat_perturbed_coords)

        # Reshape the outputs to separate forward and backward perturbations
        perturbed_outputs = flat_perturbed_outputs.view(n_coords, num_dimensions, 2, -1)

        # Compute second-order finite differences for each dimension
        forward_outputs = perturbed_outputs[:, :, 0, :]
        backward_outputs = perturbed_outputs[:, :, 1, :]
        second_order_diffs = (
            forward_outputs - 2 * target.unsqueeze(1) + backward_outputs
        ) / (delta**2)

        # Sum over dimensions to get the Laplacian
        laplacian = second_order_diffs.sum(dim=1)

        return laplacian
