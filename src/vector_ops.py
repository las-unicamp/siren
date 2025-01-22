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

        def divergence(
            grad: TensorFloatNx2 | TensorFloatNx3,
            coords: TensorFloatNx2 | TensorFloatNx3,
        ) -> TensorFloatNx1:
            """Compute divergence.

            Parameters
            ----------
            grad : torch.Tensor
                tensor of shape `(n_coords, 2)` representing the gradient wrt
                x and y, or `(n_coords, 3)`, for x, y and z-directions.

            coords : torch.Tensor
                tensor of shape `(n_coords, 2)` or `(n_coords, 3)` representing
                the coordinates.

            Returns
            -------
            div : torch.Tensor
                tensor of shape `(n_coords, 1)` representing the divergence.

            Notes
            -----
            In a 2D case this will give us f_{xx} + f_{yy}.
            """
            div = 0.0
            num_dimensions = coords.shape[1]
            for i in range(num_dimensions):
                div += self.compute_gradient(grad[..., i], coords)[..., i : i + 1]
            return div

        grad = self.compute_gradient(target, coords)
        return divergence(grad, coords)


class FiniteDifferenceDerivativesStrategy:
    def compute_gradient(
        self,
        target: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
        model: torch.nn.Module,
        delta: float,
    ) -> TensorFloatNx2 | TensorFloatNx3:
        num_dimensions = coords.shape[1]  # 2 for 2D, 3 for 3D
        grads = []

        for dim in range(num_dimensions):
            perturbed_coords = coords.clone()
            perturbed_coords[:, dim] += delta

            perturbed_output = model(perturbed_coords)

            finite_diff = (perturbed_output - target) / delta
            grads.append(finite_diff)

        return torch.cat(grads, dim=-1)

    def compute_laplacian(
        self,
        target: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
        model: torch.nn.Module,
        delta: float,
    ) -> TensorFloatNx1:
        num_dimensions = coords.shape[1]
        laplacian = torch.zeros_like(target)

        for dim in range(num_dimensions):
            # Compute second-order finite difference for Laplacian
            perturbed_coords_plus = coords.clone()
            perturbed_coords_plus[:, dim] += delta

            perturbed_coords_minus = coords.clone()
            perturbed_coords_minus[:, dim] -= delta

            output_plus = model(perturbed_coords_plus)
            output_minus = model(perturbed_coords_minus)

            laplacian += (output_plus - 2 * target + output_minus) / (delta**2)

        return laplacian
