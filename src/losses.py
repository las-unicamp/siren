import torch

from src.my_types import TensorFloatNx1, TensorFloatNx2, TensorFloatNx3, TensorScalar
from src.vector_ops import (
    AutogradDerivativesStrategy,
    DerivativesStrategy,
    FiniteDifferenceDerivativesStrategy,
)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()


class FiniteDifferenceConfig:
    def __init__(self, model: torch.nn.Module, delta: float = 1e-5):
        self.model = model
        self.delta = delta


class FitGradients(torch.nn.Module):
    def __init__(
        self,
        strategy: DerivativesStrategy,
        finite_diff_config: FiniteDifferenceConfig = None,
    ):
        super(FitGradients, self).__init__()
        self.strategy = strategy
        self.finite_diff_config = finite_diff_config

    def forward(
        self,
        y_pred: TensorFloatNx1,
        grad_y_true: TensorFloatNx2 | TensorFloatNx3,
        coords: TensorFloatNx2 | TensorFloatNx3,
    ) -> TensorScalar:
        if isinstance(self.strategy, AutogradDerivativesStrategy):
            grads = self.strategy.compute_gradient(y_pred, coords)
        elif (
            isinstance(self.strategy, FiniteDifferenceDerivativesStrategy)
            and self.finite_diff_config is not None
        ):
            grads = self.strategy.compute_gradient(
                y_pred,
                coords,
                self.finite_diff_config.model,
                self.finite_diff_config.delta,
            )
        else:
            raise ValueError(
                "For finite difference strategy, a model must be provided."
            )

        return loss_fn(grads, grad_y_true), grads


class FitLaplacian(torch.nn.Module):
    def __init__(
        self,
        strategy: DerivativesStrategy,
        finite_diff_config: FiniteDifferenceConfig = None,
    ):
        super(FitLaplacian, self).__init__()
        self.strategy = strategy
        self.finite_diff_config = finite_diff_config

    def forward(
        self,
        y_pred: TensorFloatNx1,
        lapl_y_true: TensorFloatNx1,
        coords: TensorFloatNx2 | TensorFloatNx3,
    ) -> TensorScalar:
        if isinstance(self.strategy, AutogradDerivativesStrategy):
            lapl = self.strategy.compute_laplacian(y_pred, coords)
        elif (
            isinstance(self.strategy, FiniteDifferenceDerivativesStrategy)
            and self.finite_diff_config is not None
        ):
            lapl = self.strategy.compute_laplacian(
                y_pred,
                coords,
                self.finite_diff_config.model,
                self.finite_diff_config.delta,
            )
        else:
            raise ValueError(
                "For finite difference strategy, a model must be provided."
            )

        return loss_fn(lapl, lapl_y_true), lapl
