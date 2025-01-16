import torch
from torchmetrics import Metric


class RelativeResidualError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "derivatives",
            default=torch.empty(0),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "target",
            default=torch.empty(0),
            dist_reduce_fx="cat",
        )

    def update(self, derivatives: torch.Tensor, target: torch.Tensor) -> None:
        self.derivatives = torch.cat((self.derivatives, derivatives.squeeze()), 0)
        self.target = torch.cat((self.target, target.squeeze()), 0)

    def compute(self) -> torch.Tensor:
        return torch.linalg.norm(self.target - self.derivatives) / torch.linalg.norm(
            self.target
        )
