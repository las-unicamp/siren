from typing import Dict, Protocol

from src.my_types import TensorFloatN, TensorFloatNx2, TensorFloatNx3


class NetworkTracker(Protocol):
    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric"""

    def add_batch_data(
        self, key: str, data: TensorFloatN | TensorFloatNx2 | TensorFloatNx3
    ):
        """Updates batch-level data to further save the epoch result in a file"""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging an epoch-level metric"""

    def should_save_intermediary_data(self) -> bool:
        """
        Returns whether intermediary data should be saved.
        The class should track this status internally.
        """

    def get_batch_data(
        self,
    ) -> Dict[str, TensorFloatN | TensorFloatNx2 | TensorFloatNx3]:
        """Returns the batch data for the current training step"""

    def save_epoch_data(self, name: str, step: int):
        """Implements saving an epoch-level data (prediction or derivative)"""
