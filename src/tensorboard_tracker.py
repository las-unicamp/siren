import os
import time
from pathlib import Path
from typing import Dict

import torch
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter

from src.my_types import TensorFloatN, TensorFloatNx2, TensorFloatNx3


class TensorboardTracker:
    def __init__(
        self,
        log_dir: str,
        dirname: str = "",
        create: bool = True,
        save_intermediary_data: bool = True,
    ):
        default_name = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        dirname = dirname if dirname else default_name

        self.directory = os.path.join(log_dir, dirname)
        self._validate_log_dir(log_dir, create=create)

        self._writer = SummaryWriter(self.directory)

        self._batch_data = {}

        self._save_intermediary_data = save_intermediary_data

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        path = Path(log_dir).resolve()
        if path.exists():
            return
        if not path.exists() and create:
            path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def flush(self):
        self._writer.flush()

        self._batch_data = {}

    def should_save_intermediary_data(self) -> bool:
        return self._save_intermediary_data

    def add_batch_data(
        self, key: str, data: TensorFloatN | TensorFloatNx2 | TensorFloatNx3
    ):
        if key not in self._batch_data:
            # Lazy initialization for a new key
            self._batch_data[key] = data.clone()
        else:
            # Concatenate the new data along the batch dimension
            self._batch_data[key] = torch.cat([self._batch_data[key], data], dim=0)

    def get_batch_data(
        self,
    ) -> Dict[str, TensorFloatN | TensorFloatNx2 | TensorFloatNx3]:
        """Provides access to batch data"""
        return self._batch_data

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"train/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"train/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def save_epoch_data(self, name: str, step: int):
        mat_filename = f"{name}_{step}.mat"
        mat_filepath = os.path.join(self.directory, mat_filename)
        batch_data = self.get_batch_data()
        batch_data_cpu = {
            key: value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch_data.items()
        }
        savemat(mat_filepath, batch_data_cpu)
