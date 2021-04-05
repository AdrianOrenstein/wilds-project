from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch


class BaseExperiment(pl.LightningModule):
    NAME = "base-experiment"
    TAGS = {}

    def __init__(
        self,
        **kwargs: Optional[Any],
    ):
        super().__init__()

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        return NotImplementedError

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return NotImplementedError

    def validation_step(self, batch, batch_idx) -> None:
        return NotImplementedError

    def test_step(self, batch, batch_idx) -> None:
        return NotImplementedError

    def log_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str, prog_bar: bool = False
    ) -> None:
        return NotImplementedError
