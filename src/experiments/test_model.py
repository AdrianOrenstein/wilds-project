from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning.metrics as pl_metrics
from src.datasets.iwildcam import IWildCamDataModule
from src.experiments.base import BaseExperiment
from src.models.test_model import TestModel
import torch


class ClassificationExperiment(BaseExperiment):
    NAME = "test-experiment"
    TAGS = {
        "MLFLOW_RUN_NAME": NAME,
        "dataset": "iwildcam",
        "algorithm": "test",
        "model": "TestModel",
    }

    def __init__(
        self,
        **kwargs: Optional[Any],
    ):
        super().__init__()

        self.learning_rate: float = 0.001
        self.batch_size: int = 32

        self.metrics: Dict[str, Callable] = kwargs.get(
            "metrics",
            {
                "accuracy": pl_metrics.classification.Accuracy(),
            },
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.data_module = IWildCamDataModule(batch_size=self.batch_size)
        self.model = TestModel(self.data_module.dataset.n_classes)

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        x, y, metadata = batch

        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        return x, y, y_hat, loss

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        _, y, y_hat, loss = self.step(batch)
        self.log("1_train/train_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="1_train", prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        _, y, y_hat, loss = self.step(batch)
        self.log("2_val/val_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="2_val")

    def test_step(self, batch, batch_idx) -> None:
        _, y, y_hat, loss = self.step(batch)
        self.log("3_test/test_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="3_test")

    def log_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str, prog_bar: bool = False
    ) -> None:

        for metric_name, metric_function in self.metrics.items():
            if len(y.unique()) != 1:
                self.log(
                    f"{prefix}/{metric_name}",
                    metric_function(torch.argmax(y_hat, dim=1).cpu(), y.cpu()),
                    prog_bar=prog_bar,
                )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-06)

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.1, min_lr=1e-6, verbose=True
        )

        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "2_val/val_loss",
            "interval": "epoch",
            "frequency": 2,
            "strict": True,
        }
