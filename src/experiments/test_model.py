from typing import Any, Callable, Dict, Optional

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("test-experiment")
        parser.add_argument("--learning-rate", type=float, default=0.0001)
        parser.add_argument("--batch-size", type=int, default=8)

        parser.add_argument("--max-epochs", type=int, default=2)
        parser.add_argument("--limit-train-batches", type=int, default=51)
        parser.add_argument("--limit-val-batches", type=int, default=51)
        return parent_parser

    def __init__(
        self,
        **kwargs: Optional[Any],
    ):
        super().__init__()

        self.learning_rate = kwargs.get("learning_rate")
        self.batch_size = kwargs.get("batch_size")

        self.metrics: Dict[str, Callable] = kwargs.get(
            "metrics",
            {
                "accuracy": pl_metrics.classification.Accuracy(),
            },
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.data_module = IWildCamDataModule(
            batch_size=self.batch_size, root_dir="data"
        )
        self.model = TestModel(self.data_module.dataset.n_classes)

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)
