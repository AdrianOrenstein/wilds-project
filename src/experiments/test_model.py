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
        "PYLIGHTNING_TUNE": True,
        "dataset": "iwildcam",
        "algorithm": "test",
        "model": "TestModel",
    }
    TRAINING_KWARGS = {
        "max_epochs": 2,
        "limit_train_batches": 0.1,
        "limit_val_batches": 0.1,
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
        self.data_module = IWildCamDataModule(
            batch_size=self.batch_size, root_dir="data"
        )
        self.model = TestModel(self.data_module.dataset.n_classes)

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)
