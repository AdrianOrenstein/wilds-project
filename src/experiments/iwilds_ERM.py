from typing import Any, Callable, Dict, Optional

from src.datasets.iwildcam import IWildCamDataModule
from src.experiments.base import BaseExperiment
import torch
import torch.nn as nn
import torchvision


class ClassificationExperiment(BaseExperiment):
    """
    Relevant hyperparameters:
        https://github.com/p-lambda/wilds/blob/main/examples/configs/datasets.py#L136

    """

    NAME = "iwilds-ResNet50-ERM"
    TAGS = {
        "MLFLOW_RUN_NAME": NAME,
        "dataset": "iwildcam",
        "algorithm": "ERM",
        "model": "torchvision.models.resnet50",
    }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("iwilds-ResNet50-ERM")
        parser.add_argument("--learning_rate", type=float, default=0.00003)
        parser.add_argument("--batch_size", type=int, default=8)

        parser.add_argument("--max-epochs", type=int, default=12)
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
                # import pytorch_lightning.metrics as pl_metrics
                # "accuracy": pl_metrics.classification.Accuracy(),
            },
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.data_module = IWildCamDataModule(
            batch_size=self.batch_size, root_dir="data"
        )
        self.model = self.get_resnet_50(self.data_module.dataset.n_classes)

    def get_resnet_50(self, num_classes: int):
        model = torchvision.models.resnet50(pretrained=True)
        # change output classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        return model

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)
