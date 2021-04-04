from typing import Any, Callable, Dict, Optional, Tuple

from loguru import logger
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
from src.experiments.base import BaseExperiment
import torch
from torch import nn
import torchvision
import wilds


class ClassificationExperiment(BaseExperiment):
    NAME = "test-experiment"

    def __init__(
        self,
        **kwargs: Optional[Any],
    ):
        super().__init__()

        self.metrics: Dict[str, Callable] = kwargs.get(
            "metrics",
            {
                "accuracy": pl_metrics.classification.Accuracy(),
            },
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.data_module = DataModule_iwildcam()
        self.model = SimpleModel(self.data_module.dataset.n_classes)

        self.learning_rate: float = 0.001

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


class SimpleModel(torch.nn.Module):
    NAME = "simple-model"

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.classifier(out)
        return out

    def preprocessing(self, input: torch.Tensor) -> torch.LongTensor:
        return NotImplementedError


class DataModule_iwildcam(pl.LightningDataModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.dataset = wilds.get_dataset(dataset="iwildcam", download=False)

    def prepare_data(self):
        wilds.get_dataset(dataset="iwildcam", download=True)

    def setup(self, stage=None):
        self.train_dataset = self.dataset.get_subset(
            "train",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((448, 448)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        self.val_dataset = self.dataset.get_subset(
            "val",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((448, 448)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=8,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=8,
            num_workers=4,
            pin_memory=True,
        )

    # def test_dataloader(self):
    #     test_split = Dataset(...)
    #     return DataLoader(test_split)
