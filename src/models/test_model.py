import torch
import torch.nn as nn


class TestModel(torch.nn.Module):
    NAME = "test-model"

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
