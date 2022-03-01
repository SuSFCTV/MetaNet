from torchvision import models
import torch
import torch.nn as nn


class Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_finetune = models.resnet18(pretrained=True)
        self.model_finetune.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_extractor(x)
        return x
