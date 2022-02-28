import torch
import torch.nn as nn
from torchvision import models


class MetaExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_extractor = models.resnet18(pretrained=True)
        for param in self.model_extractor.parameters():  # замораживаю веса предобученной модели
            param.requires_grad = False
        self.model_extractor.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_extractor(x)
        return x
