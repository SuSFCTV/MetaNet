from models.parallel_net.class_identity import Identity
import torch
import torch.nn as nn
from torchvision import models


class ParallelNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_pretrained = models.resnet18(pretrained=True)
        self.model_pretrained.fc = Identity()  # костыль
        for param in self.model_pretrained.parameters():  # замораживаю веса предобученной модели
            param.requires_grad = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(1088, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2 = self.model_pretrained(x.clone())
        x_1 = self.features(x.clone())
        x_1 = self.avgpool(x_1)
        x_1 = torch.flatten(x_1, 1)

        x_1 = torch.cat((x_1, x_2), 1)
        x_1 = self.classifier(x_1)
        return x_1
