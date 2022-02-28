from data_processing.transform import data_transform
from metaextractor_net import MetaExtractor
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.train_loop.train_loop import train_model
from torchvision import datasets

if __name__ == "__main__":
    meta_extractor = MetaExtractor()
    data_transforms = data_transform()
    data_dir = './data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=2)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_extractor.model_extractor.fc.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    meta_extractor, losses = train_model(meta_extractor, loss_fn, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)