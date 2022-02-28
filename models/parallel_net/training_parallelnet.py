from data_processing.transform import data_transform
from models.parallel_net.plotting import plotting
from parallel_net import ParallelNet
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.train_loop.train_loop import train_model
from torchvision import datasets

if __name__ == "__main__":
    data_transforms = data_transform()
    data_dir = './data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=2)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    parallel_net = ParallelNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(parallel_net.parameters(), lr=3e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    use_gpu = torch.cuda.is_available()
    model_parallel, losses = train_model(parallel_net, loss_fn, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)
    plotting(losses)

