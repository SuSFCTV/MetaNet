from data_processing.transform import data_transform
from models.parallel_net.plot_result_parallel import plotting
from models.parallel_net.parallel_net import ParallelNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.train_loop.train_loop import train_model


def training_parallel_net():
    data_transforms, dataloaders, dataset_sizes = data_transform()

    parallel_net = ParallelNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(parallel_net.parameters(), lr=3e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    use_gpu = torch.cuda.is_available()
    model_parallel, losses = train_model(parallel_net, loss_fn, optimizer_ft, exp_lr_scheduler, dataloaders,
                                         dataset_sizes, num_epochs=25)
    plotting(losses)
