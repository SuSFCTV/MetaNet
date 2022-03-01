from data_processing.transform import data_transform
from models.cnn.cnn_finetune import Cnn
import torch.nn as nn
import torch.optim as optim
from models.train_loop.train_loop import train_model
from torch.optim import lr_scheduler
from models.cnn.plot_result_cnn import plotting


def training_cnn_finetune():
    model_finetune = Cnn()
    data_transforms, dataloaders, dataset_sizes = data_transform()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_finetune.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    meta_extractor, losses = train_model(model_finetune, loss_fn, optimizer, exp_lr_scheduler, dataloaders,
                                         dataset_sizes, num_epochs=25)
    plotting(losses)

