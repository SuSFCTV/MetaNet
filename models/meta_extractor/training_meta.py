from data_processing.transform import data_transform
from models.meta_extractor.metaextractor_net import MetaExtractor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.train_loop.train_loop import train_model
from models.meta_extractor.plot_result_meta import plotting


def training_meta_extractor():
    meta_extractor = MetaExtractor()
    data_transforms, dataloaders, dataset_sizes = data_transform()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_extractor.model_extractor.fc.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    meta_extractor, losses = train_model(meta_extractor, loss_fn, optimizer, exp_lr_scheduler, dataloaders,
                                         dataset_sizes, num_epochs=25)
    plotting(losses)
