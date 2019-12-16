import logging
import os

import click
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_unet.globals import *
from torch_unet.tools.dataset import TrainingSet
from torch_unet.unet import UNet
from torch_unet.unet.train import train_model


def split_train_val(dataset, val_ratio, batch_size):
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return n_train, train_loader, n_val, val_loader


def get_model_dir(patch_size, step, depth, batch_size, lr, decay, padding, batch_norm,
                  dropout, augmentation, init_filters, leaky):
    name = f"depth{depth}_BS{batch_size}_lr{lr}_PS{patch_size}_ST{step}_WF{init_filters}"
    if padding:
        name += "_padding"
    if batch_norm:
        name += "_batchnorm"
    if augmentation:
        name += "_aug"
    if decay > 0:
        name += f"_decay"
    if dropout > 0:
        name += f"_dropout{dropout}"
    if leaky:
        name += "_leaky"
    
    model_dir = os.path.join(MODELS_DIR, name)
    dir_checkpoint = os.path.join(model_dir, "checkpoints/")
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)
    return dir_checkpoint, name


@click.command()
@click.option("--epochs", default=50)
@click.option("--lr", default=0.001)
@click.option("--decay", is_flag=True)
@click.option("--val-ratio", default=0.2)
@click.option("--batch-size", default=1)
@click.option("--patch-size", default=400)
@click.option("--step", default=None)
@click.option("--depth", default=4)
@click.option("--num-filters", default=16)
@click.option("--padding", is_flag=True)
@click.option("--batch-norm", is_flag=True)
@click.option("--dropout", default=0.)
@click.option("--leaky", is_flag=True)
@click.option("--augmentation", is_flag=True)
def train(epochs, lr, decay, val_ratio, batch_size, patch_size, step, depth, num_filters, padding, batch_norm, dropout,
          leaky, augmentation):
    dir_checkpoint, name = get_model_dir(patch_size, step, depth, batch_size, lr, decay, padding, batch_norm, dropout,
                                         augmentation, num_filters, leaky)
    
    dataset = TrainingSet(IMAGE_DIR, MASK_DIR, mask_threshold=MASK_THRESHOLD,
                          rotation_angles=ROTATION_ANGLES if augmentation else None, patch_size=patch_size,
                          step=step if step is None else int(step))
    
    n_train, train_loader, n_val, val_loader = split_train_val(dataset, val_ratio, batch_size)
    
    writer = SummaryWriter(comment=name)
    
    net = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=depth, init_filters=num_filters, padding=padding,
               batch_norm=batch_norm, dropout=dropout, leaky=leaky)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    lr_scheduler = None
    if decay:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    train_model(epochs, criterion, optimizer, lr_scheduler, net, train_loader, val_loader, dir_checkpoint, logger, n_train,
                n_val, batch_size, writer, val_ratio)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    train()
