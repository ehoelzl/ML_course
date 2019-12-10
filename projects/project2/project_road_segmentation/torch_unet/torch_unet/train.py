import logging
import os

import click
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_unet.dataset import TrainingSet
from torch_unet.evaluation import eval_net
from torch_unet.losses import dice_loss
from torch_unet.unet import UNet
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DATADIR = "../Datasets/training/"
IMAGE_DIR = DATADIR + "images/"
MASK_DIR = DATADIR + "groundtruth/"
MASK_THRESHOLD = 0.25

ROTATION_ANGLES = [0, 15, 30, 45]

models_dir = "./models/"


def split_train_val(dataset, val_ratio, batch_size):
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return n_train, train_loader, n_val, val_loader


@click.command()
@click.option("--num-epochs", default=100)
@click.option("--lr", default=0.001)
@click.option("--batch-size", default=1)
@click.option("--depth", default=5)
@click.option("--padding", is_flag=True)
@click.option("--batch-norm", is_flag=True)
@click.option("--augmentation", is_flag=True)
@click.option("--decay", is_flag=True)
@click.option("--dropout", default=0.)
def train_model(num_epochs=100, lr=0.001,
                val_ratio=0.2, depth=5,
                batch_size=1, img_scale=1,
                padding=False, batch_norm=False,
                augmentation=False, decay=False, dropout=0.):
    name = f"model_depth{depth}_BS{batch_size}_epochs{num_epochs}_lr{lr}"
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
    
    model_dir = os.path.join(models_dir, name)
    dir_checkpoint = os.path.join(model_dir, "checkpoints/")
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)
    
    torch.multiprocessing.set_start_method('spawn')
    
    # Register device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Load net
    net = UNet(n_channels=3, n_classes=1, depth=depth, padding=padding, batch_norm=batch_norm, dropout=dropout)
    net.to(device=device)
    
    # Load training set
    dataset = TrainingSet(IMAGE_DIR, MASK_DIR, mask_threshold=MASK_THRESHOLD,
                          rotation_angles=ROTATION_ANGLES if augmentation else None)
    
    n_train, train_loader, n_val, val_loader = split_train_val(dataset, val_ratio, batch_size)
    
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = None
    if decay:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.99)
        
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        net.train()  # Sets module in training mode
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                optimizer.zero_grad()
                masks_pred = net(imgs)  # Make predictions
                loss = criterion(masks_pred, true_masks)  # Evaluate loss
                batch_loss = loss.item()
                loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss  # Add loss to epoch
                writer.add_scalar('Train/BCE_loss', batch_loss, global_step)
                
                d_loss = dice_loss(torch.sigmoid(masks_pred), true_masks)
                writer.add_scalar('Train/Dice_loss', d_loss, global_step)
                
                pbar.set_postfix(**{'loss (batch)': batch_loss})
                
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score, val_loss = eval_net(net, val_loader, device, n_val)
                    
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    
                    writer.add_scalar('Validation/Dice_coef', val_score, global_step)
                    writer.add_scalar('Validation/Dice_loss', val_loss, global_step)
                    writer.add_images('images', imgs, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred), global_step)
        
                if (global_step + 1) % 2000 == 0:
                    torch.save(net.state_dict(),
                               dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                if scheduler is not None:
                    scheduler.step()
    writer.close()
    
    torch.save(net.state_dict(), os.path.join(model_dir, "final.pth"))


if __name__ == "__main__":
    train_model()
