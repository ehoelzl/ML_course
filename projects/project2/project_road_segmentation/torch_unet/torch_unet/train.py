import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_unet.dataset import BasicDataset
from torch_unet.evaluation import eval_net
from torch_unet.unet import UNet
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DATADIR = "../../Datasets/training/"
IMAGE_DIR = DATADIR + "images/"
MASK_DIR = DATADIR + "groundtruth/"
MASK_THRESHOLD = 0.25

val_percent = 0.2
batch_size = 1
lr = 0.001
img_scale = 1
epochs = 5


def main():
    dataset = BasicDataset(IMAGE_DIR, MASK_DIR, mask_treshold=MASK_THRESHOLD)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = UNet(n_channels=3, n_classes=1)
    net.to(device=device)
    
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting")
    for epoch in range(epochs):
        net.train()  # Sets module in training mode
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                print("loaded")
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                print("done")
                try:
                    masks_pred = net(imgs)  # Make predictions
                    print("lala")
                except Exception as e :
                    logger.info(e)
                    raise e
                
                try:
                    loss = criterion(masks_pred, true_masks)  # Evaluate loss
                except Exception as e:
                    print("lol")
                    raise e
                print("caca")
                epoch_loss += loss.item()  # Add loss to epoch
                writer.add_scalar('Loss/train', loss.item(), global_step)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                print("lol")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("lala")
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)
                    
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #         logging.info('Created checkpoint directory')
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        #     logging.info(f'Checkpoint {epoch + 1} saved !')
    
    writer.close()


if __name__ == "__main__":
    main()
