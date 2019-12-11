import torch
from tqdm import tqdm
import numpy as np
from tools.losses import dice_coeff, dice_loss
from torch_unet.unet.predict import predict_full_image
from torch_unet.globals import *


def eval_net(net, loader, device, n_val):
    """Evaluation with the dice coefficient and Dice loss"""
    tot = 0
    loss = 0
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            mask_pred = net(imgs)
            for true_mask, pred in zip(true_masks, mask_pred):
                pred = ((torch.sigmoid(pred) > 0.3) * 1).float()
                
                tot += dice_coeff(pred, true_mask).item()
                loss += dice_loss(pred, true_mask).item()
            pbar.update(imgs.shape[0])
    
    return tot / n_val, loss / n_val


def eval_net_full(net, loader, patch_size, step, device, ratio):
    tot = 0
    loss = 0
    dataset = loader.dataset.dataset
    num_images = int(ratio * dataset.get_real_length())
    image_idx = np.random.choice(np.arange(0, dataset.get_real_length()), num_images)
    
    with tqdm(total=num_images, desc="Full Validation round", unit="img", leave=False) as pbar:
        for i in image_idx:
            img, true_mask = dataset.get_raw_image(i), dataset.get_raw_mask(i)
            prediction = predict_full_image(net, img, patch_size, step, TRAIN_SIZE, device)
            prediction = torch.from_numpy(np.expand_dims((prediction > 0.3) * 1, 0)).to(device=device).float()
            true_mask = torch.from_numpy(np.expand_dims(true_mask, 0)).to(device=device).float()
            
            tot += dice_coeff(prediction, true_mask).item()
            loss += dice_loss(prediction, true_mask).item()
            pbar.update(i)
    return tot / num_images, loss / num_images, img, true_mask, prediction
