import torch
from tqdm import tqdm

from torch_unet.losses import dice_coeff, dice_loss


def eval_net(net, loader, device, n_val):
    """Evaluation with the dice coefficient and Dice loss"""
    net.eval()
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
                tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                loss += dice_loss(pred, true_masks)
            pbar.update(imgs.shape[0])
    
    return tot / n_val, loss / n_val
