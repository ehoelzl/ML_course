import torch
from torch_unet.pre_processing.patch import get_image_patches, merge_patches


def predict_full_image(net, img, patch_size, step, output_size, device):
    patches = get_image_patches(img, patch_size, None if step is None else int(step))
    
    patches = patches.transpose((0, 3, 1, 2))  # Transpose to get (n, c, h, w)
    patch_predictions = net(torch.from_numpy(patches).to(device=device, dtype=torch.float32))
    patch_predictions = torch.sigmoid(patch_predictions.squeeze(1)).cpu().detach().numpy()  # remove the channel dimension
    merged = merge_patches(patch_predictions, None if step is None else int(step), output_size)
    return merged
