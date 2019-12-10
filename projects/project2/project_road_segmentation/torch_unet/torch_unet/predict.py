import logging

import click
import matplotlib.image as mpimg
import torch
from torch.utils.data import DataLoader
from torch_unet.dataset import TestSet
from torch_unet.image_utils import combine_patches, get_image_patches
from torch_unet.unet import UNet
from tqdm import tqdm
import gc
TEST_SET = "../Datasets/test_set_images/"
DEST_DIR = "../predictions/"


@click.command()
@click.option("--model-path")
@click.option("--model-depth", )
@click.option("--padding", is_flag=True)
@click.option("--batch-norm", is_flag=True)
@click.option("--threshold", default=0.5)
def main(model_path, model_depth, padding=True, batch_norm=False, threshold=0.5):
    test_set = TestSet(TEST_SET)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net = UNet(n_channels=3, n_classes=1, depth=int(model_depth), padding=padding, batch_norm=batch_norm)
    
    net.to(device=device)
    
    net.load_state_dict(state_dict=torch.load(model_path, map_location=device))
    
    for b in tqdm(test_loader, desc="Predicting"):
        patches = get_image_patches(b['image'], 400)
        for idx, im_patches in enumerate(patches):
            pred_patches = torch.sigmoid(net(im_patches))
        
            prediction = combine_patches(pred_patches, 608)
            img = ((prediction.detach().numpy() > threshold) * 1)[0]
            mpimg.imsave(DEST_DIR + b['id'][idx] + ".png", img)
        gc.collect()


if __name__ == "__main__":
    main()
