import gc
import logging

import click
import matplotlib.image as mpimg
import torch
from torch_unet.globals import *
from torch_unet.tools.dataset import TestSet
from torch_unet.unet import UNet
from unet import predict_full_image
from tqdm import tqdm

TEST_SET = "../Datasets/test_set_images/"
DEST_DIR = "../predictions/"


@click.command()
@click.option("--model-path")
@click.option("--model-depth")
@click.option("--padding", is_flag=True)
@click.option("--num-filters", default=4)
@click.option("--batch-norm", is_flag=True)
@click.option("--dropout", default=0.)
@click.option("--threshold", default=0.5)
def main(model_path, model_depth, padding, num_filters, batch_norm, dropout):
    test_set = TestSet(TEST_SET)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=int(model_depth), padding=padding, batch_norm=batch_norm,
               init_filters=num_filters, dropout=dropout)
    
    net.to(device=device)
    
    net.load_state_dict(state_dict=torch.load(model_path, map_location=device))
    
    net.eval()
    for i in tqdm(range(len(test_set)), desc="Predicting"):
        img = test_set.get_raw_image(i)
        idx = test_set[i]['id']
        prediction = predict_full_image(net, img, device)
        img = prediction  # (prediction > threshold) * 1
        mpimg.imsave(DEST_DIR + idx + ".png", img)
        gc.collect()


if __name__ == "__main__":
    main()
