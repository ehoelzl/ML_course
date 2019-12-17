import gc
import logging

import click
import matplotlib.image as mpimg
import torch
from torch_unet.globals import *
from torch_unet.tools.dataset import TestSet
from torch_unet.unet import UNet, predict_full_image
from tqdm import tqdm


@click.command()
@click.option("--model-path")
@click.option("--model-depth")
@click.option("--padding", is_flag=True)
@click.option("--num-filters", default=16)
@click.option("--batch-norm", is_flag=True)
@click.option("--dropout", default=0.)
@click.option("--leaky", is_flag=True)
@click.option("--model-path-2", default=None)
def main(model_path, model_depth, padding, num_filters, batch_norm, dropout, leaky, model_path_2):
    test_set = TestSet(TEST_SET)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net_1 = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=int(model_depth), padding=padding,
                 batch_norm=batch_norm,
                 init_filters=num_filters, dropout=dropout, leaky=leaky)
    
    net_1.to(device=device)
    
    net_1.load_state_dict(state_dict=torch.load(model_path, map_location=device))
    
    if model_path_2 is not None:
        net_2 = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=int(model_depth), padding=padding,
                     batch_norm=batch_norm,
                     init_filters=num_filters, dropout=dropout, leaky=leaky)
        
        net_2.to(device=device)
        
        net_2.load_state_dict(state_dict=torch.load(model_path_2, map_location=device))
        net_2.eval()

    net_1.eval()
    for i in tqdm(range(len(test_set)), desc="Predicting"):
        img = test_set.get_raw_image(i)
        idx = test_set[i]['id']
        prediction_1 = predict_full_image(net_1, img, device)[0]
        img = prediction_1
        if model_path_2 is not None:
            prediction_2 = predict_full_image(net_2, img, device)[0]
            img = prediction_1 * 0.5 + prediction_2 * 0.5
        mpimg.imsave(DEST_DIR + idx + ".png", img)
        gc.collect()


if __name__ == "__main__":
    main()
