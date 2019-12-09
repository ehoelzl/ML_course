import logging
import cv2
import torch
from torch_unet.dataset import TestSet
from torch_unet.unet import UNet
from torch.utils.data import DataLoader
import matplotlib.image as mpimg
import click
from tqdm import tqdm
TEST_SET = "../Datasets/test_set_images/"
DEST_DIR = "../predictions/"


@click.command()
@click.option("--model-path")
@click.option("--model-depth", )
@click.option("--padding", is_flag=True)
@click.option("--batch-norm", is_flag=True)
def main(model_path, model_depth, padding=True, batch_norm=False):
    test_set = TestSet(TEST_SET, 400 / 608)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net = UNet(n_channels=3, n_classes=1, depth=int(model_depth), padding=padding, batch_norm=batch_norm)
    
    net.to(device=device)
    
    net.load_state_dict(state_dict=torch.load(model_path, map_location=device))
    
    for b in tqdm(test_loader, desc="Predicting"):
        preds = net(b['image'])
        
        img = (torch.sigmoid(preds)[0][0].detach().numpy() > 0.5) * 1
        img = cv2.resize(img.astype('uint8'), (608, 608), interpolation=cv2.INTER_LINEAR)
        mpimg.imsave(DEST_DIR + b['id'][0] + ".png", img)


if __name__ == "__main__":
    main()
