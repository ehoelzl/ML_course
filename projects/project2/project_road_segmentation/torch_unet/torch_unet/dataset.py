from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import matplotlib.image as mpimg


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_treshold, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_threshold = mask_treshold
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    
    def __len__(self):
        return len(self.ids)
    
    def preprocess(self, img):
        w, h, _ = img.shape
        if self.scale != 1:
            newW, newH = int(self.scale * w), int(self.scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img = img.resize((newW, newH))
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if np.max(img_trans) > 1:
            img_trans = img_trans / 255
        
        return img_trans
    
    def preprocess_mask(self, mask):
        img = np.expand_dims(mask, axis=2)
        img = (img >= self.mask_threshold) * 1
        
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if np.max(img_trans) > 1:
            img_trans = img_trans / 255
        
        return img_trans
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = mpimg.imread(mask_file[0])
        img = mpimg.imread(img_file[0])
        
        mask = self.preprocess_mask(mask)
        img = self.preprocess(img)
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
