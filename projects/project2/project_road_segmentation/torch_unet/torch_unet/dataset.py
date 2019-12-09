import logging
from glob import glob
from os import listdir
from os.path import splitext

import cv2
import matplotlib.image as mpimg
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_unet.data_augmentation import augment_image
from torch_unet.utils import show_side_by_side


class TrainingSet(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_treshold, scale=1, augmentation=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_threshold = mask_treshold
        self.augmentation = augmentation
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
        """ Gets the image with index i, and applies data augmentation randomly
        
        :param i: Shape is (channel, height, width) (tensor)
        :return:
        """
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = mpimg.imread(mask_file[0])
        img = mpimg.imread(img_file[0])
        
        if self.augmentation:
            img, mask = augment_image(img, mask)
        mask = self.preprocess_mask(mask)
        img = self.preprocess(img)
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
    
    def get_raw_image(self, i):
        """ Returns the raw image as a numpy array
        
        :param i:
        :return:
        """
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        img = mpimg.imread(img_file[0])
        return img
    
    def get_raw_mask(self, i):
        """ Returns the mask as a numpy array
        
        :param i:
        :return:
        """
        idx = self.ids[i]
        img_file = glob(self.masks_dir + idx + '*')
        img = mpimg.imread(img_file[0])
        return img
    
    def show_image(self, i, augment=False):
        """ Shows the image and its mask side by side
        
        :param i:
        :return:
        """
        img, mask = self.get_raw_image(i), self.get_raw_mask(i)
        if augment:
            img, mask = augment_image(img, mask)
        show_side_by_side(img, mask)


class TestSet(Dataset):
    
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    
    def __len__(self):
        return len(self.ids)
    
    def scale_image(self, img):
        w, h, c = img.shape
        if self.scale != 1:
            newW, newH = int(self.scale * w), int(self.scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
        return img
    
    def preprocess(self, img):
        img = self.scale_image(img)
        img = img.transpose((2, 0, 1))
        return img
    
    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        
        img = mpimg.imread(img_file[0])
        
        img = self.preprocess(img)
        return {'image': torch.from_numpy(img), 'id': idx}
    
    def get_raw_image(self, i, scale=False):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        img = mpimg.imread(img_file[0])
        if scale:
            img = self.scale_image(img)
        return img
    
    def get_image(self, i):
        return torch.as_tensor(self.preprocess(self.get_raw_image(i)))
