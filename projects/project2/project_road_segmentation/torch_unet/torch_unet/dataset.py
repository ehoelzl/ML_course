import logging
from glob import glob
from os import listdir
from os.path import splitext

import matplotlib.image as mpimg
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_unet.image_utils import crop_image, mirror_image, rotate_image
from torch_unet.utils import show_side_by_side


class TrainingSet(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_threshold, image_height=400, augmentation=False, rotation_angles=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_threshold = mask_threshold
        
        self.augmentation = augmentation
        self.rotation_angles = [0] if rotation_angles is None else rotation_angles
        self.num_rotations = len(self.rotation_angles)
        self.image_height = image_height
        self.padding = int(np.ceil(self.image_height * (np.sqrt(2) - 1) / 2))
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    
    def __len__(self):
        return len(self.ids) * len(self.rotation_angles)
    
    def preprocess(self, img):
        w, h, _ = img.shape
        
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
    
    def get_rotated_img_and_mask(self, img, mask, i):
        rotation = self.rotation_angles[i % self.num_rotations]
        if rotation == 0:
            return img, mask
        new_img = rotate_image(mirror_image(img, self.padding), rotation)
        new_mask = rotate_image(mirror_image(mask, self.padding), rotation)
        
        new_img = crop_image(new_img, self.image_height, self.image_height)
        new_mask = crop_image(new_mask, self.image_height, self.image_height)
        return new_img, new_mask
    
    def __getitem__(self, i):
        """ Gets the image with index i, and applies data augmentation randomly
        
        :param i: Shape is (channel, height, width) (tensor)
        :return:
        """
        idx = self.ids[i // self.num_rotations]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = mpimg.imread(mask_file[0])
        img = mpimg.imread(img_file[0])
        
        img, mask = self.get_rotated_img_and_mask(img, mask, i)
        
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
    
    def show_image(self, i):
        """ Shows the image and its mask side by side
        
        :param i:
        :return:
        """
        img, mask = self.get_raw_image(i), self.get_raw_mask(i)
        show_side_by_side(img, mask)


class TestSet(Dataset):
    
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    
    def __len__(self):
        return len(self.ids)
    
    def preprocess(self, img):
        img = img.transpose((2, 0, 1))
        return img
    
    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        
        img = mpimg.imread(img_file[0])
        
        img = self.preprocess(img)
        return {'image': torch.from_numpy(img), 'id': idx}
    
    def get_raw_image(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        img = mpimg.imread(img_file[0])
        return img
    
    def get_image(self, i):
        return torch.as_tensor(self.preprocess(self.get_raw_image(i)))
