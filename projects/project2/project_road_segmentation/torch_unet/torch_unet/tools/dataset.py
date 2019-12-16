import logging
from glob import glob
from os import listdir
from os.path import splitext

import matplotlib.image as mpimg
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_unet.pre_processing import get_image_patches
from torch_unet.pre_processing.image_utils import rotate_and_crop
from torch_unet.utils import show_side_by_side


class TrainingSet(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_threshold, patch_size=400, step=None,
                 rotation_angles=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_threshold = mask_threshold
        
        # Patch size for patching
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        
        # Open all images
        self.images = [mpimg.imread(glob(self.imgs_dir + idx + '.png')[0]) for idx in self.ids]
        self.masks = [mpimg.imread(glob(self.masks_dir + idx + '.png')[0]) for idx in self.ids]
        
        # Extract patches
        self.images = [patch for img in self.images for patch in get_image_patches(img, patch_size, step)]
        self.masks = [patch for mask in self.masks for patch in get_image_patches(mask, patch_size, step)]
        
        # Enhance with rotations
        if rotation_angles is not None:
            rotation_padding = int(np.ceil(patch_size * (np.sqrt(2) - 1) / 2))
            
            self.images = [rotate_and_crop(img, angle, rotation_padding, patch_size) for img in self.images for angle in
                           rotation_angles]
            self.masks = [rotate_and_crop(img, angle, rotation_padding, patch_size) for img in self.masks for angle in
                          rotation_angles]
        
        self.scale_factor = len(self.images) / len(self.ids)
        logging.info(f"Created dataset from {len(self.ids)} original images, scale factor {self.scale_factor}, "
                     f"patch size {patch_size}, step {step}, rotations {rotation_angles}, "
                     f"total images {self.scale_factor * len(self.ids)}")
    
    def __len__(self):
        return len(self.images)  # * self.scale_factor
    
    def get_real_length(self):
        return len(self.ids)
    
    def preprocess(self, img):
        w, h, _ = img.shape
        
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        
        return img_trans
    
    def preprocess_mask(self, mask):
        img = np.expand_dims(mask, axis=2)
        img = (img >= self.mask_threshold) * 1
        
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        
        return img_trans
    
    def __getitem__(self, i):
        """ Gets the image with index i, and applies data augmentation randomly
        
        :param i: Shape is (channel, height, width) (tensor)
        :return:
        """
        mask = self.preprocess_mask(self.masks[i])
        img = self.preprocess(self.images[i])
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
        return (img >= self.mask_threshold) * 1
    
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
        img_file = glob(self.imgs_dir + idx + ".png")
        assert len(img_file) == 1
        img = mpimg.imread(img_file[0])
        
        img = self.preprocess(img)
        return {'image': torch.from_numpy(img), 'id': idx}
    
    def get_raw_image(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + ".png")
        assert len(img_file) == 1
        img = mpimg.imread(img_file[0])
        return img
    
    def get_image(self, i):
        return torch.as_tensor(self.preprocess(self.get_raw_image(i)))
