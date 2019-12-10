import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch_unet.image_utils import mirror_image, rotate_image


def normalize_image(img, _min=0, _max=1):
    image_new = (img - np.min(img)) * (_max - _min) / (np.max(img) - np.min(img)) + _min
    return image_new


def gaussian_noise(img, mean, var):
    gaus_noise = np.random.normal(mean, var, img.shape) / 255
    image_n = img + gaus_noise
    return image_n


def uniform_noise(img, _min, _max):
    uni_noise = np.random.uniform(_min, _max, img.shape) / 255
    image_n = img + uni_noise
    return image_n


# def rotate_and_crop(img, angle):
#     h, w = img.shape[:2]
#     mirrored = mirror_image(img)
#     rotated = rotate_image(mirrored, angle)
#     cropped = crop_3d_image(rotated, h, w)
#     return cropped


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    seed = np.random.get_state()
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    if len(image.shape) == 3:
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    else:
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape), seed


def augment_image(img, mask):
    # Rotate by random angle
    rotation = np.random.choice([0, 90, 180, 270])
    if rotation > 0:
        img, mask = rotate_image(img, rotation), rotate_image(mask, rotation)
    
    random_rot = np.random.choice([0, 15, 30, 45, 60, 75])
    if random_rot > 0:
        img, mask = rotate_and_crop(img, random_rot), rotate_and_crop(mask, random_rot)
    return img, mask
