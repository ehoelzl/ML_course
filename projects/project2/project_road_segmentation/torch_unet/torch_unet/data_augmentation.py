import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


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


def flip(image, option_value):
    if option_value == 0:  # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:  # Horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:  # Both
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    return image


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
