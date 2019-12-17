import cv2
import numpy as np
import gc


def rotate_image(img, angle):
    height, width = img.shape[:2]
    image_center = (
        width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def crop_image(img, h, w):
    height, width = img.shape[:2]
    center_x, center_y = (int(width / 2), int(width / 2))
    bounds_x = (int(center_x - w / 2), int(center_x + w / 2))
    bounds_y = (int(center_y - h / 2), int(center_y + h / 2))
    if len(img.shape) == 2:
        return img[bounds_y[0]: bounds_y[1], bounds_x[0]:bounds_x[1]]
    
    return img[bounds_y[0]: bounds_y[1], bounds_x[0]:bounds_x[1], :]


def mirror_image(img, n):
    """ Mirror the image into a 3x3 matrix of images
    
    :param img:
    :param n
    :return:
    """
    if len(img.shape) == 3:
        return np.pad(img, ((n, n), (n, n), (0, 0)), mode="symmetric")
    else:
        return np.pad(img, ((n, n), (n, n)), mode="symmetric")


def rotate_and_crop(img, angle, padding, patch_size):
    img = mirror_image(img, padding)
    img = rotate_image(img, angle)
    img = crop_image(img, patch_size, patch_size)
    gc.collect()
    return img


def flip(image, option_value):
    if option_value == 0:  # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:  # Horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:  # Both
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    return image
