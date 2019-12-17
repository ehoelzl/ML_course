import numpy as np
from skimage.util import view_as_windows


def flatten_patches(patches):
    p_h, p_w = patches.shape[:2]
    new_shape = (p_h * p_w,) + patches.shape[2:]
    return patches.reshape(new_shape)


def extract_patches_with_step(img, patch_size, step):
    if len(img.shape) == 2:  # single channel
        shape = (patch_size, patch_size)
    elif len(img.shape) == 3:  # RGB Channel
        shape = (patch_size, patch_size, 3)
    else:
        raise ValueError("Image must be of dimension 2 or 3")
    
    overlapped_patches = view_as_windows(img, shape, step)
    if len(img.shape) == 3:
        overlapped_patches = overlapped_patches.squeeze(2)  # Remove additional dimension
    
    return flatten_patches(overlapped_patches)


def extract_non_overlapping_patches(img, patch_size):
    h = img.shape[0]
    assert h % patch_size == 0, f"Cannot extract patch for img shape {h}x{h} patch_size={patch_size}"
    
    return extract_patches_with_step(img, patch_size, patch_size)


def get_image_patches(img, patch_size, step=None):
    if step is None:
        return extract_non_overlapping_patches(img, patch_size)
    else:
        return extract_patches_with_step(img, patch_size, step)
#
#
# def patches_per_image(img_shape, patch_size, step=None):
#     dummy_img = np.zeros(img_shape)
#
#     if step is None:
#         patches = extract_non_overlapping_patches(dummy_img, patch_size)
#     else:
#         patches = extract_patches_with_step(dummy_img, patch_size, step)
#
#     return patches.shape[0]


# def get_image_patch(img, patch_size, i, step=None):
#     return get_image_patches(img, patch_size, step)[i]


def merge_patches(patches, step, shape):
    if step is None:
        return patches[0]
    num_patches, patch_size, _ = patches.shape
    patches_per_side = np.sqrt(num_patches)
    assert patches_per_side % 1 == 0
    patches_per_side = int(patches_per_side)
    
    patches = patches.reshape((patches_per_side, patches_per_side, patch_size, patch_size))
    merged = np.zeros((shape, shape))
    counts = np.zeros((shape, shape))
    for i in range(0, patches_per_side):
        for j in range(0, patches_per_side):
            merged[i * step: (i * step) + patch_size, j * step:(j * step) + patch_size] += patches[i][j]
            counts[i * step: (i * step) + patch_size, j * step:(j * step) + patch_size] += 1
    if np.any(counts == 0):
        print("Some are 0")
    return merged / counts
