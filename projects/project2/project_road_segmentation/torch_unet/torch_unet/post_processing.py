import cv2
import numpy as np


def post_process_prediction(img):
    img_post = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=np.ones((4, 4), np.uint8))
    img_post = cv2.morphologyEx(img_post, cv2.MORPH_CLOSE, kernel=np.ones((8, 8), np.uint8))
    dilated = cv2.dilate(img_post, kernel=np.ones((2, 2), np.uint8), iterations=1)
    
    return dilated
