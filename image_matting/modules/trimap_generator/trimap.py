import cv2.cv2
import numpy as np
from cv2 import cv2


def generate_trimap(image):
    erosion = _erode(image)
    dilation = _dilate(image)

    trimap = np.full(image.shape, 128, dtype=np.uint8)
    trimap[erosion == 255] = 255
    trimap[dilation == 0] = 0

    return trimap


def _erode(image, iterations=10):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=iterations)
    image = np.where(image > 0, 255, image)

    return image


def _dilate(image, iterations=20):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.dilate(image, kernel, iterations=iterations)
    image = np.where(image > 0, 255, image)

    return image
