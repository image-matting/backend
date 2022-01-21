import cv2.cv2
import numpy as np
from cv2 import cv2


def generate_trimap(image, kernel_size, iterations):
    erosion = _erode(image, kernel_size, iterations)
    dilation = _dilate(image, kernel_size, iterations)

    trimap = np.full(image.shape, 128, dtype=np.uint8)
    trimap[erosion == 255] = 255
    trimap[dilation == 0] = 0

    return trimap


def _erode(image, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = cv2.erode(image, kernel, iterations=iterations)
    image[image > 0] = 255

    return image


def _dilate(image, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = cv2.dilate(image, kernel, iterations=iterations)
    image[image > 0] = 255

    return image
