from cv2 import cv2
import numpy as np


def get_saliency_map(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    c = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:, :, 0] ** 2 + c[:, :, 1] ** 2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3, 3)))

    c[:, :, 0] = c[:, :, 0] * spectralResidual / mag
    c[:, :, 1] = c[:, :, 1] * spectralResidual / mag
    c = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = c[:, :, 0] ** 2 + c[:, :, 1] ** 2
    cv2.normalize(cv2.GaussianBlur(mag, (9, 9), 3, 3), mag, 0., 1., cv2.NORM_MINMAX)

    return mag
