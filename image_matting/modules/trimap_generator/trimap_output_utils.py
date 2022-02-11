import os

import cv2

SUFFIX = "_trimap.png"


def save_trimap_output(trimap_image, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_path = f'{output_dir}/{filename}_{SUFFIX}'
    cv2.imwrite(image_path, trimap_image)

    return image_path
