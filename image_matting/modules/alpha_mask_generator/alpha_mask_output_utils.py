import os

import cv2

SUFFIX = "_alpha.png"


def save_alpha_mask_output(alpha_mask_image, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_path = f'{output_dir}/{filename}_{SUFFIX}'
    cv2.imwrite(image_path, alpha_mask_image)

    return image_path
