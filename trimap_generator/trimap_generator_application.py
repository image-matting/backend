import argparse
from pathlib import Path

from cv2 import cv2

from trimap import generate_trimap


def parse_args():
    parser = argparse.ArgumentParser(description='Trimap Generator Application')
    parser.add_argument('-i', '--image', required=True, type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, default='.', type=str, help='path to output directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image
    output_directory_path = args.output

    filename = Path(image_path).stem

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    trimap_image = generate_trimap(image)

    cv2.imwrite(f'{output_directory_path}/{filename}_trimap.png', trimap_image)
