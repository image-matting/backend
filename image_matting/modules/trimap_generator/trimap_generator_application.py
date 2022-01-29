import argparse
from pathlib import Path

from cv2 import cv2

from trimap import generate_trimap


def main():
    args = parse_args()
    image_path = args.image
    output_directory_path = args.output

    image_path = Path(image_path)
    if not image_path.is_file():
        raise RuntimeError(f'The provided image path "{image_path}" does not exist!')

    image_filename = image_path.stem

    image_path_str = image_path.as_posix()
    image = cv2.imread(image_path_str, cv2.IMREAD_GRAYSCALE)
    trimap_image = generate_trimap(image, kernel_size=3, iterations=20)

    cv2.imwrite(f'{output_directory_path}/{image_filename}_trimap.png', trimap_image)


def parse_args():
    parser = argparse.ArgumentParser(description='Trimap Generator Application')
    parser.add_argument('-i', '--image', required=True, type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, default='.', type=str, help='path to output directory')
    return parser.parse_args()


if __name__ == "__main__":
    main()
