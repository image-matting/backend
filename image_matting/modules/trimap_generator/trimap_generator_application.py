import argparse
from pathlib import Path

from cv2 import cv2

from trimap import generate_trimap
from trimap_output_utils import save_trimap_output


def main():
    args = parse_args()
    image_path = args.image
    output_directory_path = args.output

    image_path = Path(image_path)
    if not image_path.is_file():
        raise RuntimeError(f'The provided image path "{image_path}" does not exist!')

    image_filename = image_path.stem

    saliency_image_path = image_path.as_posix()
    trimap_image = generate_trimap(saliency_image_path, kernel_size=3, iterations=20)
    save_trimap_output(trimap_image, image_filename, output_directory_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Trimap Generator Application')
    parser.add_argument('-i', '--image', required=True, type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, default='.', type=str, help='path to output directory')
    return parser.parse_args()


if __name__ == "__main__":
    main()
